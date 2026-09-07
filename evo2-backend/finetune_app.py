"""Modal app that runs the fine-tuning pipeline.

Each stage is a separate Modal function so it can be re-run on its own; all
state lives on the ``evo2-finetune-data`` volume. Feature extraction is the only
expensive stage, so it is sharded across containers and skips shards that
already exist, which makes it safe to interrupt and resume.

    modal run finetune_app.py --stage build-datasets
    modal run finetune_app.py --stage estimate
    modal run finetune_app.py --stage extract --shards 40 --max-containers 10
    modal run finetune_app.py --stage merge
    modal run finetune_app.py --stage train
    modal run finetune_app.py --stage publish --run-name run-20260814-101500

See FINETUNING.md for the full runbook.
"""

import modal

from common import VOLUMES, data_volume, evo2_image
from finetune import config

app = modal.App(config.APP_NAME, image=evo2_image)

# Extraction shards should stay small enough that a retry is cheap.
SHARD_TIMEOUT = 24 * 60 * 60

# The GPU both extraction and its benchmark run on, named once so the cost
# estimate can't quote a different card than the one actually billed. An L4's
# 24 GB fits the 7B weights with room for an 8 kb window; a 16 GB card does not.
EXTRACT_GPU = "L4"
# Approximate Modal on-demand rate for EXTRACT_GPU, in USD per hour. Override
# with --gpu-hourly-usd if Modal's pricing has moved.
EXTRACT_GPU_HOURLY_USD = 0.80


# --- Stage 1: datasets -----------------------------------------------------


@app.function(volumes=VOLUMES, timeout=3600, cpu=4.0, memory=16384)
def build_datasets(
    min_review_stars: int = 2,
    max_variants: int = 40_000,
    balance_classes: bool = True,
    force_download: bool = False,
) -> dict:
    """Build the labelled variant table and prepare both reference assemblies."""
    from finetune import data, sequences

    data_volume.reload()

    clinvar_cfg = config.ClinVarConfig(
        min_review_stars=min_review_stars,
        max_variants=max_variants if max_variants > 0 else None,
        balance_classes=balance_classes,
    )

    print("=== ClinVar ===")
    clinvar = data.build_clinvar_dataset(clinvar_cfg, force_download=force_download)

    print("\n=== BRCA1 saturation mutagenesis ===")
    brca1 = data.build_brca1_dataset()

    print("\n=== Splits ===")
    combined = data.assign_splits(clinvar, brca1)
    data.save_dataset(combined)

    print("\n=== Reference genomes ===")
    sequences.ensure_hg38()
    sequences.ensure_hg19_chr17()

    data_volume.commit()
    return {
        "n_variants": int(len(combined)),
        "by_split": combined["split"].value_counts().to_dict(),
    }


# --- Stage 2: feature extraction ------------------------------------------


def _load_shard_frame(start: int, end: int, dataset_name: str):
    """Deterministic slice of the variant table, so shard N is always shard N."""
    from finetune import data

    frame = data.load_dataset(dataset_name)
    frame = frame.sort_values("variant_id").reset_index(drop=True)
    return frame.iloc[start:end]


@app.function(
    gpu=EXTRACT_GPU,
    volumes=VOLUMES,
    timeout=SHARD_TIMEOUT,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
)
def extract_shard(spec: tuple) -> str:
    """Score one shard of variants. ``spec`` is ``(index, start, end, dataset)``."""
    from finetune import features

    shard_index, start, end, dataset_name = spec
    data_volume.reload()

    frame = _load_shard_frame(start, end, dataset_name)
    print(f"Shard {shard_index}: rows [{start}, {end}) -> {len(frame)} variants")

    path = features.extract_shard(frame, shard_index, config.DEFAULT_FEATURE_CONFIG)
    data_volume.commit()
    return path


@app.function(
    gpu=EXTRACT_GPU,
    volumes=VOLUMES,
    timeout=3600,
)
def benchmark_throughput(n_variants: int = 8, dataset_name: str = "variants") -> dict:
    """Time real extraction on a few variants so cost estimates are measured.

    Excludes model load time, which is paid once per container rather than per
    variant, but reports it so the caller can size shards sensibly.
    """
    import time

    from finetune import features
    from finetune.loader import load_evo2
    from finetune.sequences import build_variant_window, open_genome

    data_volume.reload()
    cfg = config.DEFAULT_FEATURE_CONFIG

    load_started = time.time()
    model = load_evo2(config.MODEL_NAME)
    load_seconds = time.time() - load_started
    print(f"Model load: {load_seconds:.1f}s")

    frame = _load_shard_frame(0, n_variants, dataset_name)
    genomes = {}
    timings = []

    for _, row in frame.iterrows():
        assembly = row["assembly"]
        if assembly not in genomes:
            genomes[assembly] = open_genome(assembly)

        ref_window, start = genomes[assembly].window(
            row["chrom"], int(row["pos"]), cfg.window_size
        )
        relative = int(row["pos"]) - 1 - start
        var_window, _ = build_variant_window(
            ref_window, relative, row["alt"], expected_reference=row["ref"]
        )

        started = time.time()
        features.extract_variant_features(model, ref_window, var_window, relative, cfg)
        timings.append(time.time() - started)

    # The first call includes CUDA graph / kernel warm-up.
    steady = timings[1:] or timings
    seconds_per_variant = sum(steady) / len(steady)
    print(f"Steady-state: {seconds_per_variant:.2f}s per variant")
    return {
        "seconds_per_variant": seconds_per_variant,
        "model_load_seconds": load_seconds,
        "window_size": cfg.window_size,
        "n_timed": len(steady),
    }


@app.function(volumes=VOLUMES, timeout=3600, cpu=8.0, memory=65536)
def merge_features() -> str:
    from finetune import features

    data_volume.reload()
    path = features.merge_shards(config.DEFAULT_FEATURE_CONFIG)
    data_volume.commit()
    return path


@app.function(volumes=VOLUMES, timeout=600)
def extraction_status(dataset_name: str = "variants", shards: int = 40) -> dict:
    """Report which shards are already on the volume."""
    import os

    from finetune import data, features

    data_volume.reload()
    total = len(data.load_dataset(dataset_name))
    bounds = features.shard_bounds(total, shards)

    done = [
        i for i in range(len(bounds))
        if os.path.exists(features.shard_path(i, config.DEFAULT_FEATURE_CONFIG))
    ]
    return {
        "total_variants": total,
        "shards": len(bounds),
        "completed": len(done),
        "pending": [i for i in range(len(bounds)) if i not in set(done)],
    }


@app.function(volumes=VOLUMES, timeout=1800)
def repair_part_files(apply: bool = False) -> dict:
    """Reclaim shards stranded as ``shard_N.npz.part.npz`` by an earlier bug.

    Those shards finished every variant and were written successfully; only the
    rename that followed failed, leaving valid archives under a name the merge
    glob deliberately ignores. Renaming them recovers the GPU time already paid
    for instead of extracting it a second time.

    Each candidate is opened and checked for the expected arrays first, so a
    genuinely truncated file is reported rather than promoted to a real shard.
    Defaults to a dry run; pass ``apply`` to make the changes.
    """
    import glob
    import os

    import numpy as np

    from finetune import features

    data_volume.reload()
    directory = features.features_dir(config.DEFAULT_FEATURE_CONFIG)
    expected = {"variant_id", "embedding", "scalar", "label"}

    renamed, skipped = [], []
    for path in sorted(glob.glob(f"{directory}/shard_*.npz.part.npz")):
        target = path[: -len(".part.npz")]
        if os.path.exists(target):
            skipped.append({"path": path, "reason": "target already exists"})
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = expected - set(data.files)
                if missing:
                    skipped.append({"path": path, "reason": f"missing {missing}"})
                    continue
                rows = int(len(data["label"]))
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not abort the sweep
            skipped.append({"path": path, "reason": f"unreadable: {exc}"})
            continue

        if apply:
            os.replace(path, target)
        renamed.append({"shard": os.path.basename(target), "rows": rows})

    if apply and renamed:
        data_volume.commit()

    return {
        "applied": apply,
        "recovered": renamed,
        "recovered_rows": sum(r["rows"] for r in renamed),
        "skipped": skipped,
    }


# --- Stage 3: training -----------------------------------------------------


@app.function(volumes=VOLUMES, timeout=7200, gpu="A10G", cpu=8.0, memory=65536)
def train(
    head: str = "mlp",
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.3,
    max_epochs: int = 100,
    run_name: str = "",
) -> dict:
    """Fit the classifier head on cached features and save a run directory."""
    from finetune import train as train_module

    data_volume.reload()

    train_cfg = config.TrainConfig(
        head=head,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        max_epochs=max_epochs,
    )
    trained = train_module.train_head(
        feature_config=config.DEFAULT_FEATURE_CONFIG,
        train_config=train_cfg,
        run_name=run_name or None,
    )
    data_volume.commit()
    return trained.metrics


@app.function(volumes=VOLUMES, timeout=600)
def publish_run(run_name: str) -> str:
    from finetune import train as train_module

    data_volume.reload()
    target = train_module.publish(run_name)
    data_volume.commit()
    return target


@app.function(volumes=VOLUMES, timeout=600)
def list_runs() -> None:
    from finetune import train as train_module

    data_volume.reload()
    train_module.summarise_runs()


# --- Orchestration ---------------------------------------------------------


@app.local_entrypoint()
def main(
    stage: str = "status",
    shards: int = 40,
    max_containers: int = 10,
    dataset_name: str = "variants",
    min_review_stars: int = 2,
    max_variants: int = 40_000,
    run_name: str = "",
    head: str = "mlp",
    max_epochs: int = 100,
    gpu_hourly_usd: float = EXTRACT_GPU_HOURLY_USD,
    benchmark_variants: int = 8,
):
    """Drive the pipeline. Pick a stage; see the module docstring for examples."""
    if stage == "build-datasets":
        result = build_datasets.remote(
            min_review_stars=min_review_stars,
            max_variants=max_variants,
        )
        print(f"\nDataset built: {result}")

    elif stage in ("repair-shards", "repair-shards-apply"):
        result = repair_part_files.remote(apply=stage.endswith("-apply"))
        for item in result["recovered"]:
            print(f"  {item['shard']}: {item['rows']} rows")
        for item in result["skipped"]:
            print(f"  SKIPPED {item['path']}: {item['reason']}")
        verb = "Recovered" if result["applied"] else "Would recover"
        print(
            f"\n{verb} {len(result['recovered'])} shards "
            f"({result['recovered_rows']:,} rows)"
        )
        if not result["applied"] and result["recovered"]:
            print("Re-run with --stage repair-shards-apply to make the changes.")

    elif stage == "estimate":
        _estimate(shards, max_containers, dataset_name, gpu_hourly_usd, benchmark_variants)

    elif stage == "extract":
        status = extraction_status.remote(dataset_name, shards)
        pending = status["pending"]
        print(
            f"{status['total_variants']} variants in {status['shards']} shards; "
            f"{status['completed']} already done, {len(pending)} to run"
        )
        if not pending:
            print("Nothing to do. Run `--stage merge` next.")
            return

        from finetune import features

        bounds = features.shard_bounds(status["total_variants"], shards)
        specs = [(i, bounds[i][0], bounds[i][1], dataset_name) for i in pending]

        completed = 0
        for path in extract_shard.map(
            specs, order_outputs=False, return_exceptions=True
        ):
            if isinstance(path, Exception):
                print(f"  shard failed: {path}")
            else:
                completed += 1
                print(f"  [{completed}/{len(specs)}] {path}")
        print(f"\nExtraction finished: {completed}/{len(specs)} shards written")

    elif stage == "merge":
        print(f"\nMerged into {merge_features.remote()}")

    elif stage == "train":
        metrics = train.remote(head=head, max_epochs=max_epochs, run_name=run_name)
        print("\n=== Results ===")
        for split, entry in metrics["splits"].items():
            print(
                f"  {split:<10} head AUROC={entry['head']['auroc']:.4f}  "
                f"zero-shot AUROC={entry['zero_shot']['auroc']:.4f}  "
                f"n={entry['head']['n']}"
            )
        print("\nPublish it with:  modal run finetune_app.py --stage publish "
              "--run-name <run>")

    elif stage == "publish":
        if not run_name:
            raise SystemExit("--run-name is required for the publish stage")
        print(f"\nPublished to {publish_run.remote(run_name)}")

    elif stage == "runs":
        list_runs.remote()

    elif stage == "status":
        print(extraction_status.remote(dataset_name, shards))

    else:
        raise SystemExit(
            f"Unknown stage {stage!r}. Expected one of: build-datasets, estimate, "
            "extract, merge, train, publish, runs, status"
        )


def _estimate(
    shards: int,
    max_containers: int,
    dataset_name: str,
    gpu_hourly_usd: float,
    benchmark_variants: int,
):
    """Measure real throughput and project the cost of a full extraction run."""
    status = extraction_status.remote(dataset_name, shards)
    remaining = status["total_variants"] * len(status["pending"]) / max(status["shards"], 1)

    print(f"Timing {benchmark_variants} variants on an {EXTRACT_GPU} ...")
    bench = benchmark_throughput.remote(benchmark_variants, dataset_name)

    per_variant = bench["seconds_per_variant"]
    load = bench["model_load_seconds"]

    gpu_seconds = remaining * per_variant + len(status["pending"]) * load
    gpu_hours = gpu_seconds / 3600
    wall_hours = gpu_hours / max(max_containers, 1)

    print(f"\n--- Extraction estimate (window {bench['window_size']} bp) ---")
    print(f"  measured            {per_variant:.2f} s/variant, {load:.0f} s model load")
    print(f"  variants remaining  {remaining:,.0f} across {len(status['pending'])} shards")
    print(f"  GPU time            {gpu_hours:.1f} {EXTRACT_GPU}-hours")
    print(f"  wall clock          ~{wall_hours:.1f} h at {max_containers} containers")
    print(f"  approx cost         ~${gpu_hours * gpu_hourly_usd:,.2f} "
          f"at ${gpu_hourly_usd:.2f}/{EXTRACT_GPU}-hour")
    print(
        "\n  Cost scales roughly linearly with both variant count and window size:\n"
        "  halving FeatureConfig.window_size to 4096, or lowering "
        "ClinVarConfig.max_variants,\n  halves it. Both are in finetune/config.py."
    )
