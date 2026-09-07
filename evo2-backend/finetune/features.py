"""Evo2 feature extraction for variant effect prediction.

The expensive part of the pipeline. For each SNV we run the reference and the
mutated window through Evo2 once and keep two things from the same forward pass:

* per-token log-probabilities, which give the zero-shot delta-likelihood score
* hidden states from ``blocks.28.mlp.l3``, pooled around the variant

:func:`extract_variant_features` is shared by the training job and the inference
endpoint, so the vector a trained head sees at serving time is built by exactly
the code that produced its training data.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .config import SCALAR_FEATURE_NAMES, FeatureConfig


def token_logprobs(logits, input_ids):
    """Log-probability the model assigned to each observed token.

    Returns a ``(batch, length)`` array where element ``i`` is
    ``log P(input_ids[i] | input_ids[<i])``, i.e. **indexed by sequence
    position, not shifted**. Position 0 is NaN because nothing precedes it.

    Keeping the array aligned to sequence coordinates is what lets the pooling
    code below slice by genomic offset without an off-by-one.
    """
    import numpy as np
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    # predicted[:, i] scores input_ids[:, i + 1]
    gathered = torch.gather(
        logprobs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    batch, length = input_ids.shape
    out = np.full((batch, length), np.nan, dtype=np.float64)
    out[:, 1:] = gathered.float().cpu().numpy()
    return out


def _pool_slice(length: int, centre: int, radius: Optional[int]) -> slice:
    if radius is None:
        return slice(0, length)
    return slice(max(0, centre - radius), min(length, centre + radius + 1))


def _mean_over(values, length: int, centre: int, radius: Optional[int]) -> float:
    import numpy as np

    window = values[_pool_slice(length, centre, radius)]
    if np.all(np.isnan(window)):
        return 0.0
    return float(np.nanmean(window))


def _unwrap(value):
    """Take the tensor out of a ``(tensor, ...)`` return value.

    ``Evo2.forward`` passes through whatever vortex's ``StripedHyena.forward``
    returns, which is ``(logits, inference_params)`` rather than a bare tensor.
    Other versions hand back the tensor directly, so accept both instead of
    depending on which one is installed.
    """
    while isinstance(value, (tuple, list)):
        value = value[0]
    return value


def _forward(model, seqs: Sequence[str], feature_config: FeatureConfig):
    """Run Evo2 once over ``seqs``, returning (logprobs, embeddings)."""
    import torch

    from evo2.scoring import prepare_batch

    input_ids, _ = prepare_batch(list(seqs), model.tokenizer, device="cuda:0")

    with torch.inference_mode():
        logits, embeddings = model.forward(
            input_ids,
            return_embeddings=True,
            layer_names=[feature_config.embedding_layer],
        )

    logits = _unwrap(logits)
    hidden = _unwrap(embeddings[feature_config.embedding_layer])
    if hidden.ndim == 2:  # defensive: some layers emit (length, hidden)
        hidden = hidden.unsqueeze(0)

    lp = token_logprobs(logits, input_ids)
    emb = hidden.float().cpu().numpy()
    del logits, embeddings, hidden, input_ids
    return lp, emb


def extract_variant_features(
    model,
    ref_window: str,
    var_window: str,
    relative_position: int,
    feature_config: Optional[FeatureConfig] = None,
) -> Dict[str, "np.ndarray"]:
    """Build the feature vector for one SNV.

    Returns ``{"embedding": (n_embedding_features,), "scalar": (n_scalar,),
    "scalar_names": (...)}``. The embedding block holds ``var - ref`` deltas at
    each configured pooling radius, concatenated in ``pool_radii`` order.
    """
    import numpy as np

    cfg = feature_config or config.DEFAULT_FEATURE_CONFIG

    if len(ref_window) != len(var_window):
        raise ValueError(
            f"Reference and variant windows differ in length "
            f"({len(ref_window)} vs {len(var_window)}); expected an SNV."
        )

    if cfg.pair_batch:
        lp, emb = _forward(model, [ref_window, var_window], cfg)
        ref_lp, var_lp = lp[0], lp[1]
        ref_emb, var_emb = emb[0], emb[1]
    else:
        ref_lp_batch, ref_emb_batch = _forward(model, [ref_window], cfg)
        var_lp_batch, var_emb_batch = _forward(model, [var_window], cfg)
        ref_lp, var_lp = ref_lp_batch[0], var_lp_batch[0]
        ref_emb, var_emb = ref_emb_batch[0], var_emb_batch[0]

    length = len(ref_window)
    centre = relative_position

    # --- pooled embedding deltas
    # Hidden states are indexed by sequence position with no shift, but clamp
    # the centre in case the model returns a different length than it was given.
    emb_len = ref_emb.shape[0]
    emb_centre = min(centre, emb_len - 1)

    blocks = []
    for radius in cfg.pool_radii:
        window = _pool_slice(emb_len, emb_centre, radius)
        ref_pool = ref_emb[window].mean(axis=0)
        var_pool = var_emb[window].mean(axis=0)
        blocks.append(var_pool - ref_pool)
    embedding = np.concatenate(blocks).astype(np.float32)

    # --- scalar likelihood features
    ref_full = _mean_over(ref_lp, length, centre, None)
    var_full = _mean_over(var_lp, length, centre, None)
    ref_local = _mean_over(ref_lp, length, centre, cfg.local_radius)
    var_local = _mean_over(var_lp, length, centre, cfg.local_radius)
    ref_at = float(ref_lp[centre]) if centre > 0 else 0.0
    var_at = float(var_lp[centre]) if centre > 0 else 0.0

    scalars = {
        "delta_score_full": var_full - ref_full,
        "delta_score_local": var_local - ref_local,
        "delta_lp_at_variant": var_at - ref_at,
        "ref_score_full": ref_full,
        "ref_lp_at_variant": ref_at,
        "var_lp_at_variant": var_at,
    }
    scalar = np.array(
        [scalars[name] for name in SCALAR_FEATURE_NAMES], dtype=np.float32
    )

    return {
        "embedding": embedding,
        "scalar": scalar,
        "scalar_names": SCALAR_FEATURE_NAMES,
        # Surfaced separately so the endpoint can still report the zero-shot
        # number the frontend already renders.
        "delta_score": scalars["delta_score_full"],
    }


# --- Sharded batch extraction ---------------------------------------------


def features_dir(feature_config: Optional[FeatureConfig] = None) -> str:
    cfg = feature_config or config.DEFAULT_FEATURE_CONFIG
    return f"{config.FEATURES_DIR}/{cfg.tag()}"


def shard_path(shard_index: int, feature_config: Optional[FeatureConfig] = None) -> str:
    return f"{features_dir(feature_config)}/shard_{shard_index:05d}.npz"


def shard_bounds(n_items: int, n_shards: int) -> List[Tuple[int, int]]:
    """Split ``n_items`` into ``n_shards`` contiguous ranges."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    size = (n_items + n_shards - 1) // n_shards
    return [
        (start, min(start + size, n_items))
        for start in range(0, n_items, size)
    ] or [(0, 0)]


def extract_shard(
    frame,
    shard_index: int,
    feature_config: Optional[FeatureConfig] = None,
    skip_existing: bool = True,
    model=None,
) -> str:
    """Extract features for one shard of the variant table and save an ``.npz``.

    Rows whose reference base disagrees with the assembly are dropped and
    counted rather than scored, since a coordinate or assembly error would
    otherwise enter the training set as a mislabelled example.

    ``model`` lets a caller that runs several shards in one process (the Kaggle
    notebooks) load Evo2 once and pass it in; on Modal each shard is a fresh
    container so it is left ``None`` and loaded here.
    """
    import numpy as np

    from .loader import load_evo2

    cfg = feature_config or config.DEFAULT_FEATURE_CONFIG
    out_path = shard_path(shard_index, cfg)

    if skip_existing and os.path.exists(out_path):
        print(f"Shard {shard_index} already done at {out_path}, skipping")
        return out_path

    os.makedirs(features_dir(cfg), exist_ok=True)

    if model is None:
        print(f"Loading {config.MODEL_NAME} ...")
        model = load_evo2(config.MODEL_NAME)
        print("Model loaded")

    genomes = {}
    variant_ids, embeddings, scalars, labels = [], [], [], []
    skipped = {"reference_mismatch": 0, "out_of_bounds": 0, "error": 0}

    try:
        _score_rows(
            frame, cfg, model, genomes,
            variant_ids, embeddings, scalars, labels, skipped, shard_index,
        )
    finally:
        # Leaving these open would break the volume reload at the start of the
        # next shard scheduled onto this container.
        for genome in genomes.values():
            genome.close()

    if not variant_ids:
        raise RuntimeError(
            f"Shard {shard_index} produced no features. Skipped: {skipped}"
        )

    # Written through an open file handle, not a path: given a path that does
    # not already end in ``.npz``, numpy silently appends the suffix, so
    # ``shard_0.npz.part`` lands as ``shard_0.npz.part.npz`` and the rename
    # below then fails on a file that was never created — after the shard's
    # entire GPU cost has been paid. Passing a handle disables that rewriting.
    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            variant_id=np.array(variant_ids),
            embedding=np.stack(embeddings),
            scalar=np.stack(scalars),
            label=np.array(labels, dtype=np.int64),
        )
    os.replace(tmp_path, out_path)

    print(
        f"Shard {shard_index}: wrote {len(variant_ids)} rows to {out_path} "
        f"(skipped {skipped})"
    )
    return out_path


def _score_rows(
    frame, cfg, model, genomes,
    variant_ids, embeddings, scalars, labels, skipped, shard_index,
):
    """Score every row of ``frame`` in place into the accumulator lists."""
    import numpy as np
    import torch

    from .sequences import build_variant_window, open_genome

    for counter, (_, row) in enumerate(frame.iterrows(), start=1):
        try:
            assembly = row["assembly"]
            if assembly not in genomes:
                genomes[assembly] = open_genome(assembly)
            genome = genomes[assembly]

            ref_window, start = genome.window(
                row["chrom"], int(row["pos"]), cfg.window_size
            )
            relative = int(row["pos"]) - 1 - start
            var_window, _ = build_variant_window(
                ref_window, relative, row["alt"], expected_reference=row["ref"]
            )

            result = extract_variant_features(
                model, ref_window, var_window, relative, cfg
            )
            variant_ids.append(row["variant_id"])
            embeddings.append(result["embedding"].astype(np.float16))
            scalars.append(result["scalar"])
            labels.append(int(row["label"]))

        except ValueError as exc:
            key = (
                "reference_mismatch"
                if "Reference mismatch" in str(exc)
                else "out_of_bounds"
            )
            skipped[key] += 1
        except torch.cuda.OutOfMemoryError as exc:
            # An OOM leaves the failed allocation's memory reserved, so without
            # dropping the cache here one bad row cascades into every row after
            # it failing too. Retried once, since with the cache cleared the
            # same window usually fits.
            skipped["error"] += 1
            torch.cuda.empty_cache()
            if skipped["error"] <= 5:
                print(f"  OOM on {row.get('variant_id')}, cache cleared: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill a shard
            skipped["error"] += 1
            if skipped["error"] <= 5:
                print(f"  error on {row.get('variant_id')}: {exc}")

        if counter % 100 == 0:
            print(f"  shard {shard_index}: {counter}/{len(frame)} processed")


def merge_shards(feature_config: Optional[FeatureConfig] = None) -> str:
    """Concatenate every shard into a single ``features.npz``."""
    import glob
    import re

    import numpy as np

    cfg = feature_config or config.DEFAULT_FEATURE_CONFIG
    # Deliberately strict: `shard_*.npz` would also match leftovers like
    # `shard_00000.npz.part.npz`, whose rows belong to whatever split was being
    # extracted when they were written. Silently concatenating those into the
    # training set is far worse than failing to find them.
    pattern = re.compile(r"^shard_\d{5}\.npz$")
    paths = sorted(
        p for p in glob.glob(f"{features_dir(cfg)}/shard_*.npz")
        if pattern.match(os.path.basename(p))
    )
    if not paths:
        raise FileNotFoundError(
            f"No shards under {features_dir(cfg)}. Run the extract stage first."
        )

    variant_ids, embeddings, scalars, labels = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            variant_ids.append(data["variant_id"])
            embeddings.append(data["embedding"])
            scalars.append(data["scalar"])
            labels.append(data["label"])

    merged = f"{features_dir(cfg)}/features.npz"
    np.savez(
        merged,
        variant_id=np.concatenate(variant_ids),
        embedding=np.concatenate(embeddings),
        scalar=np.concatenate(scalars),
        label=np.concatenate(labels),
    )
    total = sum(len(x) for x in labels)
    print(f"Merged {len(paths)} shards ({total} variants) into {merged}")
    return merged


def load_features(feature_config: Optional[FeatureConfig] = None):
    """Load merged features as ``(variant_id, X, y)`` with X float32."""
    import numpy as np

    cfg = feature_config or config.DEFAULT_FEATURE_CONFIG
    merged = f"{features_dir(cfg)}/features.npz"
    if not os.path.exists(merged):
        raise FileNotFoundError(
            f"{merged} not found. Run the extract and merge stages first."
        )

    with np.load(merged, allow_pickle=False) as data:
        variant_id = data["variant_id"]
        X = np.concatenate(
            [data["embedding"].astype(np.float32), data["scalar"].astype(np.float32)],
            axis=1,
        )
        y = data["label"]
    return variant_id, X, y
