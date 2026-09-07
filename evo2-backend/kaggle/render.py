"""Generate a single self-contained Kaggle notebook for the fine-tuning pipeline.

    python kaggle/render.py

Everything the pipeline needs is baked into the notebook: the `finetune/` package
is embedded (base64) and written to disk in the first cell, so there is no repo to
clone and no `evo2-finetune-src` dataset to attach. Cross-session state (feature
shards, datasets, runs) is carried by adding **this notebook's own output as an
input** on the next run -- a pure Kaggle-UI flow, no `kaggle` CLI on your laptop.

Re-run this whenever you edit `finetune/*.py` or the cell text below, then
re-upload `kaggle/evo2-finetune.ipynb` to Kaggle.
"""

import ast
import base64
import json
import pathlib

HERE = pathlib.Path(__file__).parent
FT = HERE.parent / "finetune"
EMBED = ["__init__.py", "config.py", "data.py", "sequences.py",
         "features.py", "head.py", "train.py", "loader.py"]

FILES = {
    f"finetune/{name}": base64.b64encode((FT / name).read_bytes()).decode()
    for name in EMBED
}


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


# Per-notebook knobs. Defaults produce the main notebook; `python kaggle/render.py
# --worker LO HI` also writes evo2-finetune-shards-LO-HI.ipynb, a copy that only
# extracts shards [LO, HI) so several sessions can run disjoint ranges at once.
import sys as _sys

CONFIG_STAGE = "auto"
SHARD_LO = None
SHARD_HI = None
_OUT_NAME = "evo2-finetune.ipynb"
if "--worker" in _sys.argv:
    _k = _sys.argv.index("--worker")
    SHARD_LO, SHARD_HI = int(_sys.argv[_k + 1]), int(_sys.argv[_k + 2])
    # "auto" so the worker builds its own dataset if no input is attached; the
    # dataset build is deterministic (fixed seed + variant_id sort) so every
    # worker's shard row-ranges match the main notebook's exactly.
    CONFIG_STAGE = "auto"
    _OUT_NAME = f"evo2-finetune-shards-{SHARD_LO}-{SHARD_HI}.ipynb"

CELLS = [
    md("""
# Evo2 variant-effect fine-tuning — one notebook

Kaggle port of the Modal pipeline in `evo2-backend/`. Same `finetune/` code
(embedded below), notebooks instead of Modal functions, and **notebook output as
input** instead of a persistent Volume.

## How to run

Set `STAGE` in the config cell and *Save Version* (Run All). Between runs, add the
previous version's output as input:

> Add Input → **Notebook Output** → this notebook → latest version

Order: `datasets` → `extract` (repeat until all shards done) → `merge` → `train`
→ `publish`. Or set `STAGE = "auto"` and it does the next unfinished thing each
run, extracting shards until the session is nearly out of time.

## Accelerator / settings

| stage | accelerator | internet |
| --- | --- | --- |
| datasets | None (save quota) | **on** |
| extract | **GPU T4 x2** — both T4s required, see below | **on** |
| merge / train / publish | None | off |

**`extract` needs `GPU T4 x2`, not a single GPU.** evo2_7b's weights are ~14 GB
and don't fit one 16 GB T4. vortex pipeline-parallelises the 32 blocks across
every visible CUDA device (~7 GB of weights per T4), and `features._forward`
feeds inputs on `cuda:0` to match — so two T4s work with no extra wiring, one
does not. The cell aborts early with a clear message if it sees < 2 GPUs.

On the T4 (compute capability 7.5) `loader.py` also (a) disables FP8 autocast
(`evo2_7b` sets `use_fp8_input_projections`, which needs CC ≥ 8.9) and (b) casts
the model to float16 (vortex's Triton kernels emit `.bf16` PTX, which needs
sm_80+). Both are approximations of the reference numerics and are no-ops on
Modal's Ampere+ GPUs. Watch the `extract` smoke test for VRAM headroom and for
inf/nan from the fp16 exponent range; `window_size` is held at 2048.
"""),

    md("### 1 · Write the embedded `finetune/` package to disk"),
    code(f"""
import base64, pathlib, sys

SRC = "/kaggle/working/src"
FILES = {json.dumps(FILES, indent=0)}

for rel, b64 in FILES.items():
    p = pathlib.Path(SRC, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
print("wrote", len(FILES), "modules to", SRC)
"""),

    md("### 2 · Config — set `STAGE`, then Run All"),
    code(f"""
STAGE = "{CONFIG_STAGE}"          # datasets | extract | merge | train | publish | auto

WINDOW_SIZE  = 2048     # 8192 on Modal; shrunk to fit a 16 GB T4
MODEL_NAME   = "evo2_7b"
MAX_VARIANTS = 40_000   # ClinVar cap; lower to cut extraction cost
N_SHARDS     = 60       # DO NOT change between extract runs
MIN_STARS    = 2
EXTRACT_HOURS = 8.0     # stop extracting this long into the session (9 h wall)
PUBLISH_RUN  = ""       # for STAGE=publish: the run dir name from `train`

# Worker mode: extract only shards [SHARD_LO, SHARD_HI). Leave both None on the
# main notebook (does all pending). Set a disjoint range in each parallel copy,
# then attach every copy's output to the main notebook and run merge/train.
SHARD_LO = {SHARD_LO!r}
SHARD_HI = {SHARD_HI!r}

import os, time, glob, subprocess
WORK, OUT, DATA, TMP = "/kaggle/working", "/kaggle/working/out", "/kaggle/working/out/data", "/kaggle/working/tmp"

# Kaggle pins its whole preinstalled package set via PIP_CONSTRAINT, which
# silently overrides `pip install torch==2.4.1` back to the image's torch 2.10.
# Clear it so our version pins actually take.
for _v in ("PIP_CONSTRAINT", "UV_CONSTRAINT"):
    os.environ.pop(_v, None)

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=check)
"""),

    md("""
### 3 · Restore state from a previous run

Copies `data/`, `features/`, `runs/` out of any attached input (this notebook's
earlier output, or a dataset) into the working tree. hg38 is symlinked, not
copied.
"""),
    code("""
for d in (f"{DATA}/genomes", f"{DATA}/datasets", f"{OUT}/features", f"{OUT}/runs", TMP):
    os.makedirs(d, exist_ok=True)

for base in glob.glob("/kaggle/input/*"):
    src_data = None
    for cand in (f"{base}/out/data", f"{base}/data"):
        if os.path.isdir(cand):
            src_data = cand
    if src_data:
        g = f"{src_data}/genomes"
        if os.path.isdir(g):
            for n in os.listdir(g):
                dst = f"{DATA}/genomes/{n}"
                if not os.path.exists(dst):
                    os.symlink(f"{g}/{n}", dst)
        sh(f"cp -n {src_data}/datasets/*.parquet {DATA}/datasets/ 2>/dev/null", check=False)
    for sub in ("features", "runs"):
        for cand in (f"{base}/out/{sub}", f"{base}/{sub}"):
            if os.path.isdir(cand):
                sh(f"cp -rn {cand}/. {OUT}/{sub}/ 2>/dev/null", check=False)

sh(f"ls -R {OUT} | head -40", check=False)
"""),

    md("### 4 · Point `finetune.config` at Kaggle paths"),
    code("""
import finetune.config as C
C.DATA_PATH     = DATA
C.GENOMES_DIR   = f"{DATA}/genomes"
C.DATASETS_DIR  = f"{DATA}/datasets"
C.FEATURES_DIR  = f"{OUT}/features"
C.RUNS_DIR      = f"{OUT}/runs"
C.ACTIVE_RUN_DIR = f"{C.RUNS_DIR}/active"
C.CLINVAR_VCF_GZ = f"{C.DATASETS_DIR}/clinvar.vcf.gz"
C.HG38_FASTA     = f"{C.GENOMES_DIR}/hg38.fa"
C.HG19_CHR17_FASTA = f"{C.GENOMES_DIR}/GRCh37.p13_chr17.fa"

from finetune.config import FeatureConfig, TrainConfig, ClinVarConfig
FC = FeatureConfig(window_size=WINDOW_SIZE)
print("feature tag:", FC.tag())

need = {"datasets", "extract"} if STAGE == "auto" else {STAGE}
have_ds = os.path.exists(f"{C.DATASETS_DIR}/variants.parquet")
"""),

    md("""
### 5 · Environment

`datasets` needs only light deps. `extract` rebuilds `common.py`'s CUDA stack
(torch 2.4.1 + a prebuilt flash-attn wheel + TE 1.13 + evo2 from source) — the
fragile step; needs Internet on.
"""),
    code("""
def clone_evo2(submodules):
    if os.path.isdir(f"{TMP}/evo2"):
        return
    flag = "--recurse-submodules" if submodules else "--depth 1"
    sh(f"git clone {flag} https://github.com/ArcInstitute/evo2.git {TMP}/evo2")

FLASH_WHL = ("https://github.com/Dao-AILab/flash-attention/releases/download/"
             "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE"
             "-{py}-{py}-linux_x86_64.whl")


def _pkg_version(name):
    import importlib.metadata as _md          # reads dist-info, does NOT import
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _stale_torch(_s):
    # A torch left half-imported or at the wrong version by an earlier Run All
    # cannot be fixed in-process (NameError _C, or the docstring RuntimeError).
    t = _s.modules.get("torch")
    if t is not None and (not hasattr(t, "_C")
                          or not getattr(t, "__version__", "").startswith("2.4")):
        raise SystemExit(
            "  *** Run menu -> 'Restart & Run All'. A stale torch "
            f"({getattr(t, '__version__', 'partial import')}) is loaded in this "
            "kernel and cannot be swapped in place. Installs are cached, so the "
            "next pass is quick. ***")


def build_gpu_env():
    import sys as _s

    # torch cannot be re-imported in a live kernel (RuntimeError: '...already has
    # a docstring'). If a wrong torch is already loaded, only a restart fixes it.
    _stale_torch(_s)

    py = f"cp{_s.version_info.major}{_s.version_info.minor}"
    os.environ["NVTE_FRAMEWORK"] = "pytorch"
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["MAX_JOBS"] = "4"

    sh("pip -q install 'setuptools<70' packaging wheel ninja cmake pybind11")
    clone_evo2(True)

    # evo2's install no longer hard-pins torch: the current Kaggle image ships
    # torch 2.10, which satisfies evo2's range, so `pip install .` leaves it in
    # place and the flash-attn / TE 1.13 stack below (built for 2.4's ABI) then
    # mismatches. Pin torch 2.4.1+cu124 ourselves first -- the cu124 wheel pulls
    # the matching nvidia-* runtime deps -- so evo2's install is a no-op for it.
    sh("pip -q install --force-reinstall torch==2.4.1 "
       "--index-url https://download.pytorch.org/whl/cu124")
    sh(f"cd {TMP}/evo2 && pip -q install .", check=False)

    # flash-attn: force the wheel that matches torch 2.4's ABI (evo2 may have
    # left a mismatched one). --no-deps so it can't move torch.
    sh(f"pip install --no-deps --force-reinstall '{FLASH_WHL.format(py=py)}'")

    # transformer-engine 1.13: cu12 lib has a wheel; the pytorch bindings
    # (transformer_engine_torch) are a source build against the *installed*
    # torch's ABI. evo2_7b needs FP8 input projections -> TE is mandatory.
    #
    # Two traps, both about torch:
    #  * pip's wheel cache holds a _torch build from an earlier Run All linked
    #    to a different torch -- it installs but fails to import. --no-cache-dir
    #    --no-binary forces a fresh compile against the current torch.
    #  * transformer_engine_torch's setup.py lists `torch` with NO version
    #    bound, so --force-reinstall (or an unconstrained resolve) happily pulls
    #    the newest torch off PyPI and uninstalls our 2.4.1 mid-install, leaving
    #    the .so linked to 2.4 headers but 2.x loaded (undefined-symbol on
    #    import). So: NO --force-reinstall, and pin torch==2.4.1 in the same
    #    command to hold the resolver.
    sh("pip uninstall -y transformer-engine transformer_engine transformer_engine_cu12 "
       "transformer_engine_torch", check=False)
    sh("pip -q install transformer_engine_cu12==1.13.0", check=False)
    r = subprocess.run(
        f"pip install -v --no-cache-dir --no-build-isolation "
        f"--no-binary transformer_engine_torch "
        f"torch==2.4.1 transformer_engine_torch==1.13.0 "
        f"transformer_engine[pytorch]==1.13.0 "
        f"2>&1 | tee {WORK}/te_build.log",
        shell=True,
    )
    if r.returncode:
        raise SystemExit(
            "  *** transformer-engine build failed. evo2_7b requires TE (FP8 "
            f"input projections). See {WORK}/te_build.log. ***")

    # The TE install still lists torch unbounded; make sure nothing bumped it.
    tv_now = _pkg_version("torch")
    if tv_now is None or not tv_now.startswith("2.4"):
        raise SystemExit(
            f"  *** torch became {tv_now} during the transformer-engine install "
            "(its setup.py depends on unpinned `torch`). Run menu -> 'Restart & "
            "Run All'. ***")

    # Confirm the freshly built _torch extension actually imports against this
    # torch -- an ABI mismatch here is silent until vortex sets HAS_TE=False.
    chk = subprocess.run(
        "python -c 'import transformer_engine.pytorch, transformer_engine_torch'",
        shell=True,
    )
    if chk.returncode:
        raise SystemExit(
            "  *** transformer-engine installed but fails to import (ABI "
            "mismatch with torch). Run menu -> 'Restart & Run All'; the fresh "
            "kernel rebuilds it against the pinned torch. ***")

    # Verify on disk without importing torch.
    tv, fv = _pkg_version("torch"), _pkg_version("flash-attn")
    print(f"on disk: torch={tv}  flash-attn={fv}")
    if tv is None or not tv.startswith("2.4"):
        raise SystemExit(
            f"  *** torch on disk is {tv}, expected 2.4.x. The explicit "
            "torch==2.4.1 pin above did not take (PIP_CONSTRAINT? evo2 install "
            "moved it?) -- check the pip output above. ***")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sh("pip -q install pyfaidx pyarrow openpyxl requests pandas scikit-learn numpy")
os.environ.setdefault("HF_HOME", f"{TMP}/hf")

def gpu_present():
    # Deliberately does NOT import torch -- see build_gpu_env(). Checks the
    # driver directly.
    return os.path.isdir("/proc/driver/nvidia/gpus") and bool(
        os.listdir("/proc/driver/nvidia/gpus")
    )

if need & {"extract"} and not gpu_present() and STAGE == "extract":
    raise SystemExit("STAGE=extract needs a GPU. Set Accelerator -> 'GPU T4 x2'.")

if need & {"datasets", "extract"}:
    if need & {"extract"} and gpu_present():
        build_gpu_env()
    elif need & {"extract"}:
        print("no GPU -> skipping the CUDA build; this run can only do 'datasets'")
    clone_evo2(False)
    C.HG19_CHR17_FASTA_GZ = f"{TMP}/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz"
    C.BRCA1_DMS_XLSX = f"{TMP}/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx"
"""),

    md("### 6 · Stage: build datasets"),
    code("""
if (STAGE == "datasets") or (STAGE == "auto" and not have_ds):
    from finetune import data, sequences
    cc = ClinVarConfig(min_review_stars=MIN_STARS, max_variants=MAX_VARIANTS, balance_classes=True)
    combined = data.assign_splits(data.build_clinvar_dataset(cc), data.build_brca1_dataset())
    data.save_dataset(combined)
    sequences.ensure_hg38()
    sequences.ensure_hg19_chr17()
    have_ds = True
    import pandas as pd
    print(pd.read_parquet(f"{C.DATASETS_DIR}/variants.parquet").groupby("split").size())
else:
    print("skip (STAGE=%s, datasets present=%s)" % (STAGE, have_ds))
"""),

    md("### 7 · Stage: extract features  (GPU, resumable)"),
    code("""
if STAGE in ("extract", "auto") and (STAGE == "extract" or have_ds):
    import sys as _sys
    _stale_torch(_sys)          # bail with a clear message on a polluted kernel
    import pandas as pd, torch
    from finetune import features as F
    from finetune.loader import load_evo2
    from finetune.sequences import build_variant_window, open_genome

    frame = (pd.read_parquet(f"{C.DATASETS_DIR}/variants.parquet")
               .sort_values("variant_id").reset_index(drop=True))

    # evo2_7b's weights alone are ~14 GB -- they do not fit one 16 GB T4. vortex
    # pipeline-parallelises across every visible CUDA device (ceil(num_layers /
    # device_count) blocks per GPU), so two T4s (~7 GB of weights each) is enough
    # but one is not. features._forward puts inputs on cuda:0, matching vortex's
    # first-device placement, so no other wiring is needed.
    ngpu = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory
                     for i in range(ngpu)) / 2**30
    print(f"visible CUDA devices: {ngpu}  ({total_vram:.0f} GiB total)")
    # evo2_7b weights are ~14 GiB; need real headroom for activations too. One
    # 24 GiB L4 is plenty; two 16 GiB T4s work via vortex's pipeline split; one
    # T4 does not.
    if MODEL_NAME == "evo2_7b" and total_vram < 20:
        raise SystemExit(
            f"  *** evo2_7b needs >=20 GiB of GPU memory ({total_vram:.0f} GiB "
            f"across {ngpu} device(s) here). Use Accelerator 'GPU T4 x2' or "
            "'GPU L4x4' -- not a single 'GPU T4' or 'GPU P100'. ***")

    model = load_evo2(MODEL_NAME)

    # smoke test — one variant end to end
    r = frame.iloc[0]; g = open_genome(r["assembly"])
    ref_w, st = g.window(r["chrom"], int(r["pos"]), FC.window_size)
    rel = int(r["pos"]) - 1 - st
    var_w, _ = build_variant_window(ref_w, rel, r["alt"], expected_reference=r["ref"])
    t0 = time.time(); out = F.extract_variant_features(model, ref_w, var_w, rel, FC); g.close()
    spv = time.time() - t0
    print(f"smoke ok  {spv:.2f}s/variant  peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    print(f"~{len(frame)*spv/3600:.1f} GPU-h for all {len(frame)} variants")

    bounds = F.shard_bounds(len(frame), N_SHARDS)
    pending = [i for i in range(len(bounds)) if not os.path.exists(F.shard_path(i, FC))]
    if SHARD_LO is not None:
        pending = [i for i in pending if SHARD_LO <= i < SHARD_HI]
        print(f"worker mode: restricted to shards [{SHARD_LO}, {SHARD_HI})")
    print(f"{len(bounds)-len(pending)}/{len(bounds)} shards done, {len(pending)} pending")

    deadline = time.time() + EXTRACT_HOURS * 3600
    for i in pending:
        if time.time() > deadline:
            print("time budget hit — Save Version and re-run to continue"); break
        s, e = bounds[i]
        print(f"=== shard {i} rows [{s},{e}) ===", flush=True)
        try:
            F.extract_shard(frame.iloc[s:e], i, FC, model=model)
        except Exception as ex:
            print(f"  shard {i} failed: {ex}")
    done = sum(os.path.exists(F.shard_path(i, FC)) for i in range(N_SHARDS))
    print(f"{done}/{N_SHARDS} shards on disk")
else:
    print("skip extract")
"""),

    md("### 8 · Stage: merge · train · publish"),
    code("""
from finetune import features as F, train as T

merged = os.path.exists(f"{F.features_dir(FC)}/features.npz")
all_shards = have_ds and all(os.path.exists(F.shard_path(i, FC)) for i in range(N_SHARDS))

if STAGE in ("merge", "auto") and all_shards and not merged:
    F.merge_shards(FC); merged = True

if STAGE in ("train", "auto") and merged:
    trained = T.train_head(feature_config=FC, train_config=TrainConfig(head="mlp"))
    T.train_head(feature_config=FC, train_config=TrainConfig(head="linear"))
    T.summarise_runs()
    import json as _j
    print(_j.dumps(trained.metrics["splits"], indent=2, default=float))

if STAGE == "publish":
    assert PUBLISH_RUN, "set PUBLISH_RUN to a run dir name from the train stage"
    T.publish(PUBLISH_RUN)
    sh(f"ls -la {C.ACTIVE_RUN_DIR}")
"""),

    md("""
### 9 · Next step

**Save Version** now. For the next run, *Add Input → Notebook Output → this
notebook → latest version* so the shards/datasets/runs you just produced are
restored by cell 3.

When a head is trained and published, get `out/runs/active/` onto the Modal
volume (from anywhere with the `modal` CLI):

```
# Kaggle UI → this notebook → Output → Download, then:
modal volume put evo2-finetune-data ./out/runs/active /runs/active
modal deploy main.py
```
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(CELLS):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))  # fail the build, not the notebook
    out = HERE / _OUT_NAME
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, f"({out.stat().st_size // 1024} KB); code cells parse OK")


if __name__ == "__main__":
    main()
