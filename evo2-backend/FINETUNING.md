# Fine-tuning Evo2 for variant effect prediction

The deployed endpoint scores a variant by subtracting the reference window's
mean log-likelihood from the mutated window's, then comparing that delta against
a threshold fitted on 500 BRCA1 variants. This pipeline replaces that hand-fitted
threshold with a classifier trained on Evo2's own representations.

**Evo2 itself stays frozen.** It runs through `vortex`'s `StripedHyena`, which has
no LoRA support and wraps its forward pass in `no_grad`, so backpropagating into
the 7B weights would mean either building adapter injection from scratch or
moving to Arc's `savanna` training stack. Instead the pipeline caches Evo2's
hidden states and per-token log-probabilities once, and trains a small head on
top — the same approach the Evo2 paper uses for its supervised tasks. The
expensive GPU work happens once; iterating on the head then takes minutes.

## What gets built

```
finetune/
  config.py      every constant that affects a feature vector's shape or meaning
  sequences.py   reference genome access (local FASTA for bulk, UCSC REST for one-offs)
  data.py        ClinVar + BRCA1 DMS -> one labelled schema, with splits
  features.py    the GPU stage: embeddings + log-probs, sharded and resumable
  head.py        the classifier and its serialised artifact
  train.py       fitting, threshold selection, evaluation against the zero-shot baseline
finetune_app.py  Modal app tying the stages together
common.py        image + volumes shared with main.py
```

### Features

For each SNV, the reference and mutated 8 kb windows go through Evo2 **once, as a
batch of two**, and that single pass yields both:

- **Embedding deltas** — hidden states from `blocks.28.mlp.l3` (4096-dim, the tap
  documented in the Evo2 README), pooled at three radii around the variant
  (the base itself, ±128 bp, and the whole window). Stored as `variant − reference`.
- **Six scalar likelihood features** — the whole-window delta score the zero-shot
  path already uses, a ±64 bp local version, the delta at the substituted base,
  and the reference window's own likelihood as a conservation proxy.

`extract_variant_features()` is called by both the extraction job and the
inference endpoint, so a trained head cannot be fed differently-constructed
vectors at serving time.

### Data

- **ClinVar** (GRCh38) for training. Filtered to single-nucleotide variants with
  an unambiguous pathogenic or benign assertion at **2+ review stars**
  ("multiple submitters, no conflicts" or better), then class-balanced.
- **BRCA1 saturation mutagenesis** (Findlay et al. 2018, GRCh37) as a held-out
  benchmark. It ships with the evo2 repo, and its coordinates are hg19, so it is
  scored against the vendored GRCh37 chr17 FASTA rather than lifted over.

Splits are **chromosome-disjoint**, not random. Variants within a few kb share
most of an 8 kb window, so a random split would put near-duplicate inputs on both
sides of the boundary and inflate validation numbers. The BRCA1 locus is dropped
from ClinVar training data so the DMS benchmark stays genuinely out of
distribution.

## Runbook

All commands run from `evo2-backend/`. To run this on Kaggle's free GPUs instead
of Modal, see [`kaggle/README.md`](kaggle/README.md) — same `finetune/` code, with
notebooks in place of Modal functions and versioned Datasets in place of the Volume.

### 1. Build the datasets

Downloads ClinVar (~100 MB), parses it, loads the BRCA1 table, assigns splits,
and fetches + indexes hg38 (~950 MB download, 3.1 GB on disk). Takes 15–25 min,
mostly the genome download, and only needs doing once.

```bash
modal run finetune_app.py --stage build-datasets
```

### 2. Estimate the cost before you spend it

This is the step worth not skipping. It times real extraction on an H100 and
projects the full run.

```bash
modal run finetune_app.py --stage estimate --shards 40 --max-containers 10
```

Cost scales roughly linearly with variant count and window size. Two dials in
`finetune/config.py` if the estimate is higher than you want:

- `ClinVarConfig.max_variants` (default 40,000) — a usable head needs far fewer
  than the full ClinVar set; 10–20k is a reasonable starting point.
- `FeatureConfig.window_size` (default 8192) — halving it to 4096 roughly halves
  the bill. Changing it changes `FeatureConfig.tag()`, so features land in a
  separate cache directory and nothing gets mixed.

### 3. Extract features

Sharded across containers, resumable. Shards already on the volume are skipped,
so re-running after an interruption picks up where it stopped.

```bash
modal run finetune_app.py --stage extract --shards 40 --max-containers 10
modal run finetune_app.py --stage status          # what's done, what's pending
modal run finetune_app.py --stage merge           # once everything is done
```

Each shard reloads the model (a few minutes), so prefer fewer, larger shards —
but not so large that losing one to a retry hurts. 500–1500 variants per shard
is a reasonable band.

### 4. Train

Minutes, not hours — the GPU work is already cached.

```bash
modal run finetune_app.py --stage train
modal run finetune_app.py --stage train --head linear   # regularised baseline
modal run finetune_app.py --stage runs                  # compare finished runs
```

Every run prints the trained head's AUROC **and the zero-shot AUROC on the same
rows**. If the head doesn't beat the baseline on `test` and `benchmark`, it isn't
worth publishing — that comparison is the whole point of the report.

### 5. Publish

Copies a run into `runs/active`, where the endpoint looks on startup.

```bash
modal run finetune_app.py --stage publish --run-name run-20260814-101500
modal deploy main.py
```

## What the endpoint returns

The response keeps the shape the frontend already reads
(`position`, `reference`, `alternative`, `delta_score`, `prediction`,
`classification_confidence`) and adds:

| field | meaning |
| --- | --- |
| `model` | `"zero-shot"` or `"fine-tuned"` |
| `pathogenicity_probability` | calibrated probability (fine-tuned only) |
| `zero_shot` | what the raw delta score alone would have said (fine-tuned only) |

With no head published, `load_active_head()` returns `None` and the endpoint
behaves exactly as it did before, so deploying this is safe ahead of training.

One behaviour change either way: windows are now `window_size` bases rather than
`window_size + 1`, matching how the BRCA1 threshold was originally fitted in
`run_brca1_analysis`. Delta scores shift in about the sixth decimal place.

`classification_confidence` also changes meaning once a head is live. Zero-shot,
it is distance past the threshold in units of a class standard deviation, capped
at 1. Fine-tuned, it is the model's probability for the class it predicted — a
real probability, and usually less extreme.

## Tests

`tests/test_pipeline.py` covers everything that doesn't need a GPU: log-prob
alignment, pooling, genome windowing, ClinVar filtering, split assignment,
artifact round-trips and a full training run on synthetic features. It stubs out
`evo2` with a fake model whose embeddings depend only on the token at each
position, which makes the alignment assertions exact — a one-base shift in the
centring makes them fail.

```bash
python tests/test_pipeline.py
```

Needs `numpy pandas scikit-learn pyarrow pyfaidx torch` locally; no GPU, no Modal
account, no model weights. Worth running before launching an extraction job,
since a centring bug would only show up after the GPU bill.

## Notes and limits

- **ClinVar labels carry ascertainment bias.** Variants get submitted because
  someone had a reason to look. A head trained on it partly learns which regions
  get sequenced. The BRCA1 DMS benchmark is the counterweight: it comes from a
  functional assay over every possible SNV in the region, so it doesn't share
  that bias.
- **2+ stars is a quality floor, not a guarantee.** Lowering
  `min_review_stars` to 1 roughly triples the dataset at the cost of noisier
  labels.
- **Reference mismatches are dropped, not scored.** If a record's `ref` base
  disagrees with the assembly, `extract_shard` counts it and moves on. A large
  skip count means a coordinate or assembly problem worth investigating.
- **Feature caches are keyed by config.** `FeatureConfig.tag()` encodes window
  size, layer and pooling radii, so changing any of them starts a fresh cache
  rather than silently mixing incompatible vectors.
