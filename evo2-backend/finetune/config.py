"""Shared configuration for the Evo2 variant-effect fine-tuning pipeline.

Everything that both the training jobs and the inference endpoint need to agree on
lives here. If a constant affects the shape or meaning of a feature vector, it
belongs in this file so that train/serve skew is impossible by construction.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

# --- Modal resources -------------------------------------------------------

APP_NAME = "evo2-finetune"

HF_CACHE_VOLUME = "hf_cache"
HF_CACHE_PATH = "/root/.cache/huggingface"

DATA_VOLUME = "evo2-finetune-data"
DATA_PATH = "/data"

GENOMES_DIR = f"{DATA_PATH}/genomes"
DATASETS_DIR = f"{DATA_PATH}/datasets"
FEATURES_DIR = f"{DATA_PATH}/features"
RUNS_DIR = f"{DATA_PATH}/runs"

# The deployed endpoint reads the trained head from here.
ACTIVE_RUN_DIR = f"{RUNS_DIR}/active"

# --- Model -----------------------------------------------------------------

MODEL_NAME = "evo2_7b"

# Documented in the Evo2 README as the embedding tap for evo2_7b. Output of the
# GLU MLP's final projection in block 28 of 32, so its width is hidden_size.
EMBEDDING_LAYER = "blocks.28.mlp.l3"
EMBEDDING_DIM = 4096

# --- Reference genomes -----------------------------------------------------

# ClinVar is distributed on GRCh38; the Findlay BRCA1 saturation-mutagenesis
# table is on GRCh37, and evo2 vendors a GRCh37 chr17 FASTA, so each dataset is
# scored against its own assembly rather than lifted over.
# Mirrors of the same file, tried in order. The primary host intermittently
# refuses connections outright (TLS handshake reset) from some networks, which
# no amount of retrying fixes; hgdownload2 serves byte-identical data under the
# same path, including the "chr"-prefixed sequence names the pipeline expects.
HG38_URLS = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    "https://hgdownload2.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
)
HG38_URL = HG38_URLS[0]
HG38_FASTA = f"{GENOMES_DIR}/hg38.fa"

# Present inside the image, cloned by the Dockerfile step in main.py.
HG19_CHR17_FASTA_GZ = "/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz"
HG19_CHR17_FASTA = f"{GENOMES_DIR}/GRCh37.p13_chr17.fa"

BRCA1_DMS_XLSX = "/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx"

CLINVAR_VCF_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
)
CLINVAR_VCF_GZ = f"{DATASETS_DIR}/clinvar.vcf.gz"

# BRCA1 locus, used to hold the DMS benchmark out of the ClinVar training set.
# Coordinates are 1-based inclusive.
BRCA1_LOCUS = {
    "hg38": ("chr17", 43_044_295, 43_170_245),
    "hg19": ("chr17", 41_196_312, 41_277_500),
}

# --- Feature extraction ----------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Defines the feature vector produced for a single SNV.

    ``window_size`` dominates GPU cost (StripedHyena is close to linear in
    sequence length), so halving it roughly halves the extraction bill. It
    defaults to 8192 to match the window the zero-shot endpoint already scores.
    """

    window_size: int = 8192

    # Pooling radii in bases around the variant, applied to both the embedding
    # and the per-token log-probabilities. ``0`` is the variant position alone;
    # ``None`` is the whole window.
    pool_radii: Tuple[Optional[int], ...] = (0, 128, None)

    # Radius used for the scalar "local" delta-likelihood feature.
    local_radius: int = 64

    embedding_layer: str = EMBEDDING_LAYER
    embedding_dim: int = EMBEDDING_DIM

    # Score the reference and variant sequences as a single batch of 2. Better
    # H100 utilisation than two batch-1 forwards; set False if you hit OOM.
    pair_batch: bool = True

    @property
    def n_embedding_features(self) -> int:
        return len(self.pool_radii) * self.embedding_dim

    @property
    def n_scalar_features(self) -> int:
        return len(SCALAR_FEATURE_NAMES)

    @property
    def n_features(self) -> int:
        return self.n_embedding_features + self.n_scalar_features

    def tag(self) -> str:
        """Stable identifier so feature caches with different settings don't mix."""
        radii = "-".join("full" if r is None else str(r) for r in self.pool_radii)
        return f"w{self.window_size}_l{self.embedding_layer.replace('.', '_')}_p{radii}"


# Order matters: the inference path rebuilds the vector from these names.
SCALAR_FEATURE_NAMES = (
    "delta_score_full",      # mean log-prob(variant window) - mean log-prob(reference window)
    "delta_score_local",     # same, restricted to +/- local_radius around the variant
    "delta_lp_at_variant",   # log-prob of the substituted base minus that of the reference base
    "ref_score_full",        # reference window mean log-prob: a conservation / context proxy
    "ref_lp_at_variant",     # how confidently the model predicts the reference base
    "var_lp_at_variant",
)

DEFAULT_FEATURE_CONFIG = FeatureConfig()

# --- Training --------------------------------------------------------------


@dataclass
class TrainConfig:
    head: str = "mlp"              # "mlp" or "linear"
    hidden_sizes: Tuple[int, ...] = (256, 64)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10             # early-stopping patience on validation AUROC
    seed: int = 0

    # Chromosomes held out of training. Kept disjoint so that variants in
    # linkage with each other cannot straddle the split.
    val_chromosomes: Tuple[str, ...] = ("chr8", "chr18")
    test_chromosomes: Tuple[str, ...] = ("chr2", "chr16")

    # Drop the BRCA1 locus from ClinVar training data so the DMS benchmark
    # stays an honest out-of-distribution check.
    exclude_brca1_from_training: bool = True

    class_weighting: bool = True   # ClinVar is pathogenic-heavy at 2+ stars


DEFAULT_TRAIN_CONFIG = TrainConfig()

# --- Dataset construction --------------------------------------------------


@dataclass
class ClinVarConfig:
    """Filters applied when turning the ClinVar VCF into a labelled SNV table."""

    min_review_stars: int = 2      # "multiple submitters, no conflicts" or better
    max_variants: Optional[int] = 40_000
    balance_classes: bool = True
    seed: int = 0


DEFAULT_CLINVAR_CONFIG = ClinVarConfig()
