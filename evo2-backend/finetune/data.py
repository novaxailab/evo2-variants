"""Labelled SNV datasets: ClinVar for training, BRCA1 DMS as a held-out benchmark.

Both sources are normalised onto one schema so downstream stages never need to
know where a variant came from:

    variant_id, source, assembly, chrom, pos, ref, alt, label,
    gene, review_stars, clnsig, split

``label`` is 1 for pathogenic / loss-of-function and 0 for benign / functional.
``pos`` is 1-based, matching both the VCF and the UCSC convention.
"""

import os
from typing import Dict, List, Optional, Tuple

from . import config

# --- ClinVar ---------------------------------------------------------------

# Explicit allowlists rather than substring tests: "Conflicting_classifications
# _of_pathogenicity" contains "pathogenic" but carries no usable label.
PATHOGENIC_CLNSIG = {
    "pathogenic",
    "likely_pathogenic",
    "pathogenic/likely_pathogenic",
}
BENIGN_CLNSIG = {
    "benign",
    "likely_benign",
    "benign/likely_benign",
}

REVIEW_STATUS_STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
}

_BASES = frozenset("ACGT")


def _parse_info(info: str) -> Dict[str, str]:
    fields = {}
    for entry in info.split(";"):
        key, sep, value = entry.partition("=")
        fields[key] = value if sep else "true"
    return fields


def _normalise_clnsig(raw: str) -> str:
    """Reduce a CLNSIG value to a comparable base classification.

    ClinVar appends qualifiers such as ``|other`` and ``,_low_penetrance``;
    those modify the assertion but not the pathogenic/benign call.
    """
    base = raw.split("|")[0].strip().lower()
    for suffix in (",_low_penetrance", ",_association", ",_risk_factor"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def _gene_symbol(geneinfo: str) -> str:
    if not geneinfo:
        return ""
    return geneinfo.split("|")[0].split(":")[0]


def download_clinvar(force: bool = False) -> str:
    """Fetch the ClinVar GRCh38 VCF onto the data volume."""
    import shutil

    import requests

    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    path = config.CLINVAR_VCF_GZ

    if os.path.exists(path) and not force:
        print(f"ClinVar VCF already present at {path}")
        return path

    print(f"Downloading ClinVar from {config.CLINVAR_VCF_URL} ...")
    tmp = path + ".part"
    with requests.get(config.CLINVAR_VCF_URL, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            shutil.copyfileobj(response.raw, handle, length=8 * 1024 * 1024)
    os.replace(tmp, path)
    print(f"Saved {os.path.getsize(path) / 1e6:.0f} MB to {path}")
    return path


def parse_clinvar_vcf(
    vcf_path: str,
    min_review_stars: int = 2,
) -> "pd.DataFrame":
    """Turn the ClinVar VCF into labelled single-nucleotide variants.

    Keeps only SNVs with an unambiguous pathogenic or benign assertion backed by
    at least ``min_review_stars`` review stars. Reports why records were dropped,
    because a silent filter here is the easiest way to build a biased dataset.
    """
    import gzip

    import pandas as pd

    rows: List[dict] = []
    dropped = {
        "not_snv": 0,
        "multiallelic": 0,
        "unusable_clnsig": 0,
        "below_star_threshold": 0,
    }

    with gzip.open(vcf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, vid, ref, alt, _qual, _filt, info = parts[:8]

            if "," in alt:
                dropped["multiallelic"] += 1
                continue
            if len(ref) != 1 or len(alt) != 1 or ref not in _BASES or alt not in _BASES:
                dropped["not_snv"] += 1
                continue

            fields = _parse_info(info)
            clnsig = _normalise_clnsig(fields.get("CLNSIG", ""))
            if clnsig in PATHOGENIC_CLNSIG:
                label = 1
            elif clnsig in BENIGN_CLNSIG:
                label = 0
            else:
                dropped["unusable_clnsig"] += 1
                continue

            stars = REVIEW_STATUS_STARS.get(fields.get("CLNREVSTAT", ""), 0)
            if stars < min_review_stars:
                dropped["below_star_threshold"] += 1
                continue

            rows.append(
                {
                    "variant_id": f"clinvar:{vid}",
                    "source": "clinvar",
                    "assembly": "hg38",
                    # ClinVar writes bare contig names; normalise to UCSC style.
                    "chrom": chrom if chrom.startswith("chr") else f"chr{chrom}",
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "label": label,
                    "gene": _gene_symbol(fields.get("GENEINFO", "")),
                    "review_stars": stars,
                    "clnsig": clnsig,
                }
            )

    frame = pd.DataFrame(rows)
    print(f"ClinVar: kept {len(frame)} SNVs at >={min_review_stars} review stars")
    print(f"  dropped: {dropped}")
    if len(frame):
        counts = frame["label"].value_counts().to_dict()
        print(f"  benign={counts.get(0, 0)} pathogenic={counts.get(1, 0)}")
    return frame


def build_clinvar_dataset(
    clinvar_config: Optional[config.ClinVarConfig] = None,
    force_download: bool = False,
) -> "pd.DataFrame":
    """Download, parse, filter and subsample ClinVar into the shared schema."""
    cfg = clinvar_config or config.DEFAULT_CLINVAR_CONFIG
    frame = parse_clinvar_vcf(
        download_clinvar(force=force_download),
        min_review_stars=cfg.min_review_stars,
    )
    if frame.empty:
        raise RuntimeError("No usable ClinVar records after filtering")

    if cfg.balance_classes:
        smallest = int(frame["label"].value_counts().min())
        frame = _sample_per_class(frame, smallest, cfg.seed)
        print(f"Balanced to {smallest} per class ({len(frame)} total)")

    if cfg.max_variants is not None and len(frame) > cfg.max_variants:
        # Sample within class so the cap does not reintroduce imbalance.
        per_class = cfg.max_variants // int(frame["label"].nunique())
        frame = _sample_per_class(frame, per_class, cfg.seed)
        print(f"Capped to {len(frame)} variants (max_variants={cfg.max_variants})")

    return frame.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)


def _sample_per_class(frame: "pd.DataFrame", n: int, seed: int) -> "pd.DataFrame":
    """Take up to ``n`` rows of each label, preserving the shared schema."""
    import pandas as pd

    parts = [
        group.sample(n=min(len(group), n), random_state=seed)
        for _, group in frame.groupby("label")
    ]
    return pd.concat(parts, ignore_index=True)


# --- BRCA1 saturation mutagenesis (Findlay et al. 2018) --------------------


def build_brca1_dataset(xlsx_path: Optional[str] = None) -> "pd.DataFrame":
    """Load the BRCA1 SGE table used by the Evo2 paper's zero-shot benchmark.

    Coordinates are GRCh37, which is why this dataset is scored against the
    vendored GRCh37 chr17 FASTA instead of hg38.
    """
    import pandas as pd

    path = xlsx_path or config.BRCA1_DMS_XLSX
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It ships with the evo2 repo cloned into the image."
        )

    frame = pd.read_excel(path, header=2)
    frame = frame[
        [
            "chromosome",
            "position (hg19)",
            "reference",
            "alt",
            "function.score.mean",
            "func.class",
        ]
    ].rename(
        columns={
            "chromosome": "chrom",
            "position (hg19)": "pos",
            "reference": "ref",
            "alt": "alt",
            "function.score.mean": "function_score",
            "func.class": "func_class",
        }
    )

    # Collapse to the two-class problem used in the paper: intermediate variants
    # are grouped with functional ones.
    frame["func_class"] = frame["func_class"].replace(["FUNC", "INT"], "FUNC/INT")

    frame = frame[frame["ref"].isin(_BASES) & frame["alt"].isin(_BASES)].copy()
    frame["chrom"] = frame["chrom"].astype(str).apply(
        lambda c: c if c.startswith("chr") else f"chr{c}"
    )
    frame["pos"] = frame["pos"].astype(int)
    frame["label"] = (frame["func_class"] == "LOF").astype(int)
    frame["source"] = "brca1_dms"
    frame["assembly"] = "hg19"
    frame["gene"] = "BRCA1"
    frame["review_stars"] = -1
    frame["clnsig"] = frame["func_class"]
    frame["variant_id"] = (
        "brca1:" + frame["chrom"] + ":" + frame["pos"].astype(str)
        + ":" + frame["ref"] + ">" + frame["alt"]
    )

    print(f"BRCA1 DMS: {len(frame)} SNVs")
    print(f"  {frame['func_class'].value_counts().to_dict()}")
    return frame.reset_index(drop=True)


# --- Splits ----------------------------------------------------------------


def _in_brca1_locus(frame: "pd.DataFrame") -> "pd.Series":
    import pandas as pd

    mask = pd.Series(False, index=frame.index)
    for assembly, (chrom, start, end) in config.BRCA1_LOCUS.items():
        mask |= (
            (frame["assembly"] == assembly)
            & (frame["chrom"] == chrom)
            & (frame["pos"] >= start)
            & (frame["pos"] <= end)
        )
    return mask


def assign_splits(
    clinvar: "pd.DataFrame",
    brca1: Optional["pd.DataFrame"] = None,
    train_config: Optional[config.TrainConfig] = None,
) -> "pd.DataFrame":
    """Combine the sources and assign train / val / test / benchmark splits.

    Splits are chromosome-disjoint rather than random: nearby variants share
    sequence context within an 8 kb window, so a random split would leak
    near-duplicate windows across the boundary and inflate validation scores.
    """
    import pandas as pd

    cfg = train_config or config.DEFAULT_TRAIN_CONFIG
    frame = clinvar.copy()

    if cfg.exclude_brca1_from_training:
        in_locus = _in_brca1_locus(frame)
        if in_locus.any():
            print(
                f"Excluding {int(in_locus.sum())} ClinVar variants in the BRCA1 "
                "locus to keep the DMS benchmark out of distribution"
            )
            frame = frame[~in_locus].copy()

    val = set(cfg.val_chromosomes)
    test = set(cfg.test_chromosomes)
    overlap = val & test
    if overlap:
        raise ValueError(f"Validation and test chromosomes overlap: {sorted(overlap)}")

    frame["split"] = "train"
    frame.loc[frame["chrom"].isin(val), "split"] = "val"
    frame.loc[frame["chrom"].isin(test), "split"] = "test"

    frames = [frame]
    if brca1 is not None and len(brca1):
        benchmark = brca1.copy()
        benchmark["split"] = "benchmark"
        frames.append(benchmark)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    print("Split sizes:")
    for split, group in combined.groupby("split"):
        positives = int(group["label"].sum())
        print(
            f"  {split:<10} n={len(group):<7} "
            f"positive={positives} ({positives / max(len(group), 1):.1%})"
        )

    if not (combined["split"] == "val").any():
        raise ValueError(
            f"Validation split is empty. Chromosomes {sorted(val)} matched no "
            "variants; pick chromosomes present in the ClinVar table."
        )
    return combined


# --- Persistence -----------------------------------------------------------

COLUMNS = [
    "variant_id",
    "source",
    "assembly",
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "gene",
    "review_stars",
    "clnsig",
    "split",
]


def dataset_path(name: str = "variants") -> str:
    return f"{config.DATASETS_DIR}/{name}.parquet"


def save_dataset(frame: "pd.DataFrame", name: str = "variants") -> str:
    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    path = dataset_path(name)
    keep = [c for c in COLUMNS if c in frame.columns]
    extra = [c for c in ("function_score", "func_class") if c in frame.columns]
    frame[keep + extra].to_parquet(path, index=False)
    print(f"Wrote {len(frame)} variants to {path}")
    return path


def load_dataset(name: str = "variants") -> "pd.DataFrame":
    import pandas as pd

    path = dataset_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the `build-datasets` stage first."
        )
    return pd.read_parquet(path)
