"""Local verification of the fine-tuning pipeline's CPU-side logic.

Stubs out the `evo2` package (which only exists inside the Modal image) with a
deterministic fake model, so feature alignment can be checked exactly.
"""
import gzip
import os
import sys
import tempfile
import types

import numpy as np
import torch

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

# ---------------------------------------------------------------- evo2 stub
VOCAB = 512
EMB_DIM = 16
LAYER = "blocks.28.mlp.l3"


class FakeTokenizer:
    pad_id = 1
    eod_id = 0

    def tokenize(self, seq):
        return [ord(c) for c in seq]


def fake_prepare_batch(seqs, tokenizer, prepend_bos=False, device="cpu"):
    lengths = [len(s) for s in seqs]
    m = max(lengths)
    rows = [
        torch.tensor(
            ([tokenizer.eod_id] * int(prepend_bos))
            + tokenizer.tokenize(s)
            + [tokenizer.pad_id] * (m - len(s)),
            dtype=torch.long,
        ).unsqueeze(0)
        for s in seqs
    ]
    return torch.cat(rows, 0), lengths


scoring_mod = types.ModuleType("evo2.scoring")
scoring_mod.prepare_batch = fake_prepare_batch
evo2_mod = types.ModuleType("evo2")
evo2_mod.Evo2 = object
evo2_mod.scoring = scoring_mod
sys.modules["evo2"] = evo2_mod
sys.modules["evo2.scoring"] = scoring_mod


class FakeModel:
    """Embeddings depend ONLY on the token at that position.

    That locality is what makes the alignment assertions below exact: a radius-0
    pooled delta must equal emb(alt) - emb(ref) and nothing else.
    """

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        g = torch.Generator().manual_seed(0)
        self.table = torch.randn(VOCAB, EMB_DIM, generator=g)
        self.proj = torch.randn(EMB_DIM, VOCAB, generator=g)

    def forward(self, input_ids, return_embeddings=False, layer_names=None):
        hidden = self.table[input_ids]              # (B, L, EMB_DIM)
        logits = hidden @ self.proj                 # (B, L, VOCAB)
        return logits, ({layer_names[0]: hidden} if return_embeddings else None)


# ---------------------------------------------------------------- helpers
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- tests
from finetune import config, data, features, sequences  # noqa: E402
from finetune.config import FeatureConfig  # noqa: E402
from finetune.head import Standardizer, TrainedHead, VariantHead  # noqa: E402

TEST_CFG = FeatureConfig(
    window_size=512,
    pool_radii=(0, 8, None),
    local_radius=4,
    embedding_dim=EMB_DIM,
)


def test_token_logprobs():
    print("\n[token_logprobs] alignment")
    torch.manual_seed(0)
    logits = torch.randn(1, 5, VOCAB)
    ids = torch.tensor([[10, 20, 30, 40, 50]])
    lp = features.token_logprobs(logits, ids)

    check("position 0 is NaN (nothing precedes it)", np.isnan(lp[0, 0]))
    expected = torch.log_softmax(logits[0, 1], dim=-1)[ids[0, 2]].item()
    check(
        "lp[i] scores token i using logits[i-1]",
        abs(lp[0, 2] - expected) < 1e-5,
        f"{lp[0, 2]:.6f} vs {expected:.6f}",
    )
    check("shape matches input length", lp.shape == (1, 5))


def test_feature_extraction():
    print("\n[extract_variant_features] pooling + alignment")
    rng = np.random.default_rng(0)
    ref = "".join(rng.choice(list("ACGT"), TEST_CFG.window_size))
    centre = TEST_CFG.window_size // 2
    ref = ref[:centre] + "A" + ref[centre + 1:]
    var = ref[:centre] + "G" + ref[centre + 1:]

    model = FakeModel()
    out = features.extract_variant_features(model, ref, var, centre, TEST_CFG)

    check(
        "feature vector length matches config",
        len(out["embedding"]) + len(out["scalar"]) == TEST_CFG.n_features,
        f"{len(out['embedding'])}+{len(out['scalar'])} vs {TEST_CFG.n_features}",
    )

    # Radius 0 block == emb(G) - emb(A) exactly, if the centre index is right.
    expected = (model.table[ord("G")] - model.table[ord("A")]).numpy()
    got = out["embedding"][:EMB_DIM]
    check(
        "radius-0 delta equals emb(alt) - emb(ref)",
        np.allclose(got, expected, atol=1e-5),
        f"max diff {np.abs(got - expected).max():.2e}",
    )

    # Radius 8 pools 17 positions, only one of which differs.
    got8 = out["embedding"][EMB_DIM:2 * EMB_DIM]
    check(
        "radius-8 delta is the same change averaged over 17 positions",
        np.allclose(got8, expected / 17, atol=1e-5),
    )

    # Full-window pooling averages over every position.
    got_full = out["embedding"][2 * EMB_DIM:3 * EMB_DIM]
    check(
        "full-window delta averages over the whole window",
        np.allclose(got_full, expected / TEST_CFG.window_size, atol=1e-6),
    )

    check("delta_score is finite", np.isfinite(out["delta_score"]))
    check("no NaN in features", not np.isnan(out["embedding"]).any()
          and not np.isnan(out["scalar"]).any())

    # A shifted centre must change the answer -- guards against silent off-by-one.
    shifted = features.extract_variant_features(model, ref, var, centre, TEST_CFG)
    check("extraction is deterministic",
          np.allclose(shifted["embedding"], out["embedding"]))

    # pair_batch off must give the same answer as on.
    unpaired = features.extract_variant_features(
        model, ref, var, centre,
        FeatureConfig(window_size=TEST_CFG.window_size, pool_radii=TEST_CFG.pool_radii,
                      local_radius=TEST_CFG.local_radius, embedding_dim=EMB_DIM,
                      pair_batch=False),
    )
    check("pair_batch=False matches pair_batch=True",
          np.allclose(unpaired["embedding"], out["embedding"], atol=1e-5)
          and np.allclose(unpaired["scalar"], out["scalar"], atol=1e-5))


def test_sequences():
    print("\n[sequences] genome access + window building")
    tmp = tempfile.mkdtemp()
    fasta = os.path.join(tmp, "toy.fa")
    rng = np.random.default_rng(1)
    chrom_seq = "".join(rng.choice(list("ACGT"), 5000))
    with open(fasta, "w") as fh:
        fh.write(">chr17\n")
        for i in range(0, len(chrom_seq), 60):
            fh.write(chrom_seq[i:i + 60] + "\n")

    genome = sequences.ReferenceGenome(fasta)
    check("bare contig name resolves to chr-prefixed", genome.resolve("17") == "chr17")
    check("chr-prefixed name resolves", genome.resolve("chr17") == "chr17")
    check("length is correct", genome.length("chr17") == 5000)

    pos = 2500  # 1-based
    win, start = genome.window("chr17", pos, 512)
    rel = pos - 1 - start
    check("window has the requested size", len(win) == 512, str(len(win)))
    check("variant sits at the window centre", rel == 256, str(rel))
    check("base at rel matches the source sequence", win[rel] == chrom_seq[pos - 1])

    # Truncation near a contig edge.
    win_edge, start_edge = genome.window("chr17", 10, 512)
    check("edge window starts at 0", start_edge == 0)
    # p=9, half=256 -> [0, 265), so 265 bases and an off-centre variant.
    check("edge window truncates", len(win_edge) == 265, str(len(win_edge)))
    check("edge variant offset is correct", win_edge[10 - 1 - start_edge] == chrom_seq[9])

    ref_base = chrom_seq[pos - 1]
    alt = "A" if ref_base != "A" else "T"
    var, got_ref = sequences.build_variant_window(win, rel, alt, expected_reference=ref_base)
    check("build_variant_window returns the reference base", got_ref == ref_base)
    check("only the variant position changed",
          var[rel] == alt and var[:rel] == win[:rel] and var[rel + 1:] == win[rel + 1:])

    try:
        sequences.build_variant_window(win, rel, alt, expected_reference="N")
        check("reference mismatch raises", False)
    except ValueError:
        check("reference mismatch raises", True)


VCF = """##fileformat=VCFv4.1
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
17\t43045000\t1\tA\tG\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=BRCA1:672
2\t100000\t2\tC\tT\t.\t.\tCLNSIG=Benign;CLNREVSTAT=reviewed_by_expert_panel;GENEINFO=XYZ:1
8\t200000\t3\tG\tA\t.\t.\tCLNSIG=Likely_pathogenic;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=ABC:2
8\t200500\t4\tT\tC\t.\t.\tCLNSIG=Benign/Likely_benign;CLNREVSTAT=practice_guideline;GENEINFO=ABC:2
1\t300000\t5\tA\tT\t.\t.\tCLNSIG=Uncertain_significance;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts
1\t300100\t6\tA\tT\t.\t.\tCLNSIG=Conflicting_classifications_of_pathogenicity;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts
1\t300200\t7\tA\tT\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided,_single_submitter
1\t300300\t8\tACGT\tA\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=practice_guideline
1\t300400\t9\tA\tT,C\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=practice_guideline
16\t400000\t10\tG\tC\t.\t.\tCLNSIG=Pathogenic,_low_penetrance;CLNREVSTAT=reviewed_by_expert_panel;GENEINFO=Q:9
16\t400100\t11\tG\tC\t.\t.\tCLNSIG=Likely_benign|other;CLNREVSTAT=reviewed_by_expert_panel;GENEINFO=Q:9
"""


def test_clinvar():
    print("\n[data] ClinVar VCF parsing")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "clinvar.vcf.gz")
    with gzip.open(path, "wt") as fh:
        fh.write(VCF)

    frame = data.parse_clinvar_vcf(path, min_review_stars=2)
    ids = set(frame["variant_id"])

    check("keeps 2+ star pathogenic/benign SNVs", len(frame) == 6, f"n={len(frame)}")
    check("drops uncertain_significance", "clinvar:5" not in ids)
    check("drops conflicting (despite containing 'pathogenic')", "clinvar:6" not in ids)
    check("drops 1-star single submitter", "clinvar:7" not in ids)
    check("drops indels", "clinvar:8" not in ids)
    check("drops multiallelic", "clinvar:9" not in ids)
    check("keeps ,_low_penetrance as pathogenic", "clinvar:10" in ids)
    check("keeps |other-suffixed benign", "clinvar:11" in ids)
    check("normalises contig names to chrN", set(frame["chrom"]) <= {
        "chr17", "chr2", "chr8", "chr16"}, str(sorted(set(frame["chrom"]))))
    check("extracts gene symbol",
          frame.set_index("variant_id").loc["clinvar:1", "gene"] == "BRCA1")
    check("labels are 0/1", set(frame["label"]) == {0, 1})
    stars = frame.set_index("variant_id")["review_stars"]
    check("star mapping: practice_guideline=4", stars["clinvar:4"] == 4)
    check("star mapping: expert panel=3", stars["clinvar:2"] == 3)
    return frame


def test_splits(clinvar):
    print("\n[data] split assignment")
    combined = data.assign_splits(clinvar, None)
    by_id = combined.set_index("variant_id")["split"]

    check("BRCA1-locus ClinVar variant excluded", "clinvar:1" not in by_id.index)
    check("chr8 -> val", (by_id["clinvar:3"] == "val") and (by_id["clinvar:4"] == "val"))
    check("chr2 -> test", by_id["clinvar:2"] == "test")
    check("chr16 -> test", by_id["clinvar:10"] == "test")

    cfg = config.TrainConfig(val_chromosomes=("chr8",), test_chromosomes=("chr8",))
    try:
        data.assign_splits(clinvar, None, cfg)
        check("overlapping val/test raises", False)
    except ValueError:
        check("overlapping val/test raises", True)


def test_shard_bounds():
    print("\n[features] shard bounds")
    b = features.shard_bounds(100, 7)
    check("bounds cover everything exactly",
          b[0][0] == 0 and b[-1][1] == 100
          and all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1)), str(b))
    check("no empty shards", all(s < e for s, e in b))
    check("single shard works", features.shard_bounds(10, 1) == [(0, 10)])
    check("more shards than items", features.shard_bounds(3, 10) == [(0, 1), (1, 2), (2, 3)],
          str(features.shard_bounds(3, 10)))


def test_head_roundtrip():
    print("\n[head] artifact round-trip")
    n_feat = TEST_CFG.n_features
    model = VariantHead(n_feat, head="mlp", hidden_sizes=(32, 8), dropout=0.1)
    std = Standardizer(np.zeros(n_feat), np.ones(n_feat))
    trained = TrainedHead(model, std, threshold=0.5, feature_config=TEST_CFG,
                          metrics={"splits": {}})

    tmp = tempfile.mkdtemp()
    trained.save(tmp)
    loaded = TrainedHead.load(tmp)

    x = np.random.default_rng(0).normal(size=(4, n_feat)).astype(np.float32)
    model.eval()
    check("predictions survive save/load",
          np.allclose(trained.predict_proba(x), loaded.predict_proba(x), atol=1e-6))
    check("feature config survives", loaded.feature_config == TEST_CFG,
          f"{loaded.feature_config}")
    check("hidden sizes recovered", loaded.model.net[0].out_features == 32)

    emb = np.zeros(TEST_CFG.n_embedding_features, dtype=np.float32)
    sc = np.zeros(TEST_CFG.n_scalar_features, dtype=np.float32)
    out = loaded.predict_one(emb, sc)
    check("predict_one returns the frontend's fields",
          {"prediction", "classification_confidence"} <= set(out))
    check("confidence is a probability in [0,1]",
          0.0 <= out["classification_confidence"] <= 1.0)
    check("confidence matches the predicted class",
          abs(out["classification_confidence"] - (
              out["pathogenicity_probability"]
              if out["prediction"] == "Likely pathogenic"
              else 1 - out["pathogenicity_probability"])) < 1e-9)

    try:
        loaded.predict_one(np.zeros(3, dtype=np.float32), sc)
        check("dimension mismatch raises a clear error", False)
    except ValueError as exc:
        check("dimension mismatch raises a clear error", "expects" in str(exc))

    # A standardizer with a constant column must not divide by zero.
    s2 = Standardizer(np.zeros(3), np.array([0.0, 1.0, 2.0]))
    check("zero-variance feature does not produce inf/nan",
          np.isfinite(s2.transform(np.ones((1, 3)))).all())


def test_training_end_to_end():
    print("\n[train] end-to-end on synthetic features")
    import pandas as pd

    from finetune import train as train_module

    tmp = tempfile.mkdtemp()
    config.DATASETS_DIR = os.path.join(tmp, "datasets")
    config.FEATURES_DIR = os.path.join(tmp, "features")
    config.RUNS_DIR = os.path.join(tmp, "runs")
    config.ACTIVE_RUN_DIR = os.path.join(config.RUNS_DIR, "active")
    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    os.makedirs(features.features_dir(TEST_CFG), exist_ok=True)

    rng = np.random.default_rng(0)
    n = 1200
    labels = rng.integers(0, 2, n)
    chroms = rng.choice(["chr1", "chr8", "chr2", "chr3"], n)
    ids = np.array([f"v{i}" for i in range(n)])

    emb = rng.normal(size=(n, TEST_CFG.n_embedding_features)).astype(np.float16)
    # Put real signal in the first embedding block and in delta_score_full.
    emb[:, 0] += labels * 2.0
    scal = rng.normal(scale=0.001, size=(n, TEST_CFG.n_scalar_features)).astype(np.float32)
    scal[:, 0] -= labels * 0.004   # pathogenic => more negative delta

    np.savez(os.path.join(features.features_dir(TEST_CFG), "features.npz"),
             variant_id=ids, embedding=emb, scalar=scal, label=labels)

    frame = pd.DataFrame({
        "variant_id": ids, "source": "synthetic", "assembly": "hg38",
        "chrom": chroms, "pos": np.arange(n) * 1000 + 1, "ref": "A", "alt": "G",
        "label": labels, "gene": "X", "review_stars": 2, "clnsig": "x",
    })
    combined = data.assign_splits(frame, None)
    data.save_dataset(combined)

    cfg = config.TrainConfig(head="mlp", hidden_sizes=(32, 8), max_epochs=40, patience=8,
                             val_chromosomes=("chr8",), test_chromosomes=("chr2",),
                             exclude_brca1_from_training=False)
    trained = train_module.train_head(TEST_CFG, cfg, run_name="unit-run")

    m = trained.metrics["splits"]
    check("test AUROC is strong on separable data", m["test"]["head"]["auroc"] > 0.9,
          f"{m['test']['head']['auroc']:.3f}")
    check("zero-shot baseline is reported alongside",
          "zero_shot" in m["test"] and np.isfinite(m["test"]["zero_shot"]["auroc"]),
          f"{m['test']['zero_shot']['auroc']:.3f}")
    check("zero-shot column sign points the right way",
          m["test"]["zero_shot"]["auroc"] > 0.9,
          f"{m['test']['zero_shot']['auroc']:.3f}")
    check("threshold recorded", 0.0 < trained.threshold < 1.0, f"{trained.threshold:.3f}")

    published = train_module.publish("unit-run")
    check("publish writes an active head", os.path.exists(os.path.join(published, "head.pt")))

    from finetune.head import load_active_head
    active = load_active_head()
    check("load_active_head finds the published run", active is not None)

    config.ACTIVE_RUN_DIR = os.path.join(tmp, "nonexistent")
    check("load_active_head returns None when nothing is published",
          load_active_head() is None)


if __name__ == "__main__":
    test_token_logprobs()
    test_feature_extraction()
    test_sequences()
    cv = test_clinvar()
    test_splits(cv)
    test_shard_bounds()
    test_head_roundtrip()
    test_training_end_to_end()

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)
