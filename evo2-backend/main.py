import modal
from pydantic import BaseModel

from common import VOLUMES, data_volume, evo2_image
from finetune import config as ft_config


class VariantRequest(BaseModel):
    variant_position: int
    alternative: str
    genome: str
    chromosome: str


app = modal.App("variant-analysis-evo2", image=evo2_image)

WINDOW_SIZE = ft_config.DEFAULT_FEATURE_CONFIG.window_size

# Zero-shot decision boundary, fitted on 500 BRCA1 SNVs by run_brca1_analysis
# below. Used when no fine-tuned head has been published.
ZERO_SHOT_THRESHOLD = -0.0009178519
ZERO_SHOT_LOF_STD = 0.0015140239
ZERO_SHOT_FUNC_STD = 0.0009016589


@app.function(gpu="H100", volumes=VOLUMES, timeout=1000)
def run_brca1_analysis():
    import base64
    from io import BytesIO
    from Bio import SeqIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import roc_auc_score, roc_curve

    from finetune.loader import load_evo2

    print("Loading evo2 model...")
    model = load_evo2('evo2_7b')
    print("Evo2 model loaded")

    brca1_df = pd.read_excel(
        '/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx',
        header=2,
    )
    brca1_df = brca1_df[[
        'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',
    ]]

    brca1_df.rename(columns={
        'chromosome': 'chrom',
        'position (hg19)': 'pos',
        'reference': 'ref',
        'alt': 'alt',
        'function.score.mean': 'score',
        'func.class': 'class',
    }, inplace=True)

    # Convert to two-class system
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')

    with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break

    # Build mappings of unique reference sequences
    ref_seqs = []
    ref_seq_to_index = {}

    # Parse sequences and store indexes
    ref_seq_indexes = []
    var_seqs = []

    brca1_subset = brca1_df.iloc[:500].copy()

    for _, row in brca1_subset.iterrows():
        p = row["pos"] - 1  # Convert to 0-indexed position
        full_seq = seq_chr17

        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + \
            row["alt"] + ref_seq[snv_pos_in_ref+1:]

        # Get or create index for reference sequence
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)

        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)

    ref_seq_indexes = np.array(ref_seq_indexes)

    print(
        f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
    ref_scores = model.score_sequences(ref_seqs)

    print(
        f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
    var_scores = model.score_sequences(var_seqs)

    # Subtract score of corresponding reference sequences from scores of variant sequences
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]

    # Add delta scores to dataframe
    brca1_subset[f'evo2_delta_score'] = delta_scores

    y_true = (brca1_subset['class'] == 'LOF')
    auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])

    # --- Calculate threshold START
    y_true = (brca1_subset["class"] == "LOF")

    fpr, tpr, thresholds = roc_curve(y_true, -brca1_subset["evo2_delta_score"])

    optimal_idx = (tpr - fpr).argmax()

    optimal_threshold = -thresholds[optimal_idx]

    lof_scores = brca1_subset.loc[brca1_subset["class"]
                                  == "LOF", "evo2_delta_score"]
    func_scores = brca1_subset.loc[brca1_subset["class"]
                                   == "FUNC/INT", "evo2_delta_score"]

    lof_std = lof_scores.std()
    func_std = func_scores.std()

    confidence_params = {
        "threshold": optimal_threshold,
        "lof_std": lof_std,
        "func_std": func_std
    }

    print("Confidence params:", confidence_params)

    # --- Calculate threshold END

    plt.figure(figsize=(4, 2))

    # Plot stripplot of distributions
    p = sns.stripplot(
        data=brca1_subset,
        x='evo2_delta_score',
        y='class',
        hue='class',
        order=['FUNC/INT', 'LOF'],
        palette=['#777777', 'C3'],
        size=2,
        jitter=0.3,
    )

    # Mark medians from each distribution
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'visible': False},
                medianprops={'color': 'k', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                x="evo2_delta_score",
                y="class",
                data=brca1_subset,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.ylabel('BRCA1 SNV class')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {'variants': brca1_subset.to_dict(orient="records"), "plot": plot_data, "auroc": auroc}


@app.function()
def brca1_example():
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    print("Running BRCA1 variant analysis with Evo2...")

    # Run inference
    result = run_brca1_analysis.remote()

    if "plot" in result:
        plot_data = base64.b64decode(result["plot"])
        with open("brca1_analysis_plot.png", "wb") as f:
            f.write(plot_data)

        img = mpimg.imread(BytesIO(plot_data))
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def zero_shot_prediction(delta_score: float) -> dict:
    """Classify from the raw delta-likelihood score alone.

    The confidence is a heuristic — distance past the threshold in units of the
    corresponding class's standard deviation — not a probability. The fine-tuned
    head replaces it with a calibrated one.
    """
    if delta_score < ZERO_SHOT_THRESHOLD:
        prediction = "Likely pathogenic"
        confidence = min(
            1.0, abs(delta_score - ZERO_SHOT_THRESHOLD) / ZERO_SHOT_LOF_STD
        )
    else:
        prediction = "Likely benign"
        confidence = min(
            1.0, abs(delta_score - ZERO_SHOT_THRESHOLD) / ZERO_SHOT_FUNC_STD
        )

    return {
        "prediction": prediction,
        "classification_confidence": float(confidence),
    }


@app.cls(gpu="H100", volumes=VOLUMES, max_containers=3, retries=2, scaledown_window=120)
class Evo2Model:
    @modal.enter()
    def load_evo2_model(self):
        from finetune.head import load_active_head
        from finetune.loader import load_evo2

        print("Loading evo2 model...")
        self.model = load_evo2(ft_config.MODEL_NAME)
        print("Evo2 model loaded")

        # A published head is optional: without one the endpoint serves the
        # zero-shot score exactly as before.
        data_volume.reload()
        self.head = load_active_head(device="cuda")
        if self.head is None:
            print("No fine-tuned head published; serving zero-shot predictions")
            self.feature_config = ft_config.DEFAULT_FEATURE_CONFIG
        else:
            metrics = self.head.metrics.get("splits", {})
            print(
                "Loaded fine-tuned head "
                f"(benchmark AUROC "
                f"{metrics.get('benchmark', {}).get('head', {}).get('auroc', float('nan')):.4f})"
            )
            # Always build features with the config the head was trained on.
            self.feature_config = self.head.feature_config

    @modal.fastapi_endpoint(method="POST")
    def analyze_single_variant(self, request: VariantRequest):
        from finetune.features import extract_variant_features
        from finetune.sequences import build_variant_window, fetch_window_ucsc

        print(
            f"Analyzing {request.chromosome}:{request.variant_position} "
            f">{request.alternative} on {request.genome}"
        )

        window_seq, seq_start = fetch_window_ucsc(
            position=request.variant_position,
            genome=request.genome,
            chromosome=request.chromosome,
            window_size=self.feature_config.window_size,
        )

        relative_pos = request.variant_position - 1 - seq_start
        if relative_pos < 0 or relative_pos >= len(window_seq):
            raise ValueError(
                f"Variant position {request.variant_position} is outside the fetched "
                f"window (start={seq_start + 1}, end={seq_start + len(window_seq)})"
            )

        var_seq, reference = build_variant_window(
            window_seq, relative_pos, request.alternative
        )
        print(f"Reference base at {request.variant_position}: {reference}")

        # One paired forward pass yields both the likelihood scores and the
        # embeddings, so the trained head costs no extra GPU time.
        extracted = extract_variant_features(
            self.model, window_seq, var_seq, relative_pos, self.feature_config
        )
        delta_score = float(extracted["delta_score"])

        result = {
            "position": request.variant_position,
            "reference": reference,
            "alternative": request.alternative,
            "delta_score": delta_score,
        }

        zero_shot = zero_shot_prediction(delta_score)
        if self.head is None:
            result.update(zero_shot)
            result["model"] = "zero-shot"
        else:
            result.update(
                self.head.predict_one(extracted["embedding"], extracted["scalar"])
            )
            result["model"] = "fine-tuned"
            # Kept so the UI can still show what the raw score alone would say.
            result["zero_shot"] = zero_shot

        print(f"Result: {result['prediction']} (delta_score={delta_score:.6f})")
        return result


@app.local_entrypoint()
def main():
    # Example of how you'd call the deployed Modal Function from your client
    import requests

    evo2Model = Evo2Model()

    url = evo2Model.analyze_single_variant.web_url

    payload = {
        "variant_position": 43119628,
        "alternative": "G",
        "genome": "hg38",
        "chromosome": "chr17"
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)
