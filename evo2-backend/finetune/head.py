"""The trained classifier head and its serialised artifact.

The head is deliberately small: Evo2 stays frozen and only this sits on top of
its embeddings. Everything the inference endpoint needs to reproduce a
prediction — architecture, feature standardisation, decision threshold and the
feature config the vector was built with — travels in a single ``head.pt`` so a
checkpoint can never be paired with the wrong preprocessing.
"""

import json
import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from . import config
from .config import FeatureConfig

ARTIFACT_NAME = "head.pt"
METRICS_NAME = "metrics.json"


class VariantHead(nn.Module):
    """Linear or small MLP classifier over standardised Evo2 features."""

    def __init__(
        self,
        n_features: int,
        head: str = "mlp",
        hidden_sizes: Sequence[int] = (256, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.head = head

        if head == "linear":
            self.net = nn.Linear(n_features, 1)
        elif head == "mlp":
            layers = []
            in_dim = n_features
            for width in hidden_sizes:
                layers += [
                    nn.Linear(in_dim, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
                in_dim = width
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown head {head!r}; expected 'linear' or 'mlp'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``."""
        return self.net(x).squeeze(-1)


class Standardizer:
    """Per-feature mean/std normalisation, fitted on the training split only."""

    def __init__(self, mean, std):
        import numpy as np

        self.mean = np.asarray(mean, dtype=np.float32)
        # A constant feature would divide by zero; leave it at zero instead.
        std = np.asarray(std, dtype=np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, X) -> "Standardizer":
        return cls(X.mean(axis=0), X.std(axis=0))

    def transform(self, X):
        import numpy as np

        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(
            np.float32
        )


class TrainedHead:
    """A head plus everything needed to apply it to a fresh variant."""

    def __init__(
        self,
        model: VariantHead,
        standardizer: Standardizer,
        threshold: float,
        feature_config: FeatureConfig,
        metrics: Optional[Dict] = None,
    ):
        self.model = model
        self.standardizer = standardizer
        # Decision threshold on the *probability*, chosen on the validation split.
        self.threshold = float(threshold)
        self.feature_config = feature_config
        self.metrics = metrics or {}

    # -- application ------------------------------------------------------

    def predict_proba(self, X) -> "np.ndarray":
        import numpy as np

        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(self.standardizer.transform(X)).to(device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.atleast_1d(probs)

    def predict_one(self, embedding, scalar) -> Dict[str, float]:
        """Classify a single variant from :func:`features.extract_variant_features`.

        ``classification_confidence`` is the calibrated probability of whichever
        class was predicted, so it is a genuine probability rather than the
        distance-over-sigma heuristic the zero-shot path uses.
        """
        import numpy as np

        x = np.concatenate(
            [np.asarray(embedding, dtype=np.float32), np.asarray(scalar, dtype=np.float32)]
        ).reshape(1, -1)
        if x.shape[1] != self.model.n_features:
            raise ValueError(
                f"Feature vector has {x.shape[1]} dimensions but the head expects "
                f"{self.model.n_features}. The head was trained with a different "
                "FeatureConfig than the one used to build this vector."
            )

        probability = float(self.predict_proba(x)[0])
        is_pathogenic = probability >= self.threshold
        return {
            "prediction": "Likely pathogenic" if is_pathogenic else "Likely benign",
            "pathogenicity_probability": probability,
            "classification_confidence": (
                probability if is_pathogenic else 1.0 - probability
            ),
        }

    # -- persistence ------------------------------------------------------

    def save(self, run_dir: str) -> str:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, ARTIFACT_NAME)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "head": self.model.head,
                "n_features": self.model.n_features,
                "hidden_sizes": _infer_hidden_sizes(self.model),
                "mean": self.standardizer.mean,
                "std": self.standardizer.std,
                "threshold": self.threshold,
                "feature_config": _feature_config_to_dict(self.feature_config),
                "metrics": self.metrics,
            },
            path,
        )
        with open(os.path.join(run_dir, METRICS_NAME), "w") as handle:
            json.dump(self.metrics, handle, indent=2, default=float)
        print(f"Saved head to {path}")
        return path

    @classmethod
    def load(cls, run_dir: str, device: str = "cpu") -> "TrainedHead":
        path = os.path.join(run_dir, ARTIFACT_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained head at {path}")

        blob = torch.load(path, map_location=device, weights_only=False)
        model = VariantHead(
            n_features=blob["n_features"],
            head=blob["head"],
            hidden_sizes=blob.get("hidden_sizes", (256, 64)),
            dropout=0.0,  # inference: dropout off
        )
        model.load_state_dict(blob["state_dict"])
        model.to(device).eval()

        return cls(
            model=model,
            standardizer=Standardizer(blob["mean"], blob["std"]),
            threshold=blob["threshold"],
            feature_config=_feature_config_from_dict(blob["feature_config"]),
            metrics=blob.get("metrics", {}),
        )


def _infer_hidden_sizes(model: VariantHead):
    if model.head == "linear":
        return ()
    return tuple(
        layer.out_features
        for layer in model.net
        if isinstance(layer, nn.Linear)
    )[:-1]


def _feature_config_to_dict(cfg: FeatureConfig) -> Dict:
    return {
        "window_size": cfg.window_size,
        # JSON/torch round-trips turn None into null cleanly, tuples into lists.
        "pool_radii": list(cfg.pool_radii),
        "local_radius": cfg.local_radius,
        "embedding_layer": cfg.embedding_layer,
        "embedding_dim": cfg.embedding_dim,
        "pair_batch": cfg.pair_batch,
    }


def _feature_config_from_dict(blob: Dict) -> FeatureConfig:
    return FeatureConfig(
        window_size=blob["window_size"],
        pool_radii=tuple(blob["pool_radii"]),
        local_radius=blob["local_radius"],
        embedding_layer=blob["embedding_layer"],
        embedding_dim=blob["embedding_dim"],
        pair_batch=blob.get("pair_batch", True),
    )


def load_active_head(device: str = "cpu") -> Optional[TrainedHead]:
    """Load the head the endpoint should serve, or ``None`` if none is published.

    Returning ``None`` rather than raising is deliberate: the endpoint falls
    back to zero-shot scoring so it keeps working before the first training run.
    """
    try:
        return TrainedHead.load(config.ACTIVE_RUN_DIR, device=device)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - never take the endpoint down
        print(f"Could not load trained head from {config.ACTIVE_RUN_DIR}: {exc}")
        return None
