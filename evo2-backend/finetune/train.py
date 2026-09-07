"""Train the classifier head on cached Evo2 features and evaluate it.

Every run reports the zero-shot delta-likelihood AUROC alongside the trained
head's, on the same rows. Without that baseline there is no way to tell whether
the head learned anything beyond what thresholding the raw score already gave.
"""

import json
import os
import shutil
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import config, data, features
from .config import SCALAR_FEATURE_NAMES, FeatureConfig, TrainConfig
from .head import Standardizer, TrainedHead, VariantHead


def _zero_shot_column(feature_config: FeatureConfig) -> int:
    """Index of ``delta_score_full`` within the assembled feature matrix."""
    return feature_config.n_embedding_features + SCALAR_FEATURE_NAMES.index(
        "delta_score_full"
    )


def assemble_splits(
    feature_config: Optional[FeatureConfig] = None,
    dataset_name: str = "variants",
) -> Dict[str, Tuple["np.ndarray", "np.ndarray"]]:
    """Join cached features against the variant table and group them by split."""
    cfg = feature_config or config.DEFAULT_FEATURE_CONFIG

    variant_ids, X, y = features.load_features(cfg)
    table = data.load_dataset(dataset_name).set_index("variant_id")

    # A duplicated id would make the .loc lookup below return more rows than it
    # was asked for, silently misaligning features against splits.
    if not table.index.is_unique:
        duplicated = table.index[table.index.duplicated()].unique()[:5].tolist()
        raise RuntimeError(
            f"{dataset_name}.parquet has duplicate variant_ids, e.g. {duplicated}. "
            "Rebuild the dataset before training."
        )

    known = table.index
    mask = np.isin(variant_ids, known)
    if not mask.all():
        print(
            f"Warning: {int((~mask).sum())} feature rows have no matching row in "
            f"{dataset_name}.parquet and were dropped"
        )
    variant_ids, X, y = variant_ids[mask], X[mask], y[mask]

    splits = table.loc[variant_ids, "split"].to_numpy()

    grouped = {}
    for name in ("train", "val", "test", "benchmark"):
        idx = np.flatnonzero(splits == name)
        if len(idx):
            grouped[name] = (X[idx], y[idx])
            positives = int(y[idx].sum())
            print(
                f"  {name:<10} n={len(idx):<7} positive={positives} "
                f"({positives / len(idx):.1%})"
            )

    for required in ("train", "val"):
        if required not in grouped:
            raise RuntimeError(
                f"Split {required!r} is empty. Check the chromosome assignments "
                "in TrainConfig against the variants you extracted."
            )
        # Early stopping and threshold selection both need two classes; a
        # single-class split would give a NaN AUROC and no usable threshold.
        if len(np.unique(grouped[required][1])) < 2:
            raise RuntimeError(
                f"Split {required!r} contains only one class. Pick different "
                "val/test chromosomes in TrainConfig."
            )
    return grouped


def _metrics(y_true, scores, threshold: Optional[float] = None) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        roc_auc_score,
    )

    # A split with a single class makes AUROC undefined.
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(len(y_true))}

    out = {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "n": int(len(y_true)),
    }
    if threshold is not None:
        predicted = (scores >= threshold).astype(int)
        out["accuracy"] = float(accuracy_score(y_true, predicted))
        out["threshold"] = float(threshold)
    return out


def _youden_threshold(y_true, scores) -> float:
    """Threshold maximising TPR - FPR, the criterion the zero-shot path used."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def train_head(
    feature_config: Optional[FeatureConfig] = None,
    train_config: Optional[TrainConfig] = None,
    dataset_name: str = "variants",
    run_name: Optional[str] = None,
) -> TrainedHead:
    feature_cfg = feature_config or config.DEFAULT_FEATURE_CONFIG
    cfg = train_config or config.DEFAULT_TRAIN_CONFIG

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Assembling splits ...")
    splits = assemble_splits(feature_cfg, dataset_name)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    standardizer = Standardizer.fit(X_train)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} with {X_train.shape[1]} features")

    def to_tensors(X, y):
        return (
            torch.from_numpy(standardizer.transform(X)).to(device),
            torch.from_numpy(y.astype(np.float32)).to(device),
        )

    Xt, yt = to_tensors(X_train, y_train)
    Xv, yv = to_tensors(X_val, y_val)

    model = VariantHead(
        n_features=X_train.shape[1],
        head=cfg.head,
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
    ).to(device)

    pos_weight = None
    if cfg.class_weighting:
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=device)
            print(f"Class weighting: pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_auroc, best_state, best_epoch = -1.0, None, -1
    started = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(len(Xt), device=device)
        epoch_loss = 0.0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(Xv)).cpu().numpy()
        val_auroc = _metrics(y_val, val_scores)["auroc"]

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == cfg.max_epochs - 1:
            print(
                f"  epoch {epoch:>3} loss={epoch_loss / len(order):.4f} "
                f"val_auroc={val_auroc:.4f} (best {best_auroc:.4f} @ {best_epoch})"
            )

        if epoch - best_epoch >= cfg.patience:
            print(f"  early stop at epoch {epoch} (no gain for {cfg.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Trained in {time.time() - started:.1f}s; best val AUROC {best_auroc:.4f}")

    # Threshold is fitted on validation, never on test.
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(Xv)).cpu().numpy()
    threshold = _youden_threshold(y_val, val_scores)
    print(f"Decision threshold (Youden's J on val): {threshold:.4f}")

    trained = TrainedHead(
        model=model,
        standardizer=standardizer,
        threshold=threshold,
        feature_config=feature_cfg,
    )

    # --- evaluation, trained head vs the zero-shot score it replaces
    zero_shot_col = _zero_shot_column(feature_cfg)
    report: Dict[str, Dict] = {}
    for name, (X, y) in splits.items():
        probs = trained.predict_proba(X)
        # Pathogenic variants have a *more negative* delta likelihood, so the
        # baseline score is negated to point the same way as the head's output.
        baseline = -X[:, zero_shot_col]
        report[name] = {
            "head": _metrics(y, probs, threshold),
            "zero_shot": _metrics(y, baseline),
        }
        print(
            f"{name:<10} head AUROC={report[name]['head']['auroc']:.4f}  "
            f"zero-shot AUROC={report[name]['zero_shot']['auroc']:.4f}"
        )

    trained.metrics = {
        "splits": report,
        "best_val_auroc": best_auroc,
        "best_epoch": best_epoch,
        "n_features": int(X_train.shape[1]),
        "train_config": vars(cfg),
        "feature_tag": feature_cfg.tag(),
    }

    run = run_name or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = f"{config.RUNS_DIR}/{run}"
    trained.save(run_dir)
    print(f"Run directory: {run_dir}")
    return trained


def publish(run_name: str) -> str:
    """Point the inference endpoint at a finished run.

    Copies rather than symlinks so the endpoint keeps serving a coherent
    checkpoint even if the source run directory is later deleted.
    """
    source = f"{config.RUNS_DIR}/{run_name}"
    if not os.path.exists(os.path.join(source, "head.pt")):
        raise FileNotFoundError(f"No head.pt in {source}")

    target = config.ACTIVE_RUN_DIR
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(source, target)

    with open(os.path.join(target, "PUBLISHED_FROM"), "w") as handle:
        handle.write(f"{run_name}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Published {run_name} to {target}; the endpoint will serve it on next start")
    return target


def summarise_runs() -> None:
    """Print the metrics of every completed run."""
    if not os.path.exists(config.RUNS_DIR):
        print("No runs yet")
        return

    for run in sorted(os.listdir(config.RUNS_DIR)):
        metrics_path = f"{config.RUNS_DIR}/{run}/metrics.json"
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        print(f"\n{run}")
        for split, entry in metrics.get("splits", {}).items():
            print(
                f"  {split:<10} head={entry['head']['auroc']:.4f} "
                f"zero-shot={entry['zero_shot']['auroc']:.4f} n={entry['head']['n']}"
            )
