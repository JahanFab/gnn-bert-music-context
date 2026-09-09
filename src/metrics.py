"""Multi-label metrics for Section 6 of the spec.

    Prec_k = TP_k / (TP_k + FP_k)
    Rec_k  = TP_k / (TP_k + FN_k)
    F1_k   = 2 Prec_k Rec_k / (Prec_k + Rec_k)
    Macro-F1 = mean_k F1_k          Micro-F1 pools TP/FP/FN globally
    AUC-PR   = mean_k average_precision_k

Tags with zero positives in a split have undefined precision/recall; they are
scored as 0 for F1 (`zero_division=0`) and excluded from the AUC-PR mean, which
is stated explicitly rather than left to sklearn's default behaviour.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


def binarize(probs: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Threshold may be a scalar or a per-tag vector of shape (K,)."""
    return (probs >= threshold).astype(np.int8)


def multilabel_scores(
    probs: np.ndarray, targets: np.ndarray, threshold: float | np.ndarray = 0.5
) -> dict:
    """Headline metrics at a given decision threshold."""
    y_true = targets.astype(np.int8)
    y_pred = binarize(probs, threshold)

    scores = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "auc_pr": macro_auc_pr(probs, y_true),
        "mean_predicted_tags": float(y_pred.sum(axis=1).mean()),
        "mean_true_tags": float(y_true.sum(axis=1).mean()),
    }
    return scores


def macro_auc_pr(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean average-precision over tags that actually occur in this split."""
    present = targets.sum(axis=0) > 0
    if not present.any():
        return 0.0
    aps = [
        average_precision_score(targets[:, k], probs[:, k])
        for k in np.flatnonzero(present)
    ]
    return float(np.mean(aps))


def per_tag_report(
    probs: np.ndarray,
    targets: np.ndarray,
    label_vocab: list[str],
    threshold: float | np.ndarray = 0.5,
) -> list[dict]:
    """Per-tag P/R/F1/AP plus support, sorted by descending support."""
    y_true = targets.astype(np.int8)
    y_pred = binarize(probs, threshold)
    thr = np.full(len(label_vocab), threshold) if np.isscalar(threshold) else np.asarray(threshold)

    rows = []
    for k, tag in enumerate(label_vocab):
        support = int(y_true[:, k].sum())
        rows.append(
            {
                "tag": tag,
                "support": support,
                "threshold": float(thr[k]),
                "precision": float(precision_score(y_true[:, k], y_pred[:, k], zero_division=0)),
                "recall": float(recall_score(y_true[:, k], y_pred[:, k], zero_division=0)),
                "f1": float(f1_score(y_true[:, k], y_pred[:, k], zero_division=0)),
                "ap": float(average_precision_score(y_true[:, k], probs[:, k])) if support else None,
            }
        )
    return sorted(rows, key=lambda r: -r["support"])


def tune_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.05,
    fallback: float = 0.5,
) -> np.ndarray:
    """Per-tag thresholds that maximise F1 -- fitted on VALIDATION data only.

    A single global 0.5 cut-off is a poor fit for heavily imbalanced tags, so the
    per-tag threshold is swept on validation and then frozen for the test run.
    Fitting it on test would leak the labels and inflate the reported score.
    """
    grid = np.arange(start, stop + 1e-9, step)
    thresholds = np.full(probs.shape[1], fallback, dtype=np.float32)

    for k in range(probs.shape[1]):
        if targets[:, k].sum() == 0:
            continue
        best_f1, best_t = -1.0, fallback
        for t in grid:
            f1 = f1_score(targets[:, k], (probs[:, k] >= t).astype(np.int8), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t
    return thresholds


# --------------------------------------------------------------------------- #
# Emotion regression (DEAM valence/arousal, spec Section 6)
# --------------------------------------------------------------------------- #
def regression_scores(preds: np.ndarray, targets: np.ndarray, names: tuple[str, ...] = ("valence", "arousal")) -> dict:
    """MAE and R^2 per dimension.

        MAE = (1/N) sum |v_i - v_hat_i|
        R^2 = 1 - sum (y_i - y_hat_i)^2 / sum (y_i - y_bar)^2

    Returns an empty dict when no sample in the split carries emotion targets,
    so callers can distinguish "not evaluated" from "scored zero".
    """
    if preds.size == 0 or targets.size == 0:
        return {}

    scores = {}
    for k, name in enumerate(names):
        residual = ((targets[:, k] - preds[:, k]) ** 2).sum()
        total = ((targets[:, k] - targets[:, k].mean()) ** 2).sum()
        scores[f"mae_{name}"] = float(np.abs(targets[:, k] - preds[:, k]).mean())
        scores[f"r2_{name}"] = float(1.0 - residual / total) if total > 0 else None
    return scores


# --------------------------------------------------------------------------- #
# Single-label metrics (Task 2 genre classification)
# --------------------------------------------------------------------------- #
def multiclass_scores(logits: np.ndarray, targets: np.ndarray, n_classes: int | None = None) -> dict:
    """Accuracy, macro/micro-F1 and top-3 accuracy for genre classification.

    Micro-F1 equals accuracy in the single-label case; both are reported because
    the spec's metric table asks for them alongside the multi-label results.
    """
    preds = logits.argmax(axis=1)
    n_classes = n_classes or int(max(targets.max() + 1, logits.shape[1]))
    labels = np.arange(n_classes)

    scores = {
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", labels=labels, zero_division=0)),
        "micro_f1": float(f1_score(targets, preds, average="micro", labels=labels, zero_division=0)),
        "macro_precision": float(
            precision_score(targets, preds, average="macro", labels=labels, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(targets, preds, average="macro", labels=labels, zero_division=0)
        ),
    }
    if n_classes > 3:
        try:
            scores["top3_accuracy"] = float(
                top_k_accuracy_score(targets, logits, k=3, labels=labels)
            )
        except ValueError:  # a class missing from this split
            scores["top3_accuracy"] = None
    return scores


def per_class_report(logits: np.ndarray, targets: np.ndarray, classes: list[str]) -> list[dict]:
    preds = logits.argmax(axis=1)
    labels = np.arange(len(classes))
    f1s = f1_score(targets, preds, average=None, labels=labels, zero_division=0)
    precs = precision_score(targets, preds, average=None, labels=labels, zero_division=0)
    recs = recall_score(targets, preds, average=None, labels=labels, zero_division=0)
    return [
        {
            "class": name,
            "support": int((targets == k).sum()),
            "precision": float(precs[k]),
            "recall": float(recs[k]),
            "f1": float(f1s[k]),
        }
        for k, name in enumerate(classes)
    ]


def confusion(logits: np.ndarray, targets: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(targets, logits.argmax(axis=1), labels=np.arange(n_classes))


# --------------------------------------------------------------------------- #
# Baseline B1 (spec Section 8): majority-class / random tag predictors
# --------------------------------------------------------------------------- #
def random_baseline(train_targets: np.ndarray, test_targets: np.ndarray, seed: int = 42) -> dict:
    """Sample each tag independently from its training prior."""
    rng = np.random.default_rng(seed)
    prior = train_targets.mean(axis=0)
    probs = rng.random((test_targets.shape[0], test_targets.shape[1]))
    preds = (probs < prior).astype(np.int8)
    scores = multilabel_scores(preds.astype(np.float32), test_targets, 0.5)
    scores["auc_pr"] = macro_auc_pr(np.tile(prior, (test_targets.shape[0], 1)), test_targets)
    return scores


def majority_baseline(train_targets: np.ndarray, test_targets: np.ndarray, top_n: int | None = None) -> dict:
    """Always predict the most frequent tags.

    `top_n` defaults to the rounded average number of tags per training clip, so
    the baseline emits a realistic number of predictions instead of all K.
    """
    prior = train_targets.mean(axis=0)
    if top_n is None:
        top_n = int(round(train_targets.sum(axis=1).mean()))
    top_n = max(1, min(top_n, train_targets.shape[1]))

    preds = np.zeros_like(test_targets, dtype=np.int8)
    preds[:, np.argsort(-prior)[:top_n]] = 1
    scores = multilabel_scores(preds.astype(np.float32), test_targets, 0.5)
    scores["auc_pr"] = macro_auc_pr(np.tile(prior, (test_targets.shape[0], 1)), test_targets)
    scores["top_n"] = top_n
    return scores
