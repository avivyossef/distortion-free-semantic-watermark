"""ROC, AUROC, TPR at empirical FPR, mean ROC curves and paired retention."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def _scores(values: Sequence[float], label: str) -> "list[float]":
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("%s scores must be nonempty and finite" % label)
    return result


def roc_curve(clean: Sequence[float], watermarked: Sequence[float]) -> "list[dict[str, float]]":
    """Empirical ROC points ``{threshold, fpr, tpr}`` from ``inf`` down to ``-inf``.

    A sample is called watermarked when ``score >= threshold``. Thresholds are every
    distinct observed score plus the two infinities, so vertical and horizontal
    segments keep both endpoints.
    """
    negatives, positives = _scores(clean, "clean"), _scores(watermarked, "watermarked")
    thresholds = [math.inf, *sorted(set(negatives + positives), reverse=True), -math.inf]
    return [
        {
            "threshold": threshold,
            "fpr": sum(value >= threshold for value in negatives) / len(negatives),
            "tpr": sum(value >= threshold for value in positives) / len(positives),
        }
        for threshold in thresholds
    ]


def auroc(clean: Sequence[float], watermarked: Sequence[float]) -> float:
    """Area under the ROC curve; ties count one half."""
    negatives = np.asarray(_scores(clean, "clean"), dtype=np.float64)
    positives = np.asarray(_scores(watermarked, "watermarked"), dtype=np.float64)
    wins = ties = 0
    for value in positives:
        wins += int(np.count_nonzero(value > negatives))
        ties += int(np.count_nonzero(value == negatives))
    return (wins + 0.5 * ties) / (len(negatives) * len(positives))


def tpr_at_fpr(roc: Sequence[Mapping[str, float]], requested_fpr: float) -> "dict[str, float]":
    """Operating point at an empirical-FPR target, read from a ROC curve.

    Picks, among ROC points with ``fpr <= requested_fpr``, the one with the highest TPR
    (ties: the higher threshold). Returns ``requested_fpr``, ``threshold``,
    ``empirical_fpr`` and ``tpr``.
    """
    if not isinstance(requested_fpr, (int, float)) or not 0 <= float(requested_fpr) <= 1:
        raise ValueError("fixed FPR values must lie in [0, 1]")
    candidates = [row for row in roc if row["fpr"] <= float(requested_fpr)]
    chosen = max(candidates, key=lambda row: (row["tpr"], row["threshold"]))
    return {
        "requested_fpr": float(requested_fpr),
        "threshold": float(chosen["threshold"]),
        "empirical_fpr": float(chosen["fpr"]),
        "tpr": float(chosen["tpr"]),
    }


def clean_detection_roc(
    clean: Sequence[float], watermarked: Sequence[float], fixed_fprs: Sequence[float] = (0.01, 0.05, 0.20)
) -> "dict[str, Any]":
    """ROC, AUROC and TPR at each requested empirical FPR for one replicate."""
    roc = roc_curve(clean, watermarked)
    return {
        "sample_count": len(clean),
        "auc": auroc(clean, watermarked),
        "roc": roc,
        "fixed_fpr": [tpr_at_fpr(roc, requested) for requested in fixed_fprs],
    }


def interpolate_roc(roc: Sequence[Mapping[str, float]], grid: Sequence[float]) -> np.ndarray:
    """Resample one ROC curve to a common FPR grid (for averaging replicates).

    Both endpoints of vertical segments are kept; dropping the lower endpoints would
    connect the preceding point to the upper endpoint.
    """
    points = sorted((float(row["fpr"]), float(row["tpr"])) for row in roc)
    xs, ys = zip(*points)
    return np.interp(np.asarray(grid, dtype=np.float64), xs, ys)


def mean_roc(rocs: Sequence[Sequence[Mapping[str, float]]], grid: Sequence[float]) -> "dict[str, np.ndarray]":
    """Mean and standard deviation of replicate ROC curves on a common FPR grid."""
    curves = np.stack([interpolate_roc(roc, grid) for roc in rocs])
    return {"mean": curves.mean(axis=0), "sd": curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(curves.shape[1])}


def paired_retention(
    original_scores: Sequence[float], edited_scores: Sequence[float], threshold: float
) -> "dict[str, Any]":
    """Fraction of watermarked samples detected before an edit that stay detected after.

    ``original_scores[i]`` and ``edited_scores[i]`` are the same image before and after
    the edit; both are compared with the same frozen ``threshold``.
    """
    identity = np.asarray(original_scores, dtype=np.float64)
    attacked = np.asarray(edited_scores, dtype=np.float64)
    if identity.ndim != 1 or attacked.shape != identity.shape or identity.size == 0:
        raise ValueError("paired score vectors must have one shared nonempty shape")
    if not np.isfinite(identity).all() or not np.isfinite(attacked).all() or np.isnan(threshold):
        raise ValueError("paired scores must be finite and threshold cannot be NaN")
    detected = identity >= float(threshold)
    identity_count = int(np.sum(detected))
    retained_count = int(np.sum((attacked >= float(threshold)) & detected))
    return {
        "identity_detected_count": identity_count,
        "attacked_retained_count": retained_count,
        "paired_detection_retention": (
            float(retained_count / identity_count) if identity_count else float("nan")
        ),
    }
