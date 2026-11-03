"""Detector calibration: the boundary threshold delta and the score threshold tau.

The paper's activation rate is the fraction of clean *watermarked* images whose
representation lies within delta of at least one of the M hyperplanes. Every function
here takes the population it calibrates on explicitly (a margin matrix or a score
vector); the caller passes the watermarked margins to implement the paper's definition.
"""

from __future__ import annotations

import math

import numpy as np


def calibrate_delta(margins: np.ndarray, *, target_activation: float) -> float:
    """Choose delta so exactly the requested fraction has any margin at or below delta.

    ``margins`` is an ``(images, M)`` matrix of absolute hyperplane margins of the
    calibration population. Raises when ties prevent hitting the target exactly.
    """
    values = np.asarray(margins, dtype=np.float64)
    target = float(target_activation)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("margins must be a nonempty matrix")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("margins must be finite and nonnegative")
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target activation must lie in [0, 1]")
    requested = target * values.shape[0]
    count = int(round(requested))
    if not math.isclose(requested, count, abs_tol=1e-12):
        raise ValueError("target activation must represent an integer number of clean samples")
    if count == 0:
        return 0.0
    minima = np.sort(np.min(values, axis=1))
    delta = float(minima[count - 1])
    if int(np.count_nonzero(minima <= delta)) != count:
        raise ValueError("margin ties prevent exact empirical activation calibration")
    return delta


def crossfit_deltas(margins: np.ndarray, target_activation: float, folds: int) -> np.ndarray:
    """Per-image delta calibrated on the other folds of the same margin population.

    Image ``i`` belongs to fold ``i % folds``; its delta is calibrated on the images of
    the remaining folds, so no image sets its own boundary threshold. A clean image and
    the watermarked image of the same prompt share an index and hence a delta.
    """
    if not isinstance(folds, int) or folds < 2 or folds > len(margins):
        raise ValueError("calibration folds must lie between two and the sample count")
    if float(target_activation) == 1.0:
        return np.full(len(margins), np.inf, dtype=np.float64)
    output = np.empty(len(margins), dtype=np.float64)
    indices = np.arange(len(margins))
    for fold in range(folds):
        held_out = indices % folds == fold
        output[held_out] = calibrate_delta(margins[~held_out], target_activation=target_activation)
    return output


def clean_threshold(scores: np.ndarray, requested_fpr: float) -> "tuple[float, float]":
    """Freeze the score threshold tau using only clean scores.

    Returns ``(tau, empirical_fpr)``: the threshold admitting the largest empirical FPR
    not above ``requested_fpr`` (ties pick the stricter threshold). ``(inf, 0.0)`` when no
    threshold qualifies. A score is detected when ``score >= tau``.
    """
    values = np.asarray(scores, dtype=np.float64)
    unique = sorted(set(float(x) for x in values), reverse=True)
    choices = []
    for threshold in unique:
        fpr = float(np.mean(values >= threshold))
        if fpr <= requested_fpr:
            choices.append((fpr, threshold))
    if not choices:
        return math.inf, 0.0
    best = max(choices, key=lambda item: (item[0], item[1]))
    return best[1], best[0]
