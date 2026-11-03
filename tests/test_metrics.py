from __future__ import annotations

import numpy as np
import pytest

from watermark import metrics


def test_auroc_counts_ties_as_half_and_matches_known_values():
    assert metrics.auroc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert metrics.auroc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert metrics.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert metrics.auroc([0.1, 0.5], [0.5, 0.9]) == pytest.approx((1 + 0.5 + 1 + 1) / 4)


def test_auroc_equals_the_trapezoid_area_under_the_roc_curve():
    rng = np.random.default_rng(0)
    clean, wm = rng.normal(size=60), rng.normal(0.7, 1, size=60)
    roc = metrics.roc_curve(clean, wm)
    fpr = [row["fpr"] for row in roc]
    tpr = [row["tpr"] for row in roc]
    area = sum((fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2 for i in range(len(roc) - 1))
    assert metrics.auroc(clean, wm) == pytest.approx(area)


def test_roc_curve_spans_zero_to_one_and_is_monotone():
    roc = metrics.roc_curve([0.1, 0.4, 0.6], [0.3, 0.7, 0.9])
    assert (roc[0]["fpr"], roc[0]["tpr"]) == (0.0, 0.0) and (roc[-1]["fpr"], roc[-1]["tpr"]) == (1.0, 1.0)
    assert all(a["fpr"] <= b["fpr"] and a["tpr"] <= b["tpr"] for a, b in zip(roc, roc[1:]))


def test_tpr_at_fpr_reads_the_best_point_within_the_fpr_budget():
    clean = np.linspace(0, 1, 100, endpoint=False)
    wm = np.linspace(0.3, 1.3, 100, endpoint=False)
    roc = metrics.roc_curve(clean, wm)
    point = metrics.tpr_at_fpr(roc, 0.2)
    assert point["empirical_fpr"] <= 0.2 and point["requested_fpr"] == 0.2
    assert point["empirical_fpr"] == pytest.approx(0.2)
    bigger = metrics.tpr_at_fpr(roc, 0.5)
    assert bigger["tpr"] >= point["tpr"]
    assert metrics.tpr_at_fpr(roc, 0.0)["empirical_fpr"] == 0.0
    with pytest.raises(ValueError):
        metrics.tpr_at_fpr(roc, 1.5)


def test_clean_detection_roc_bundles_auc_and_operating_points():
    result = metrics.clean_detection_roc([0.1, 0.2, 0.3, 0.4], [0.35, 0.5, 0.6, 0.7], fixed_fprs=(0.25, 0.5))
    assert result["sample_count"] == 4 and result["auc"] == pytest.approx(15 / 16)
    assert [row["requested_fpr"] for row in result["fixed_fpr"]] == [0.25, 0.5]
    with pytest.raises(ValueError):
        metrics.clean_detection_roc([], [0.1])
    with pytest.raises(ValueError):
        metrics.clean_detection_roc([float("nan")], [0.1])


def test_mean_roc_interpolates_on_a_common_grid():
    grid = np.linspace(0, 1, 11)
    perfect = metrics.roc_curve([0.1, 0.2], [0.8, 0.9])
    chance = metrics.roc_curve([0.1, 0.9], [0.1, 0.9])
    averaged = metrics.mean_roc([perfect, chance], grid)
    assert averaged["mean"].shape == (11,) and averaged["sd"].shape == (11,)
    assert averaged["mean"][0] >= 0.5 and averaged["mean"][-1] == 1.0
    assert np.allclose(metrics.interpolate_roc(perfect, grid)[1:], 1.0)
    assert np.allclose(metrics.mean_roc([perfect], grid)["sd"], 0.0)


def test_paired_retention_counts_only_initially_detected_images():
    original = [0.9, 0.8, 0.2, 0.7]
    edited = [0.9, 0.4, 0.95, 0.6]
    result = metrics.paired_retention(original, edited, 0.5)
    assert result["identity_detected_count"] == 3 and result["attacked_retained_count"] == 2
    assert result["paired_detection_retention"] == pytest.approx(2 / 3)
    nothing = metrics.paired_retention([0.1, 0.2], [0.9, 0.9], 0.5)
    assert np.isnan(nothing["paired_detection_retention"])
    with pytest.raises(ValueError):
        metrics.paired_retention([0.1], [0.1, 0.2], 0.5)
