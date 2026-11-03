from __future__ import annotations

import numpy as np
import pytest

from watermark import calibration, coding, detection, prf

KEY = bytes(range(32))


def _margins(rows: int, m: int, seed: int = 0) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).normal(size=(rows, m))) * 0.1


@pytest.mark.parametrize("target", [0.1, 0.5, 0.75])
def test_delta_hits_the_requested_activation_exactly(target):
    margins = _margins(200, 8)
    delta = calibration.calibrate_delta(margins, target_activation=target)
    assert np.mean(margins.min(axis=1) <= delta) == pytest.approx(target)


def test_delta_edge_cases_and_validation():
    margins = _margins(10, 4)
    assert calibration.calibrate_delta(margins, target_activation=0.0) == 0.0
    with pytest.raises(ValueError):  # 0.25 * 10 is not an integer count
        calibration.calibrate_delta(margins, target_activation=0.25)
    with pytest.raises(ValueError):
        calibration.calibrate_delta(-margins, target_activation=0.5)
    tied = np.full((10, 2), 0.1)
    with pytest.raises(ValueError):  # ties make half the images impossible to separate
        calibration.calibrate_delta(tied, target_activation=0.5)
    with pytest.raises(ValueError):
        calibration.calibrate_delta(margins, target_activation=1.5)


def test_crossfit_never_calibrates_an_image_on_its_own_fold():
    margins = _margins(100, 6)
    deltas = calibration.crossfit_deltas(margins, 0.5, 5)
    assert deltas.shape == (100,)
    for fold in range(5):
        held_out = np.arange(100) % 5 == fold
        assert len(set(deltas[held_out])) == 1
        assert deltas[held_out][0] == calibration.calibrate_delta(margins[~held_out], target_activation=0.5)
    assert np.isinf(calibration.crossfit_deltas(margins, 1.0, 5)).all()
    with pytest.raises(ValueError):
        calibration.crossfit_deltas(margins, 0.5, 1)
    with pytest.raises(ValueError):
        calibration.crossfit_deltas(margins[:3], 0.5, 5)


def test_clean_threshold_admits_the_largest_fpr_not_above_the_target():
    scores = np.arange(1, 101) / 100.0
    tau, fpr = calibration.clean_threshold(scores, 0.20)
    assert fpr == pytest.approx(0.20) and tau == pytest.approx(0.81)
    assert np.mean(scores >= tau) <= 0.20
    tau, fpr = calibration.clean_threshold(np.array([0.5, 0.5, 0.5, 0.5]), 0.2)
    assert np.isinf(tau) and fpr == 0.0  # the tie cannot be split below 20 percent


def test_boundary_neighbourhood_is_restricted_to_uncertain_bits():
    table = prf.score_table(KEY, 4)
    margins = np.array([0.5, 0.01, 0.5, 0.02])
    score0, size0, unc0 = detection.boundary_max_score(0b1010, margins, table, delta=0.05, radius=0)
    assert (size0, unc0) == (1, 2) and score0 == table[0b1010]
    score1, size1, _ = detection.boundary_max_score(0b1010, margins, table, delta=0.05, radius=1)
    assert size1 == 3
    flips = {0b1010, 0b1010 ^ 0b0100, 0b1010 ^ 0b0001}  # bits 1 and 3 counted from the left
    assert score1 == max(table[i] for i in flips)
    _, size2, _ = detection.boundary_max_score(0b1010, margins, table, delta=0.05, radius=2)
    assert size2 == 4
    _, size_none, _ = detection.boundary_max_score(0b1010, margins, table, delta=0.005, radius=2)
    assert size_none == 1  # no margin within delta: only the observed code


def test_bit_order_of_flips_matches_code_ids():
    table = np.arange(16, dtype=float)
    # Uncertain bit 0 is the most significant bit.
    score, size, _ = detection.boundary_max_score(0b0000, np.array([0.0, 1, 1, 1]), table, delta=0.1, radius=1)
    assert (score, size) == (8.0, 2)


def _synthetic_pool(prompts: int, n: int, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(prompts, 1, dim))
    candidates = base + 0.6 * rng.normal(size=(prompts, n, dim))
    cleans = base[:, 0] + 0.6 * rng.normal(size=(prompts, dim))
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    return unit(candidates), unit(cleans), tuple("p%03d" % i for i in range(prompts))


def test_replicate_scores_are_deterministic_and_well_formed():
    candidates, cleans, uids = _synthetic_pool(100, 16, 32, 0)
    planes = coding.sample_hyperplanes(11, 32)
    kwargs = dict(frame_id="frame-00", key_trial=0, n=16, m=8, activation=0.5, radius=1, folds=5)
    first = detection.replicate_scores(candidates, cleans, uids, planes, KEY, **kwargs)
    second = detection.replicate_scores(candidates, cleans, uids, planes, KEY, **kwargs)
    assert np.array_equal(first.watermarked.scores, second.watermarked.scores)
    assert first.clean.scores.shape == (100,) and first.selected_index.shape == (100,)
    assert ((first.clean.scores > 0) & (first.clean.scores < 1)).all()
    assert (first.selected_index >= 0).all() and (first.selected_index < 16).all()
    # Delta is calibrated on the watermarked (selected) margins: activation hits the target.
    activated = first.watermarked.neighbourhood_sizes > 1
    assert 0.4 <= activated.mean() <= 0.6
    other = detection.replicate_scores(candidates, cleans, uids, planes, KEY, **{**kwargs, "slot": 2})
    assert not np.array_equal(first.selected_index, other.selected_index)


def test_selection_lifts_watermarked_scores_above_clean_scores():
    candidates, cleans, uids = _synthetic_pool(300, 16, 32, 1)
    planes = coding.sample_hyperplanes(12, 32)
    result = detection.replicate_scores(
        candidates, cleans, uids, planes, KEY, frame_id="frame-00", key_trial=0, n=16, m=8,
        activation=0.1, radius=1, folds=5)
    assert result.watermarked.scores.mean() > result.clean.scores.mean() + 0.05


@pytest.mark.parametrize("rule", ["gumbel", "tournament"])
def test_alternative_rules_run_through_the_same_pipeline(rule):
    candidates, cleans, uids = _synthetic_pool(50, 8, 16, 2)
    planes = coding.sample_hyperplanes(13, 16)
    result = detection.replicate_scores(
        candidates, cleans, uids, planes, KEY, frame_id="frame-01", key_trial=1, n=8, m=4,
        activation=0.1, radius=1, folds=5, rule=rule)
    assert result.watermarked.scores.shape == (50,)
    assert result.table.shape == (16,)


def test_replicate_scores_validate_shapes():
    candidates, cleans, uids = _synthetic_pool(10, 4, 8, 3)
    planes = coding.sample_hyperplanes(1, 8)
    with pytest.raises(ValueError):
        detection.replicate_scores(candidates, cleans, uids, planes, KEY, frame_id="f", key_trial=0, n=8, m=4,
                                   activation=0.5, radius=1, folds=5)
    with pytest.raises(ValueError):
        detection.replicate_scores(candidates, cleans, uids[:-1] + (uids[0],), planes, KEY, frame_id="f",
                                   key_trial=0, n=4, m=4, activation=0.5, radius=1, folds=5)


def test_set_score_is_the_mean_of_image_scores():
    a, b, c = np.array([0.1, 0.9]), np.array([0.3, 0.5]), np.array([0.2, 0.7])
    assert np.allclose(detection.aggregate_scores([a, b, c]), [0.2, 0.7])
    assert np.array_equal(detection.aggregate_scores([a]), a)


def test_scoring_edited_embeddings_with_frozen_deltas():
    candidates, cleans, uids = _synthetic_pool(40, 8, 16, 4)
    planes = coding.sample_hyperplanes(14, 16)
    result = detection.replicate_scores(
        candidates, cleans, uids, planes, KEY, frame_id="frame-00", key_trial=0, n=8, m=4,
        activation=0.5, radius=1, folds=5)
    unchanged = detection.score_embeddings(cleans, planes, 4, result.table, result.deltas, radius=1)
    assert np.array_equal(unchanged.scores, result.clean.scores)
