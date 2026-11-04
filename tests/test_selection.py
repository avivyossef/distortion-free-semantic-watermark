from __future__ import annotations

import numpy as np
import pytest

from watermark import prf, selection, selection_rules as rules

KEY = bytes(range(32))


def _key(index: int) -> bytes:
    return prf.derive_trial_key(KEY, index)


def test_all_distinct_codes_select_the_best_exponential_race_code():
    codes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    trace = selection.select_candidate(codes, KEY, np.random.default_rng(0))
    scores = [prf.score_code(KEY, code) for code in codes]
    assert trace.distinct_code_count == 4 and trace.tie_count == 1
    assert trace.selected_index == int(np.argmax(scores)) and trace.selected_score == max(scores)


def test_identical_codes_select_a_uniform_candidate():
    codes = [(1, 0)] * 4
    picks = [selection.select_candidate(codes, KEY, np.random.default_rng(seed)).selected_index
             for seed in range(400)]
    counts = np.bincount(picks, minlength=4)
    assert (counts > 60).all() and set(picks) == {0, 1, 2, 3}


def test_group_size_enters_the_race_as_log_score_over_n_c():
    # Code (0, 0) holds three candidates: its race value is log(0.3) / 3 = -0.40, which beats
    # log(0.5) = -0.69 of the singleton code, although the singleton has the larger score.
    ids = np.array([0, 0, 0, 1])
    scores = np.array([0.3, 0.5])
    for seed in range(20):
        index, distinct = selection.select_code_ids(ids, scores, np.random.default_rng(seed))
        assert index in (0, 1, 2) and distinct == 2
    # with two singletons the larger score wins
    index, _ = selection.select_code_ids(np.array([0, 1]), scores, np.random.default_rng(0))
    assert index == 1


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError):
        selection.select_candidate([], KEY, np.random.default_rng(0))


@pytest.mark.parametrize("seed", range(6))
def test_code_id_selection_matches_bit_tuple_selection(seed):
    rng = np.random.default_rng(seed)
    m = 3
    ids = rng.integers(0, 1 << m, size=8)
    table = prf.score_table(KEY, m)
    codes = [prf.code_bits(int(code_id), m) for code_id in ids]
    trace = selection.select_candidate(codes, KEY, np.random.default_rng(100 + seed))
    index, distinct = selection.select_code_ids(ids, table, np.random.default_rng(100 + seed))
    assert (index, distinct) == (trace.selected_index, trace.distinct_code_count)


def test_bit_tuple_selection_handles_long_codes():
    codes = [tuple(int(b) for b in np.random.default_rng(i % 3).integers(0, 2, size=64)) for i in range(6)]
    trace = selection.select_candidate(codes, KEY, np.random.default_rng(0))
    assert 0 <= trace.selected_index < 6 and trace.distinct_code_count == 3


def test_representative_sampling_branches_and_selects_a_candidate():
    ids = np.array([0, 0, 3, 1])
    table = prf.score_table(KEY, 2)
    branches = set()
    for seed in range(60):
        index, branch, distinct = rules.select_representative(ids, table, np.random.default_rng(seed))
        branches.add(branch)
        assert 0 <= index < 4 and distinct == 3
        if branch == "duplicate_uniform":
            assert index in (0, 1)
    assert branches == {"unique_max", "duplicate_uniform"}
    index, branch, _ = rules.select_representative(np.array([0, 1, 2, 3]), np.array([.2, .9, .5, .1]),
                                                   np.random.default_rng(0))
    assert (index, branch) == (1, "unique_max")


def _selection_law(pick, trials: int = 3000, n: int = 4):
    counts = np.zeros(n)
    for trial in range(trials):
        counts[pick(trial)] += 1
    return counts / trials


def test_gumbel_selection_is_distortion_free_over_keys():
    # Averaged over keys and sampler randomness every candidate is chosen with probability 1/N,
    # even when codes collide.
    ids = np.array([0, 0, 3, 1])
    law = _selection_law(lambda t: selection.select_code_ids(
        ids, prf.score_table(_key(t), 2), np.random.default_rng(t))[0])
    assert np.abs(law - 0.25).max() < 0.035


@pytest.mark.parametrize("rule", ["representative", "gumbel", "tournament"])
def test_every_rule_is_distortion_free_over_keys(rule):
    ids = np.array([0, 0, 3, 1])

    def pick(trial: int) -> int:
        key = _key(trial)
        scores = prf.score_table(key, 2)
        tables = rules.tournament_layer_tables(key, m=2, layers=2) if rule == "tournament" else None
        return rules.select(rule, ids, scores=scores, layer_tables=tables, rng=np.random.default_rng(trial))[0]

    assert np.abs(_selection_law(pick) - 0.25).max() < 0.035


def test_gumbel_rule_is_the_selection_module_function():
    assert rules.select_gumbel is selection.select_code_ids
    assert rules.RULES == ("representative", "gumbel", "tournament")


def test_gumbel_picks_the_best_exponential_race_code():
    ids = np.array([0, 1, 2, 3])
    scores = np.array([0.2, 0.9, 0.5, 0.1])
    index, distinct = rules.select_gumbel(ids, scores, np.random.default_rng(0))
    assert (index, distinct) == (1, 4)
    duplicated = np.array([0, 0, 0, 1])  # 0.2**(1/3) beats 0.3 when code 0 has three members
    index, distinct = rules.select_gumbel(duplicated, np.array([0.5, 0.3]), np.random.default_rng(0))
    assert index in (0, 1, 2) and distinct == 2


def test_tournament_layers_and_tables():
    assert [rules.tournament_layers(n) for n in (2, 8, 32)] == [1, 3, 5]
    with pytest.raises(ValueError):
        rules.tournament_layers(12)
    tables = rules.tournament_layer_tables(KEY, m=3, layers=2)
    assert tables.shape == (2, 8) and ((tables > 0) & (tables < 1)).all()
    assert not np.array_equal(tables[0], tables[1])
    assert np.array_equal(rules.detector_table("tournament", scores=np.zeros(8), layer_tables=tables),
                          tables.mean(axis=0))
    scores = prf.score_table(KEY, 3)
    assert np.array_equal(rules.detector_table("gumbel", scores=scores, layer_tables=None), scores)
    with pytest.raises(ValueError):  # two layers cannot decide a pool of eight
        rules.select_tournament(np.zeros(8, dtype=int), tables, np.random.default_rng(0))
    with pytest.raises(ValueError):  # the tournament needs its layer tables
        rules.select("tournament", np.zeros(4, dtype=int), scores=scores, layer_tables=None,
                     rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        rules.select("unknown", np.zeros(4, dtype=int), scores=scores, layer_tables=None,
                     rng=np.random.default_rng(0))


def test_cell_rng_depends_on_every_identity_part():
    a = selection.cell_rng("x", 1, "p").random()
    assert a == selection.cell_rng("x", 1, "p").random()
    assert a != selection.cell_rng("x", 2, "p").random() and a != selection.cell_rng("x", 1, "q").random()


def test_selector_streams_follow_the_papers_identities():
    from watermark.detection import DETECTION_STREAM, EDIT_STREAM, selector_rng

    def same(a, b):
        return a.random() == b.random()

    args = ("frame-02", 3, 16, 8)
    assert same(selector_rng("gumbel", *args, "p", slot=2),
                selection.cell_rng(DETECTION_STREAM, "gumbel", *args, "p:slot:2"))
    assert same(selector_rng("gumbel", *args, "p"), selection.cell_rng(EDIT_STREAM, "gumbel", *args, "p"))
    assert same(selector_rng("representative", *args, "p", slot=1),
                selection.cell_rng("candidate-pool-sweep", "frame-02", 3, "hmac", 16, 8, "p:slot:1"))
    assert not same(selector_rng("gumbel", *args, "p"), selector_rng("gumbel", *args, "p", slot=1))
    assert not same(selector_rng("gumbel", *args, "p"), selector_rng("tournament", *args, "p"))
