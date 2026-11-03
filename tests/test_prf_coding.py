from __future__ import annotations

import numpy as np
import pytest

from watermark import coding, prf

KEY = bytes(range(32))


def test_scores_are_deterministic_keyed_and_in_the_open_unit_interval():
    code = (1, 0, 1, 1)
    score = prf.score_code(KEY, code)
    assert 0.0 < score < 1.0
    assert score == prf.score_code(KEY, code)
    assert score != prf.score_code(bytes(32), code)
    assert score != prf.score_code(KEY, (1, 0, 1, 0))
    # The code length is part of the message: a prefix scores differently.
    assert prf.score_code(KEY, (1, 0, 1)) != prf.score_code(KEY, (1, 0, 1, 0))


def test_score_validation_rejects_bad_keys_and_codes():
    with pytest.raises(ValueError):
        prf.score_code(b"short", (0, 1))
    with pytest.raises(ValueError):
        prf.score_code(KEY, (0, 2))
    with pytest.raises(ValueError):
        prf.score_code(KEY, ())
    with pytest.raises(ValueError):
        prf.score_code(KEY, (0,) * 65)


def test_pack_code_uses_network_order_with_zero_padding():
    assert prf.pack_code((1, 0, 0, 0, 0, 0, 0, 1, 1)) == bytes([0b10000001, 0b10000000])


def test_score_table_is_indexed_by_packed_code_id():
    table = prf.score_table(KEY, 3)
    assert table.shape == (8,)
    for code_id in range(8):
        assert table[code_id] == prf.score_code(KEY, prf.code_bits(code_id, 3))
    assert prf.code_bits(5, 3) == (1, 0, 1)


def test_scores_look_uniform_over_many_codes():
    table = prf.score_table(KEY, 12)
    assert abs(table.mean() - 0.5) < 0.02
    assert 0.27 < table.std() < 0.30  # uniform on (0, 1) has standard deviation 0.2887
    assert table.min() > 0 and table.max() < 1


def test_trial_zero_is_the_master_key_and_later_trials_are_distinct():
    keys = [prf.derive_trial_key(KEY, trial) for trial in range(5)]
    assert keys[0] == KEY and len(set(keys)) == 5 and all(len(key) == 32 for key in keys)
    assert prf.derive_trial_key(KEY, 3) == keys[3]
    with pytest.raises(ValueError):
        prf.derive_trial_key(KEY, -1)
    with pytest.raises(ValueError):
        prf.derive_trial_key(b"x", 1)


def test_derived_seeds_are_domain_separated_and_64_bit():
    a = prf.derive_seed(KEY, "d1", "p")
    assert a == prf.derive_seed(KEY, "d1", "p")
    assert a != prf.derive_seed(KEY, "d2", "p") and a != prf.derive_seed(KEY, "d1", "q")
    assert 0 <= a < 2 ** 64
    assert len(prf.generate_master_key()) == 32 and prf.generate_master_key() != prf.generate_master_key()


def test_hyperplanes_are_reproducible_unit_rows_with_64_planes():
    planes = coding.sample_hyperplanes(1234, 32)
    assert planes.shape == (64, 32) and planes.dtype == np.float64
    assert np.allclose(np.linalg.norm(planes, axis=1), 1.0)
    assert np.array_equal(planes, coding.sample_hyperplanes(1234, 32))
    assert not np.array_equal(planes, coding.sample_hyperplanes(1235, 32))
    with pytest.raises(ValueError):
        coding.sample_hyperplanes(-1, 4)


def test_code_embedding_is_scale_invariant_and_matches_projection_signs():
    planes = coding.sample_hyperplanes(7, 16)
    vector = np.random.default_rng(0).normal(size=16)
    code = coding.code_embedding(vector, planes)
    assert len(code) == 64 and set(code) <= {0, 1}
    assert code == coding.code_embedding(5.0 * vector, planes)
    projection = coding.project(vector / np.linalg.norm(vector), planes, 8)
    assert tuple(int(x >= 0) for x in projection) == code[:8]
    with pytest.raises(ValueError):
        coding.code_embedding(np.zeros(16), planes)
    with pytest.raises(ValueError):
        coding.code_embedding(vector, planes[:, :8])


def test_bits_to_code_ids_packs_most_significant_bit_first():
    bits = np.array([[True, False, True], [False, False, True]])
    assert coding.bits_to_code_ids(bits).tolist() == [5, 1]
    with pytest.raises(ValueError):
        coding.bits_to_code_ids(np.array([1, 0, 1]))
