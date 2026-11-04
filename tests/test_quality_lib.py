"""CPU tests of the shared helpers of the quality experiments (synthetic data)."""

import sys
from pathlib import Path

import numpy as np
import pytest

QUALITY = Path(__file__).resolve().parents[1] / "experiments" / "quality"
sys.path.insert(0, str(QUALITY))

import _quality_lib as lib  # noqa: E402
from watermark.coding import bits_to_code_ids, sample_hyperplanes  # noqa: E402
from watermark.prf import derive_trial_key, score_table  # noqa: E402
from watermark.selection import cell_rng, select_code_ids  # noqa: E402

KEY = bytes(range(32))


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _pools(prompts=20, slots=3, n=16, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(prompts, 1, 1, dim))
    candidates = _unit(base + 0.6 * rng.normal(size=(prompts, slots, n, dim)))
    cleans = _unit(base[:, :, 0] + 0.6 * rng.normal(size=(prompts, slots, dim)))
    return candidates, cleans, ["p%02d" % i for i in range(prompts)]


def test_selection_is_gumbel_selection_on_the_quality_stream():
    candidates, _, uids = _pools()
    planes = sample_hyperplanes(7, 12)
    positions = lib.select_positions(candidates, planes, KEY, frame_id="frame-00", key_trial=2, n=16, m=8,
                                     prompt_uids=uids)
    assert positions.shape == (20, 3) and positions.min() >= 1 and positions.max() <= 16
    table = score_table(KEY, 8)
    for slot in range(3):
        identity = uids[:] if slot == 0 else ["%s:slot:%d" % (uid, slot + 1) for uid in uids]  # slot 1 is bare
        projection = np.einsum("pnd,md->pnm", candidates[:, slot], planes[:8])
        ids = bits_to_code_ids(projection >= 0.0)
        expected = [select_code_ids(ids[p], table, cell_rng("gumbel-paper-rerun/fid10k/v1", "frame-00", 2, "hmac",
                                                           16, 8, identity[p]))[0] + 1 for p in range(len(uids))]
        assert positions[:, slot].tolist() == expected
    assert lib.pool_identity("p", 1) == "p" and lib.pool_identity("p", 2) == "p:slot:2"


def test_selection_uses_only_the_first_n_candidates():
    candidates, _, uids = _pools()
    planes = sample_hyperplanes(7, 12)
    positions = lib.select_positions(candidates, planes, KEY, frame_id="frame-01", key_trial=0, n=8, m=4,
                                     prompt_uids=uids)
    assert positions.max() <= 8
    truncated = lib.select_positions(candidates[:, :, :8], planes, KEY, frame_id="frame-01", key_trial=0, n=8,
                                     m=4, prompt_uids=uids)
    assert np.array_equal(positions, truncated)


def test_matched_random_is_uniform_deterministic_and_slot_one_is_stable():
    uids = ["p%03d" % i for i in range(400)]
    three = lib.matched_random_positions(uids, 3, 8, 3)
    again = lib.matched_random_positions(uids, 3, 8, 3)
    one = lib.matched_random_positions(uids, 3, 8, 1)
    assert np.array_equal(three, again) and np.array_equal(three[:, :1], one)
    assert three.min() == 1 and three.max() == 8
    counts = np.bincount(three[:, 0], minlength=9)[1:]
    assert counts.min() > 25 and counts.max() < 80
    assert not np.array_equal(three, lib.matched_random_positions(uids, 4, 8, 3))
    assert not np.array_equal(three[:, 0], three[:, 1])
    with pytest.raises(ValueError):
        lib.matched_random_positions(["a", "a"], 0, 8, 1)


def test_uniform_positions_depend_on_frame_n_prompt_and_slot():
    base = lib.uniform_positions("p1", "frame-00", 8, 60)
    assert base.shape == (60,) and base.min() >= 1 and base.max() <= 8
    assert np.array_equal(base, lib.uniform_positions("p1", "frame-00", 8, 60))
    assert not np.array_equal(base, lib.uniform_positions("p1", "frame-01", 8, 60))
    assert not np.array_equal(base, lib.uniform_positions("p2", "frame-00", 8, 60))


def test_trial_keys_follow_derive_trial_key():
    keys = lib.trial_keys(KEY, 4)
    assert keys[0] == KEY and keys[3] == derive_trial_key(KEY, 3) and len(set(keys.values())) == 4
    with pytest.raises(ValueError):
        lib.trial_keys(KEY, 0)


def test_gather_picks_the_selected_candidates():
    values = np.arange(2 * 3 * 4 * 2).reshape(2, 3, 4, 2)
    positions = np.array([[1, 2, 3], [4, 4, 1]])
    picked = lib.gather(values, positions)
    assert picked.shape == (2, 3, 2) and np.array_equal(picked[1, 0], values[1, 0, 3])
    with pytest.raises(ValueError):
        lib.gather(values, np.zeros((2, 3), dtype=int))


def test_frechet_distance_of_gaussians():
    rng = np.random.default_rng(0)
    mu1, mu2 = rng.normal(size=5), rng.normal(size=5)
    v1, v2 = rng.uniform(0.5, 2, size=5), rng.uniform(0.5, 2, size=5)
    expected = float(np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(v1) - np.sqrt(v2)) ** 2))
    assert lib.frechet_distance(mu1, np.diag(v1), mu2, np.diag(v2)) == pytest.approx(expected, rel=1e-6)
    assert lib.frechet_distance(mu1, np.diag(v1), mu1, np.diag(v1)) == pytest.approx(0.0, abs=1e-6)


def test_population_statistics_accepts_pooled_arrays():
    values = np.random.default_rng(1).normal(size=(6, 4, 3))
    mean, covariance = lib.population_statistics(values)
    assert mean.shape == (3,) and covariance.shape == (3, 3)
    assert np.allclose(mean, values.reshape(-1, 3).mean(axis=0))
    with pytest.raises(ValueError):
        lib.activation_statistics(values[:1, 0])


def test_mean_interval_uses_the_t_distribution():
    result = lib.mean_interval([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == 2.5 and result["count"] == 4
    assert result["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert result["ci95_high"] - result["mean"] == pytest.approx(3.182446 * result["se"], rel=1e-5)
    single = lib.mean_interval([5.0])
    assert single["ci95_low"] == single["ci95_high"] == 5.0


def test_cosine_vendi_counts_distinct_directions():
    assert lib.cosine_vendi_score(np.eye(5)) == pytest.approx(5.0)
    assert lib.cosine_vendi_score(np.ones((4, 3))) == pytest.approx(1.0)
    assert lib.cosine_vendi_score(np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])) < 3
    with pytest.raises(ValueError):
        lib.cosine_vendi_score(np.zeros((2, 3)))


def test_shards_round_trip_and_reject_gaps(tmp_path):
    rng = np.random.default_rng(0)
    full = rng.normal(size=(4, 2, 3)).astype(np.float32)
    for index, rows in enumerate(([0, 2], [1, 3])):
        lib.write_shard(tmp_path / ("activations-shard-%03d-of-002.npz" % index), ["u%d" % r for r in rows],
                        cleans=full[rows])
    merged = lib.load_shards(tmp_path, ["cleans"], ["u3", "u0", "u1", "u2"])
    assert np.allclose(merged["cleans"], full[[3, 0, 1, 2]])
    with pytest.raises(ValueError):
        lib.load_shards(tmp_path, ["cleans"], ["u0", "missing"])
    with pytest.raises(ValueError):
        lib.write_shard(tmp_path / "activations-shard-000-of-002.npz", ["u0"], cleans=full[:1])
    (tmp_path / "activations-shard-001-of-002.npz").unlink()
    with pytest.raises(ValueError):
        lib.load_shards(tmp_path, ["cleans"], ["u0"])


def test_selection_file_round_trip(tmp_path):
    arrays = {"n8_m4_random": np.array([[[1, 2]], [[3, 4]]]), "n8_m4_watermarked": np.ones((1, 2, 1, 2), dtype=int)}
    lib.write_selections(tmp_path / "s.npz", ["a"], arrays, {"num_keys": 2})
    uids, loaded, metadata = lib.load_selections(tmp_path / "s.npz")
    assert uids == ["a"] and metadata == {"num_keys": 2}
    assert np.array_equal(loaded["n8_m4_random"], arrays["n8_m4_random"])


def test_parse_named_paths():
    assert lib.parse_named_paths(["a=/x/y", "b=z"]) == {"a": Path("/x/y"), "b": Path("z")}
    with pytest.raises(ValueError):
        lib.parse_named_paths(["a=/x", "a=/y"])
    with pytest.raises(ValueError):
        lib.parse_named_paths(["novalue"])


def _captions():
    images = [{"id": 2, "file_name": "000000000002.jpg"}, {"id": 1, "file_name": "000000000001.jpg"}]
    annotations = [
        {"id": 10, "image_id": 1, "caption": "A  dog—runs. "},
        {"id": 11, "image_id": 1, "caption": "A cat sleeps"},
        {"id": 12, "image_id": 2, "caption": "A cat sleeps"},
        {"id": 13, "image_id": 2, "caption": "a dog—runs."},
    ]
    return {"images": images, "annotations": annotations}


def test_fid5k_panel_assigns_unique_hash_ranked_captions():
    rows = lib.build_fid5k_panel(_captions())
    assert [row["image_id"] for row in rows] == [1, 2] and [row["panel_index"] for row in rows] == [0, 1]
    assert rows[0]["file_name"] == "000000000001.jpg"
    texts = [row["text"] for row in rows]
    assert len({text.casefold() for text in texts}) == 2 and all("  " not in text for text in texts)
    assert len({row["prompt_uid"] for row in rows}) == 2
    assert rows == lib.build_fid5k_panel(_captions())


def test_fid5k_panel_rejects_missing_or_duplicate_records():
    bad = _captions()
    bad["annotations"].append({"id": 10, "image_id": 1, "caption": "again"})
    with pytest.raises(ValueError):
        lib.build_fid5k_panel(bad)
    orphan = _captions()
    orphan["annotations"].append({"id": 99, "image_id": 7, "caption": "no image"})
    with pytest.raises(ValueError):
        lib.build_fid5k_panel(orphan)
    single = {"images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
              "annotations": [{"id": 1, "image_id": 1, "caption": "x"}, {"id": 2, "image_id": 2, "caption": "X"}]}
    with pytest.raises(ValueError):
        lib.build_fid5k_panel(single)


def test_normalize_prompt():
    assert lib.normalize_prompt("  A dog–x \n\u0000y ") == "A dog-x y"
    with pytest.raises(TypeError):
        lib.normalize_prompt(3)


def test_native_images_follow_the_baselines_layout(tmp_path):
    import json

    root = tmp_path / "p1"
    root.mkdir()
    slots = [{"slot": s, "watermarked": {"relative_path": "slot_%03d/watermarked.png" % s},
              "clean": {"relative_path": "slot_%03d/clean.png" % s}} for s in (2, 1)]
    (root / "panel.json").write_text(json.dumps({"prompt_uid": "p1", "slots": slots}))
    marked, clean = lib.native_images(tmp_path, "p1")
    assert marked == [root / "slot_001/watermarked.png", root / "slot_002/watermarked.png"]
    assert clean[1] == root / "slot_002/clean.png"
    assert len(lib.native_images(tmp_path, "p1", 1)[0]) == 1
    with pytest.raises(ValueError):
        lib.native_images(tmp_path, "p1", 3)
