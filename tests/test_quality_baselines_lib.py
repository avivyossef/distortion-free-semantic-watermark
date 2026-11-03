"""Native-baseline helpers: seeds, fixed state, panel layout, identifiers, aggregation (CPU, synthetic)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "baselines"))

import _baselines_lib as lib  # noqa: E402
from watermark import generation  # noqa: E402
from watermark.attacks import EDIT_GROUPS, EDIT_IDS  # noqa: E402

KEY = bytes(range(32))


def test_slot_one_seed_is_the_corpus_clean_seed():
    assert lib.generation_seed("p1", 1) == generation._native_clean_seed("p1", "key-00")


def test_later_slots_have_distinct_bounded_seeds():
    seeds = [lib.generation_seed("p1", slot) for slot in range(1, 12)]
    assert len(set(seeds)) == len(seeds)
    assert all(seed < 2 ** 63 for seed in seeds[1:])
    assert lib.generation_seed("p1", 3) != lib.generation_seed("p2", 3)
    with pytest.raises(ValueError):
        lib.generation_seed("p1", 0)


def test_fixed_state_is_deterministic_and_key_dependent():
    other = bytes(reversed(range(32)))
    assert lib.tree_ring_seed(KEY) == lib.tree_ring_seed(KEY) < 2 ** 32
    assert lib.tree_ring_seed(KEY) != lib.tree_ring_seed(other)
    key, nonce = lib.gaussian_key_nonce(KEY)
    assert len(key) == 32 and len(nonce) == 12 and (key, nonce) != lib.gaussian_key_nonce(other)
    bits = lib.gaussian_watermark_bits(KEY, 4, 8)
    assert bits.shape == (1, 4, 8, 8) and set(np.unique(bits)) <= {0, 1}
    assert np.array_equal(bits, lib.gaussian_watermark_bits(KEY, 4, 8))
    assert not np.array_equal(bits, lib.gaussian_watermark_bits(other, 4, 8))


def test_image_ids_round_trip():
    image_id = lib.make_image_id("p1", "clean", "jpeg_q90")
    assert lib.parse_image_id(image_id) == ("p1", "clean", "jpeg_q90")
    assert lib.parse_image_id(lib.make_image_id("p1", "watermarked")) == ("p1", "watermarked", "identity")
    with pytest.raises(ValueError):
        lib.make_image_id("p|1", "clean")
    with pytest.raises(ValueError):
        lib.make_image_id("p1", "other")


def test_shards_partition_prompts():
    owners = [[i for i in range(10) if lib.in_shard(i, shard, 3)] for shard in range(3)]
    assert sorted(sum(owners, [])) == list(range(10))
    with pytest.raises(ValueError):
        lib.in_shard(0, 3, 3)


def _write_panel(root: Path, uid: str, slots: int = 1, value: int = 100) -> None:
    from PIL import Image

    def write(staging: Path):
        records = []
        for slot in range(1, slots + 1):
            directory = staging / lib.slot_dir_name(slot)
            directory.mkdir()
            entry = {"slot": slot}
            for role in lib.ROLES:
                Image.new("RGB", (16, 16), (value, slot, 0)).save(directory / (role + ".png"))
                entry[role] = lib.image_record(directory / (role + ".png"), staging)
            records.append(entry)
        return records

    lib.write_prompt_panel(root, {"prompt_uid": uid, "method": "tree_ring", "prompt": "a " + uid}, write)


def test_prompt_panel_layout_and_records(tmp_path):
    _write_panel(tmp_path, "p1", slots=2)
    panel = lib.read_panel(tmp_path, "p1")
    assert [item["slot"] for item in panel["slots"]] == [1, 2]
    assert panel["slots"][1]["clean"]["relative_path"] == "slot_002/clean.png"
    assert lib.panel_image_path(tmp_path, panel, 2, "watermarked").is_file()
    from watermark.io import sha256_file

    path = lib.panel_image_path(tmp_path, panel, 1, "clean")
    assert sha256_file(path) == panel["slots"][0]["clean"]["file_sha256"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p1"]  # no staging leftovers
    with pytest.raises(KeyError):
        lib.panel_image_path(tmp_path, panel, 3, "clean")


def test_failed_prompt_leaves_nothing(tmp_path):
    def boom(staging):
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError):
        lib.write_prompt_panel(tmp_path, {"prompt_uid": "p1"}, boom)
    assert list(tmp_path.iterdir()) == []


def _table(uids, wm, clean, edits=None):
    """Score table with the given unedited scores and per-edit (watermarked, clean) overrides."""
    table = {(lib.IDENTITY, "watermarked"): dict(zip(uids, wm)), (lib.IDENTITY, "clean"): dict(zip(uids, clean))}
    for edit_id in EDIT_IDS:
        marked, other = (edits or {}).get(edit_id, (wm, clean))
        table[(edit_id, "watermarked")] = dict(zip(uids, marked))
        table[(edit_id, "clean")] = dict(zip(uids, other))
    return table


def test_robustness_hand_computed():
    uids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    clean = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    wm = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # at FPR 0.1 the threshold is the top clean score 0.9: four detections, FPR 0.1
    edits = {"crop_50": ([1.0, 2.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0], [0.95] + [0.0] * 9)}
    per_edit, categories = lib.analyze_robustness(_table(uids, wm, clean, edits), [0.1])
    assert len(per_edit) == 20 and len(categories) == 2
    row = next(r for r in per_edit if r["edit_id"] == "crop_50")
    assert row["threshold"] == 0.9 and row["calibration_fpr"] == 0.1
    assert row["unedited_tpr"] == 0.4
    assert row["edited_tpr"] == pytest.approx(0.3)  # 1.0, 2.0 and the 0.95
    assert row["edited_fpr"] == pytest.approx(0.1)
    assert row["retention"] == pytest.approx(0.5)  # a, b kept of a, b, c, d
    other = next(r for r in per_edit if r["edit_id"] == "crop_75")
    assert other["retention"] == 1.0 and other["edited_fpr"] == 0.1 and other["category"] == "geometric"
    geometric = next(c for c in categories if c["category"] == "geometric")
    assert geometric["settings"] == len(EDIT_GROUPS["geometric"])
    assert geometric["edited_tpr"] == pytest.approx((0.3 + 0.4 * 5) / 6)
    assert geometric["retention"] == pytest.approx((0.5 + 5) / 6)


def test_robustness_requires_every_edit_and_matching_prompts():
    uids = ["a", "b"]
    table = _table(uids, [1.0, 2.0], [0.0, 0.1])
    del table[("jpeg_q75", "clean")]
    with pytest.raises(ValueError):
        lib.analyze_robustness(table, [0.5])
    table = _table(uids, [1.0, 2.0], [0.0, 0.1])
    table[("crop_50", "watermarked")] = {"a": 1.0}
    with pytest.raises(ValueError):
        lib.analyze_robustness(table, [0.5])


def test_collect_scores_rejects_duplicates_and_nonfinite():
    rows = [{"image_id": lib.make_image_id("a", "clean"), "score": 1.0}]
    assert lib.collect_scores(rows) == {(lib.IDENTITY, "clean"): {"a": 1.0}}
    with pytest.raises(ValueError):
        lib.collect_scores(rows + rows)
    with pytest.raises(ValueError):
        lib.collect_scores([{"image_id": lib.make_image_id("a", "clean"), "score": float("nan")}])


def test_clean_detection_separable_scores():
    result = lib.clean_detection([0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3], (0.01, 0.05, 0.20))
    assert result["auc"] == 1.0
    assert [point["tpr"] for point in result["fixed_fpr"]] == [1.0, 1.0, 1.0]
