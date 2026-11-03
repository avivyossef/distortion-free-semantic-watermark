"""End-to-end CPU run of the aggregation scripts of the baselines experiment on synthetic images."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BASELINES = Path(__file__).resolve().parents[1] / "experiments" / "baselines"
sys.path.insert(0, str(BASELINES))

import _baselines_lib as lib  # noqa: E402
from test_quality_baselines_lib import _write_panel  # noqa: E402
from watermark.attacks import EDIT_IDS  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_jsonl, write_jsonl  # noqa: E402


def _run(name, argv, monkeypatch):
    spec = importlib.util.spec_from_file_location("baselines_script_" + name, str(BASELINES / (name + ".py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [name] + [str(item) for item in argv])
    return module.main()


def _panel(tmp_path, count=4):
    uids = ["p%d" % i for i in range(count)]
    write_jsonl(tmp_path / "prompts.jsonl", [{"prompt_uid": uid, "text": "a " + uid} for uid in uids])
    for uid in uids:
        _write_panel(tmp_path / "gen", uid, slots=1, value=60 + 10 * len(uid))
    return uids


def test_list_images_covers_both_roles(tmp_path, monkeypatch):
    uids = _panel(tmp_path)
    _run("list_images", ["--generation-dir", tmp_path / "gen", "--prompts", tmp_path / "prompts.jsonl",
                         "--output", tmp_path / "images.jsonl"], monkeypatch)
    rows = read_jsonl(tmp_path / "images.jsonl")
    assert [row["image_id"] for row in rows[:2]] == ["p0|watermarked|identity", "p0|clean|identity"]
    assert len(rows) == 2 * len(uids) and all(Path(row["path"]).is_file() for row in rows)
    assert rows[0]["prompt"] == "a p0"


def test_attack_images_renders_every_edit_for_both_roles(tmp_path, monkeypatch):
    uids = _panel(tmp_path, count=3)
    args = ["--generation-dir", tmp_path / "gen", "--prompts", tmp_path / "prompts.jsonl",
            "--output-dir", tmp_path / "edits"]
    _run("attack_images", args + ["--images-output", tmp_path / "e0.jsonl", "--num-shards", "2"], monkeypatch)
    _run("attack_images", args + ["--images-output", tmp_path / "e1.jsonl", "--num-shards", "2",
                                  "--shard-index", "1"], monkeypatch)
    rows = read_jsonl(tmp_path / "e0.jsonl") + read_jsonl(tmp_path / "e1.jsonl")
    assert len(rows) == len(uids) * 2 * len(EDIT_IDS)
    assert len({row["image_id"] for row in rows}) == len(rows)
    assert all(Path(row["path"]).is_file() for row in rows)


def test_generate_skips_finished_prompts_without_loading_a_model(tmp_path, monkeypatch):
    _panel(tmp_path, count=2)
    key = tmp_path / "key.bin"
    key.write_bytes(bytes(32))
    assert _run("generate", ["--method", "tree_ring", "--prompts", tmp_path / "prompts.jsonl", "--num-slots", "1",
                             "--official-root", tmp_path / "missing", "--master-key-file", key,
                             "--output-dir", tmp_path / "gen"], monkeypatch) == 0


def _scores(uids, edit_id, wm, clean):
    rows = []
    for uid, a, b in zip(uids, wm, clean):
        rows.append({"image_id": lib.make_image_id(uid, "watermarked", edit_id), "method": "m", "score": a})
        rows.append({"image_id": lib.make_image_id(uid, "clean", edit_id), "method": "m", "score": b})
    return rows


def test_clean_detection_and_robustness_scripts(tmp_path, monkeypatch):
    uids = ["p%d" % i for i in range(20)]
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 1.0, 20)
    marked = rng.normal(3.0, 1.0, 20)
    write_jsonl(tmp_path / "unedited.jsonl", _scores(uids, lib.IDENTITY, marked, clean))
    edited = []
    for edit_id in EDIT_IDS:
        edited.extend(_scores(uids, edit_id, marked - 0.5, clean))
    write_jsonl(tmp_path / "edited.jsonl", edited)

    _run("clean_detection", ["--scores", tmp_path / "unedited.jsonl", "--output-dir", tmp_path / "cd"], monkeypatch)
    summary = (tmp_path / "cd" / "summary.csv").read_text().splitlines()
    assert summary[0] == "method,prompts,auroc" and summary[1].startswith("m,20,")
    points = (tmp_path / "cd" / "operating_points.csv").read_text().splitlines()
    assert len(points) == 1 + len(load_detection_config().fpr_targets)

    _run("robustness", ["--scores", tmp_path / "unedited.jsonl", tmp_path / "edited.jsonl",
                        "--output-dir", tmp_path / "rb"], monkeypatch)
    per_edit = (tmp_path / "rb" / "per_edit.csv").read_text().splitlines()
    assert len(per_edit) == 1 + len(EDIT_IDS) * len(load_detection_config().fpr_targets)
    categories = (tmp_path / "rb" / "categories.csv").read_text().splitlines()
    assert len(categories) == 1 + 2 * len(load_detection_config().fpr_targets)


def test_seal_match_score_counts_patches_below_tau():
    import _baselines_native as native

    assert native.seal_match_score([0.1, 2.299, 2.3, 5.0]) == 2.0
    assert native.seal_match_score([]) == 0.0
