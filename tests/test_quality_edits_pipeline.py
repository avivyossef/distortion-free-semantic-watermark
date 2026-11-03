"""Generative-edit experiment: source listing, replicate scoring and baseline scoring on synthetic data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from test_quality_edits_lib import load  # noqa: E402  (same directory; tests/ is on sys.path)
from watermark.config import validate_detection_config

DIM, PROMPTS, CANDIDATES = 12, 50, 8


@pytest.fixture(scope="module")
def modules():
    return {name: load("edits_test_" + name[:-3], name) for name in
            ("build_sources.py", "score_ours.py", "prepare_baseline_scoring.py", "summarize.py")}


@pytest.fixture(scope="module")
def cfg():
    return validate_detection_config({
        "prf": "hmac-sha256",
        "configs": [{"N": 8, "M": 4, "activation": 0.10, "radius": 1}],
        "activation_population": "clean_watermarked",
        "calibration": {"population": "clean", "folds": 5, "fpr_targets": [0.2]},
        "replication": {"frames": 2, "keys": 2, "frame_seed_base": 7},
        "set_sizes": [1], "aggregation": "mean", "exceptions": {},
    })


def unit(rng, *shape):
    values = rng.normal(size=shape)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("edits")
    rng = np.random.default_rng(3)
    uids = ["p%03d" % index for index in range(PROMPTS)]
    cache = {"prompt_uids": uids, "candidates": unit(rng, PROMPTS, CANDIDATES, DIM), "cleans": unit(rng, PROMPTS, DIM)}
    panel = [{"prompt_uid": uid, "text": "t " + uid, "edit_category": category}
             for uid, category in zip(uids[:8], ["attribute_edit", "background_cleanup", "cat_attack",
                                                 "camera_viewpoint_change"] * 2)]
    corpus, baseline = root / "corpus", root / "baseline"
    for row in panel:
        uid = row["prompt_uid"]
        (corpus / uid).mkdir(parents=True)
        (baseline / uid).mkdir(parents=True)
        candidates = [{"candidate_position": p + 1, "relative_path": "slot_001/candidate_%02d.png" % (p + 1),
                       "file_sha256": "%s-c%d" % (uid, p + 1)} for p in range(CANDIDATES)]
        record = {"prompt_uid": uid, "mode": "candidates", "batch_slots": [
            {"slot": 1, "candidate_images": candidates,
             "clean": {"relative_path": "slot_001/clean.png", "file_sha256": uid + "-clean"}}]}
        (corpus / uid / "panel.json").write_text(json.dumps(record))
        record = {"prompt_uid": uid, "slots": [{"slot": 1,
                  "watermarked": {"relative_path": "slot_001/watermarked.png", "file_sha256": uid + "-bw"},
                  "clean": {"relative_path": "slot_001/clean.png", "file_sha256": uid + "-bc"}}]}
        (baseline / uid / "panel.json").write_text(json.dumps(record))
    for uid in uids[8:]:  # baseline calibration images exist for every prompt
        (baseline / uid).mkdir(parents=True)
        record = {"prompt_uid": uid, "slots": [{"slot": 1,
                  "watermarked": {"relative_path": "slot_001/watermarked.png", "file_sha256": uid + "-bw"},
                  "clean": {"relative_path": "slot_001/clean.png", "file_sha256": uid + "-bc"}}]}
        (baseline / uid / "panel.json").write_text(json.dumps(record))
    return {"cache": cache, "panel": panel, "corpus": corpus, "baseline": baseline, "key": bytes(range(32)),
            "rng": rng}


def test_sources_cover_clean_controls_and_every_selected_candidate(modules, cfg, world):
    entries = modules["build_sources.py"].ours_entries(world["panel"], world["corpus"], world["cache"], cfg, world["key"])
    roles = {entry["role"] for entry in entries}
    assert roles == {"ours_clean", "ours_watermarked"}
    assert sum(entry["role"] == "ours_clean" for entry in entries) == len(world["panel"])
    assert len(entries) == len(world["panel"]) * (1 + 2 * 2)  # clean + frames x keys selections
    lib = modules["build_sources.py"]
    merged = lib.merge_sources(entries)
    assert len({row["source_id"] for row in merged}) == len(merged) <= len(entries)
    for entry in entries:
        if entry["role"] == "ours_watermarked":
            assert entry["source_id"].startswith(entry["prompt_uid"] + "-c")
    base = modules["build_sources.py"].baseline_entries(world["panel"], "seal", world["baseline"])
    assert {entry["role"] for entry in base} == {"seal_watermarked", "seal_clean"} and len(base) == 16


def test_ours_replicate_rows_use_frozen_detector(modules, cfg, world):
    score = modules["score_ours.py"]
    entries = modules["build_sources.py"].ours_entries(world["panel"], world["corpus"], world["cache"], cfg, world["key"])
    rng = np.random.default_rng(5)
    vectors = {entry["source_id"]: unit(rng, DIM) for entry in entries}
    rows = score.score_replicates(world["panel"], world["cache"], world["corpus"], cfg, world["key"], vectors)
    categories = {"attribute_edit", "background_cleanup", "cat_attack", "camera_viewpoint_change", "all"}
    assert len(rows) == 1 * 2 * 2 * 1 * len(categories)
    assert {row["category"] for row in rows} == categories
    pooled = [row for row in rows if row["category"] == "all"]
    assert all(row["edits"] == 8 and row["method"] == "ours_n8_m4" for row in pooled)
    for row in pooled:
        assert 0.0 <= row["edited_tpr"] <= 1.0 and 0.0 <= row["edited_fpr"] <= 1.0
        assert row["calibration_fpr"] <= 0.2
    # an unedited image is detected exactly as its original score says: feeding the source embeddings
    # back as the "edits" gives retention 1 (or undefined when nothing was detected)
    candidates = {}
    for row in world["panel"]:
        uid = row["prompt_uid"]
        index = world["cache"]["prompt_uids"].index(uid)
        for position in range(CANDIDATES):
            candidates["%s-c%d" % (uid, position + 1)] = world["cache"]["candidates"][index, position]
        candidates[uid + "-clean"] = world["cache"]["cleans"][index]
    identical = score.score_replicates(world["panel"], world["cache"], world["corpus"], cfg, world["key"], candidates)
    for row in identical:
        assert row["retention"] == 1.0 or np.isnan(row["retention"])
        assert row["edited_tpr"] == pytest.approx(row["original_tpr"])
    # the same call twice is deterministic
    again = score.score_replicates(world["panel"], world["cache"], world["corpus"], cfg, world["key"], candidates)
    assert [r["edited_tpr"] for r in again] == [r["edited_tpr"] for r in identical]


def test_missing_edit_is_reported(modules, cfg, world):
    with pytest.raises(ValueError):
        modules["score_ours.py"].score_replicates(world["panel"], world["cache"], world["corpus"], cfg,
                                                  world["key"], {})


def test_panel_outside_calibration_prompts_is_rejected(modules, world):
    lib = modules["build_sources.py"]
    outside = [{"prompt_uid": "other", "text": "t", "edit_category": "cat_attack"}]
    with pytest.raises(ValueError):
        lib.panel_rows_in_cache(outside, world["cache"])


def test_edited_embeddings_roundtrip(modules, tmp_path):
    path = tmp_path / "edited.npz"
    vectors = np.eye(3)
    np.savez_compressed(str(path), source_ids=np.asarray(["a", "b", "c"]), vectors=vectors)
    loaded = modules["score_ours.py"].load_edited_vectors(path)
    assert set(loaded) == {"a", "b", "c"} and np.allclose(loaded["b"], vectors[1])


def test_baseline_listing_and_summary(modules, world):
    prompts = [{"prompt_uid": uid, "text": "t " + uid} for uid in world["cache"]["prompt_uids"]]
    edited = {}
    for row in world["panel"]:
        edited[row["prompt_uid"] + "-bw"] = Path("edits") / (row["prompt_uid"] + "-bw.png")
        edited[row["prompt_uid"] + "-bc"] = Path("edits") / (row["prompt_uid"] + "-bc.png")
    rows = modules["prepare_baseline_scoring.py"].baseline_images(world["baseline"], prompts, world["panel"], edited)
    ids = [row["image_id"] for row in rows]
    assert len(ids) == len(set(ids)) == PROMPTS + 3 * len(world["panel"])
    assert sum(i.startswith("calibration-clean:") for i in ids) == PROMPTS
    with pytest.raises(ValueError):
        modules["prepare_baseline_scoring.py"].baseline_images(world["baseline"], prompts, world["panel"], {})

    rng = np.random.default_rng(1)
    scores = []
    for image_id in ids:
        group = image_id.split(":")[0]
        value = {"calibration-clean": rng.random(), "original": 5.0, "edited-watermarked": 5.0 if image_id.endswith(
            ("p000", "p001", "p002", "p003")) else -5.0, "edited-clean": rng.random()}[group]
        scores.append({"image_id": image_id, "method": "seal", "score": value})
    summary = modules["summarize.py"].baseline_rows("seal", scores, world["panel"], [0.2])
    pooled = [row for row in summary if row["category"] == "all"][0]
    assert pooled["original_tpr"] == 1.0 and pooled["edited_tpr"] == 0.5 and pooled["retention"] == 0.5
    assert pooled["calibration_fpr"] <= 0.2 and pooled["edits"] == 8
    with pytest.raises(ValueError):
        modules["summarize.py"].baseline_rows("seal", scores[:-3], world["panel"], [0.2])
    averaged = modules["summarize.py"].average_replicates(summary)
    assert {row["category"] for row in averaged} == {r["category"] for r in summary}


def test_summary_reads_our_replicate_csv(modules, tmp_path):
    path = tmp_path / "replicates.csv"
    fields = ["method", "frame_id", "key_trial", "fpr_target", "category", "threshold", "calibration_fpr",
              "original_tpr", "edited_tpr", "retention", "edited_fpr", "edits"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial, tpr in enumerate((0.4, 0.6)):
            writer.writerow({"method": "ours_n8_m4", "frame_id": "frame-00", "key_trial": trial, "fpr_target": 0.2,
                             "category": "all", "threshold": 0.9, "calibration_fpr": 0.2, "original_tpr": 0.5,
                             "edited_tpr": tpr, "retention": 0.5, "edited_fpr": 0.2, "edits": 80})
    rows = modules["summarize.py"].read_ours(path)
    averaged = modules["summarize.py"].average_replicates(rows)
    assert len(averaged) == 1 and averaged[0]["edited_tpr"] == pytest.approx(0.5) and averaged[0]["replicates"] == 2


@pytest.mark.parametrize("script", ["select_panel.py", "plan_instructions.py", "render_edits.py", "embed_edits.py"])
def test_gpu_and_cli_scripts_import_without_heavy_packages(script):
    module = load("edits_import_" + script[:-3], script)
    assert callable(module.main)


def test_select_panel_writes_a_balanced_panel(tmp_path, monkeypatch):
    module = load("edits_select_panel", "select_panel.py")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt_uid": "p%03d" % i, "text": "t%d" % i}) + "\n" for i in range(40)))
    monkeypatch.setattr("sys.argv", ["select_panel.py", "--prompts", str(prompts), "--per-category", "5",
                                     "--output-dir", str(tmp_path / "out")])
    assert module.main() == 0
    panel = [json.loads(line) for line in (tmp_path / "out" / "panel.jsonl").read_text().splitlines()]
    assert len(panel) == 20 and len({row["prompt_uid"] for row in panel}) == 20
    assert {row["edit_category"] for row in panel} == {"attribute_edit", "background_cleanup", "cat_attack",
                                                       "camera_viewpoint_change"}
