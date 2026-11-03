"""Generative-edit experiment: panel rule, planner prompts and plan validation, detection metrics."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "generative_edits"


def load(alias: str, filename: str):
    spec = importlib.util.spec_from_file_location(alias, str(EXPERIMENT / filename))
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXPERIMENT))
    return module


@pytest.fixture(scope="module")
def lib():
    return load("test_edits_lib_module", "_edits_lib.py")


def rows(count):
    return [{"prompt_uid": "p%03d" % index, "text": "prompt %d" % index} for index in range(count)]


def test_balanced_panel_is_balanced_unique_and_order_independent(lib):
    base = rows(120)
    first = lib.balanced_panel(base, per_category=20)
    second = lib.balanced_panel(list(reversed(base)), per_category=20)
    assert [(r["prompt_uid"], c) for r, c in first] == [(r["prompt_uid"], c) for r, c in second]
    assert len(first) == 80 and len({r["prompt_uid"] for r, _ in first}) == 80
    for category in lib.CATEGORIES:
        assert sum(c == category for _, c in first) == 20
    with pytest.raises(ValueError):
        lib.balanced_panel(rows(30), per_category=20)
    with pytest.raises(ValueError):
        lib.balanced_panel(base, per_category=0)


def test_balanced_panel_ranks_by_the_category_hash(lib):
    import hashlib

    base = rows(40)
    chosen = [r["prompt_uid"] for r, c in lib.balanced_panel(base, per_category=3) if c == "attribute_edit"]
    ranked = sorted(base, key=lambda r: hashlib.sha256(("attribute_edit:%s:" % r["prompt_uid"]).encode()).hexdigest())
    assert chosen == [r["prompt_uid"] for r in ranked[:3]]


def test_plan_prompt_uses_the_paper_constraint_and_json_contract(lib):
    for category in lib.CATEGORIES:
        prompt = lib.edit_plan_prompt(category)
        assert lib.CATEGORY_CONSTRAINTS[category] in prompt
        assert "Assigned category: %s." % category in prompt
        assert prompt.endswith("The edit_category must be exactly %s." % category)
        assert "faces, text, logos, or watermarks" in prompt
    assert "camera viewpoint" in lib.CATEGORY_CONSTRAINTS["camera_viewpoint_change"]
    with pytest.raises(ValueError):
        lib.edit_plan_prompt("global_semantic_change")


def good_plan(category="cat_attack"):
    return json.dumps({"visible_subject": "a dog", "context": "a park", "edit_category": category,
                       "edit_instruction": "Add a small cat in the background."})


def test_validate_edit_plan_accepts_only_the_exact_typed_plan(lib):
    plan = lib.validate_edit_plan(good_plan(), expected_category="cat_attack")
    assert plan["edit_instruction"].startswith("Add a small cat")
    for bad in (" " + good_plan(), good_plan() + "\n", "not json", "[]",
                json.dumps({"visible_subject": "a"}),
                json.dumps({"visible_subject": "a", "context": "b", "edit_category": "cat_attack",
                            "edit_instruction": " padded "}),
                json.dumps({"visible_subject": "a", "context": "b", "edit_category": "cat_attack",
                            "edit_instruction": ""})):
        with pytest.raises(ValueError):
            lib.validate_edit_plan(bad, expected_category="cat_attack")
    with pytest.raises(ValueError):
        lib.validate_edit_plan(good_plan("attribute_edit"), expected_category="cat_attack")


def test_pinned_models_and_render_settings(lib):
    assert lib.PLANNER["revision"].startswith("0c351dd0") and len(lib.PLANNER["revision"]) == 40
    assert lib.EDITOR["revision"].startswith("ac7f9318") and len(lib.EDITOR["revision"]) == 40
    assert (lib.EDITOR["seed"], lib.EDITOR["num_inference_steps"], lib.EDITOR["true_cfg_scale"]) == (20260830, 50, 4.0)
    assert lib.PLANNER["do_sample"] is False


def test_merge_sources_deduplicates_by_file_hash_and_keeps_roles(lib):
    entry = {"source_id": "a" * 64, "prompt_uid": "p1", "text": "t", "edit_category": "cat_attack",
             "path": "x.png", "role": "ours_watermarked"}
    merged = lib.merge_sources([entry, dict(entry, role="tree_ring_clean"), dict(entry)])
    assert len(merged) == 1 and merged[0]["roles"] == ["ours_watermarked", "tree_ring_clean"]
    with pytest.raises(ValueError):
        lib.merge_sources([entry, dict(entry, prompt_uid="p2")])


def test_deal_is_round_robin_and_validates(lib):
    assert lib.deal(list(range(7)), 1, 3) == [1, 4]
    with pytest.raises(ValueError):
        lib.deal([1], 3, 3)


def test_detection_metrics_freeze_tau_on_clean_scores(lib):
    calibration = np.linspace(0.0, 0.99, 100)  # clean scores
    original = np.array([0.99, 0.995, 0.5, 0.996])
    edited = np.array([0.999, 0.1, 0.999, 0.1])
    edited_clean = np.array([0.1, 0.95, 0.2, 0.3])
    result = lib.detection_metrics(calibration, original, edited, edited_clean, 0.05)
    assert result["calibration_fpr"] <= 0.05
    tau = result["threshold"]
    assert result["original_tpr"] == pytest.approx(np.mean(original >= tau))
    assert result["edited_tpr"] == pytest.approx(0.5)
    originally_detected = original >= tau
    assert result["retention"] == pytest.approx(
        np.sum((edited >= tau) & originally_detected) / np.sum(originally_detected))
    assert result["edited_fpr"] == pytest.approx(np.mean(edited_clean >= tau))


def test_retention_is_nan_without_original_detections(lib):
    result = lib.detection_metrics(np.linspace(0, 1, 50), np.zeros(4), np.ones(4), np.zeros(4), 0.2)
    assert math.isnan(result["retention"]) and result["edited_tpr"] == 1.0


def test_category_metrics_pool_all_categories(lib):
    categories = [c for c in lib.CATEGORIES for _ in range(3)]
    rng = np.random.default_rng(0)
    original, edited, clean = rng.random(12), rng.random(12), rng.random(12)
    result = lib.category_metrics(rng.random(60), categories, original, edited, clean, 0.2)
    assert set(result) == {*lib.CATEGORIES, "all"}
    assert result["all"]["edits"] == 12 and all(result[c]["edits"] == 3 for c in lib.CATEGORIES)
    tau = result["all"]["threshold"]
    assert result["all"]["edited_tpr"] == pytest.approx(np.mean(edited >= tau))
    with pytest.raises(ValueError):
        lib.category_metrics(rng.random(60), ["attribute_edit"] * 12, original, edited, clean, 0.2)


def test_average_replicates_means_rates_and_skips_undefined_retention(lib):
    base = {"method": "m", "fpr_target": 0.2, "category": "all", "edits": 80, "original_tpr": 0.5,
            "edited_tpr": 0.4, "edited_fpr": 0.2, "calibration_fpr": 0.2}
    averaged = lib.average_replicates([
        dict(base, retention=0.5), dict(base, retention=float("nan"), edited_tpr=0.6),
        dict(base, retention=0.7)])
    assert len(averaged) == 1
    row = averaged[0]
    assert row["replicates"] == 3 and row["retention"] == pytest.approx(0.6)
    assert row["edited_tpr"] == pytest.approx((0.4 + 0.6 + 0.4) / 3)
