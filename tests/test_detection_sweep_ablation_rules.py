"""Detector sweep, encoder ablation and sampling-rule experiments on synthetic corpora."""

from __future__ import annotations

import numpy as np
import pytest

import detection_helpers as helpers
import _robustness_lib as _lib
import detection_corpus as corpus
import plot_ablation
import run_ablation
import run_rules
import run_sweep
import summarize_rules
import summarize_sweep

from watermark import detection
from watermark.config import load_detection_config
from watermark.selection_rules import RULES


@pytest.fixture(scope="module")
def world():
    cfg = helpers.small_config(frames=2, keys=1)
    cache = helpers.synthetic_cache(prompts=100, slots=1, n=16)
    planes = corpus.frame_planes(cfg, 24)
    keys = corpus.trial_keys(helpers.KEY, cfg)
    sources = _lib.all_selected_sources(cache, cfg, planes, keys, list(RULES))
    return cfg, cache, planes, keys, helpers.synthetic_edits(cache, sources)


def test_shipped_sweep_grid_is_valid_and_includes_the_selected_settings(tmp_path):
    sweep = run_sweep.load_sweep_config()
    cfg = load_detection_config()
    assert 0.0 in sweep["activation_targets"] and 0 in sweep["radii"]  # exact-code reference point
    assert all(setting.activation in sweep["activation_targets"] and setting.radius in sweep["radii"]
               for setting in cfg.settings)
    bad = tmp_path / "bad.yaml"
    bad.write_text("activation_targets: [1.5]\nradii: [0]\n")
    with pytest.raises(ValueError):
        run_sweep.load_sweep_config(bad)


def test_sweep_grid_points_match_direct_scoring(world):
    cfg, cache, planes, keys, edits = world
    sweep = {"activation_targets": [0.0, 0.25, 1.0], "radii": [0, 1, 2]}
    setting = cfg.settings[1]
    rows = run_sweep.sweep_rows(cache, planes["frame-00"], keys[0], setting, edits, sweep, frame_id="frame-00",
                                key_trial=0, cfg=cfg)
    assert len(rows) == 9
    configured = detection.replicate_scores(
        cache.candidate_vectors[:, 0, :16], cache.clean_vectors[:, 0], cache.prompt_uids, planes["frame-00"], keys[0],
        frame_id="frame-00", key_trial=0, n=16, m=8, activation=0.25, radius=1, folds=5)
    row = next(r for r in rows if (r["activation"], r["radius"]) == (0.25, 1))
    from watermark.metrics import auroc

    assert row["auc"] == auroc(configured.clean.scores, configured.watermarked.scores)
    assert row["realized_activation"] == pytest.approx(np.mean(configured.watermarked.neighbourhood_sizes > 1))
    # exact-code detection ignores delta; full activation widens every neighbourhood
    exact = [r for r in rows if r["radius"] == 0]
    assert len({r["auc"] for r in exact}) == 1
    assert next(r for r in rows if (r["activation"], r["radius"]) == (1.0, 2))["realized_activation"] == 1.0
    assert next(r for r in rows if (r["activation"], r["radius"]) == (0.0, 1))["auc"] == exact[0]["auc"]
    for r in rows:
        assert 0.0 <= r["edited_tpr_20"] <= 1.0 and 0.0 <= r["edited_fpr_20"] <= 1.0


def test_sweep_summary_reports_differences_from_exact_detection(world, tmp_path):
    cfg, cache, planes, keys, edits = world
    sweep = {"activation_targets": [0.0, 0.25], "radii": [0, 1]}
    rows = []
    for setting in cfg.settings:
        for frame_id, _, trial in corpus.replicate_cells(cfg):
            rows.extend(run_sweep.sweep_rows(cache, planes[frame_id], keys[trial], setting, edits, sweep,
                                             frame_id=frame_id, key_trial=trial, cfg=cfg))
    merged = summarize_sweep.merge(rows)
    assert len(merged) == len(cfg.settings) * 4
    exact = next(r for r in merged if (r["N"], r["activation"], r["radius"]) == (8, 0.0, 0))
    assert exact["auc_delta_from_exact"] == 0.0 and exact["replicate_count"] == 2
    other = next(r for r in merged if (r["N"], r["activation"], r["radius"]) == (8, 0.25, 1))
    assert other["auc_delta_from_exact"] == pytest.approx(other["auc"] - exact["auc"])


def test_ablation_of_the_trained_encoder_scores_the_edit_stream_replicate(world):
    from watermark.metrics import auroc

    cfg, cache, planes, keys, _ = world
    setting = cfg.settings[0]
    mine = run_ablation.ablation_row(cache, planes["frame-01"], keys[0], setting, frame_id="frame-01", key_trial=0,
                                     cfg=cfg)
    replicate = detection.replicate_scores(
        cache.candidate_vectors[:, 0, :setting.n], cache.clean_vectors[:, 0], cache.prompt_uids, planes["frame-01"],
        keys[0], frame_id="frame-01", key_trial=0, n=setting.n, m=setting.m, activation=setting.activation,
        radius=setting.radius, folds=cfg.folds)
    assert mine["auc"] == auroc(replicate.clean.scores, replicate.watermarked.scores)
    assert len(mine["roc_tpr"]) == len(run_ablation.ROC_GRID)


def test_ablation_runs_in_the_encoders_native_dimension_and_plots(world, tmp_path):
    cfg, _, _, _, _ = world
    results = {}
    for index, (name, dim) in enumerate((("ours", 24), ("dinov2_base", 40))):
        cache = helpers.synthetic_cache(prompts=100, slots=1, n=16, dim=dim, seed=index)
        planes = corpus.frame_planes(cfg, dim)
        keys = corpus.trial_keys(helpers.KEY, cfg)
        rows = [run_ablation.ablation_row(cache, planes[f], keys[t], s, frame_id=f, key_trial=t, cfg=cfg)
                for s in cfg.settings for f, _, t in corpus.replicate_cells(cfg)]
        assert {r["dimension"] for r in rows} == {dim}
        corpus.write_jsonl_rows(tmp_path / name / "replicates-shard-000-of-001.jsonl", rows)
        results[name] = plot_ablation.load_encoder_rows(tmp_path / name, cfg)
    summary = plot_ablation.summarize(results, cfg)
    assert {(r["encoder"], r["N"]) for r in summary} == {(e, n) for e in results for n in (8, 16)}
    plot_ablation.plot(results, cfg, tmp_path / "ablation.pdf")
    assert (tmp_path / "ablation.pdf").stat().st_size > 1000
    with pytest.raises(ValueError):
        corpus.write_jsonl_rows(tmp_path / "ours" / "replicates-shard-000-of-001.jsonl", results["ours"][:-1])
        plot_ablation.load_encoder_rows(tmp_path / "ours", cfg)


def test_gumbel_clean_row_is_the_clean_detection_pipeline(world):
    cfg, cache, planes, keys, _ = world
    setting = cfg.settings[0]
    row = run_rules.clean_row(cache, planes["frame-00"], keys[0], setting, "gumbel", frame_id="frame-00",
                              key_trial=0, cfg=cfg)
    import run_clean_detection as clean

    reference = clean.replicate_rows(cache, planes["frame-00"], keys[0], setting, frame_id="frame-00", key_trial=0,
                                     cfg=helpers.small_config(frames=2, keys=1, set_sizes=(1,)))[0]
    assert row["auc"] == reference["auc"] and row["tpr_interp_20"] == reference["tpr_interp_20"]
    other = run_rules.clean_row(cache, planes["frame-00"], keys[0], setting, "representative", frame_id="frame-00",
                                key_trial=0, cfg=cfg)
    assert other["auc"] != row["auc"]


def test_rule_tables_cover_rules_and_edits(world, tmp_path):
    cfg, cache, planes, keys, edits = world
    clean_rows, edit_rows = [], []
    for rule in RULES:
        for setting in cfg.settings:
            for frame_id, _, trial in corpus.replicate_cells(cfg):
                clean_rows.append(run_rules.clean_row(cache, planes[frame_id], keys[trial], setting, rule,
                                                      frame_id=frame_id, key_trial=trial, cfg=cfg))
                edit_rows.extend(run_rules.edit_rows(cache, planes[frame_id], keys[trial], setting, rule, edits,
                                                     frame_id=frame_id, key_trial=trial, cfg=cfg))
    summary = summarize_rules.clean_summary(clean_rows, cfg)
    assert len(summary) == len(RULES) * len(cfg.settings)
    table = summarize_rules.latex_table(summary, cfg)
    assert table.count("textbf") >= 2 * len(cfg.settings) and "Gumbel" in table and "Tournament" in table
    assert "\\((8,4)\\)" in table
    edits_summary = summarize_rules.edit_summary(edit_rows, cfg)
    assert len(edits_summary) == len(RULES) * len(cfg.settings)
    assert all(0.0 <= r["edited_tpr"] <= 1.0 and 0.0 <= r["retention"] <= 1.0 for r in edits_summary)
    with pytest.raises(ValueError):
        summarize_rules.clean_summary(clean_rows[:-1], cfg)


def test_mark_ranks_ties_and_second_best():
    assert summarize_rules._marks([0.9, 0.9, 0.5]) == ["samplingbest", "samplingbest", "samplingsecond"]
    assert summarize_rules._marks([0.2, 0.9, 0.5]) == ["", "samplingbest", "samplingsecond"]
