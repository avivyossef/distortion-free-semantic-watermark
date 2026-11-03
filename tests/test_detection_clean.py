"""Clean-detection experiment: replicate metrics, set sizes, summary, operating points, ROC figure."""

from __future__ import annotations

import json

import numpy as np
import pytest

import detection_helpers as helpers
import detection_corpus as corpus
import plot_clean_roc
import run_clean_detection as run
import summarize_clean_detection as summ

from watermark import detection, metrics


def _grid_rows(cfg, cache):
    planes = corpus.frame_planes(cfg, cache.candidate_vectors.shape[-1])
    keys = corpus.trial_keys(helpers.KEY, cfg)
    rows = []
    for setting in cfg.settings:
        for frame_id, _, trial in corpus.replicate_cells(cfg):
            rows.extend(run.replicate_rows(cache, planes[frame_id], keys[trial], setting, frame_id=frame_id,
                                           key_trial=trial, cfg=cfg))
    return rows


def test_set_scores_are_the_mean_of_independent_slot_scores():
    cfg = helpers.small_config()
    cache = helpers.synthetic_cache(prompts=100, slots=3, n=16)
    planes = corpus.frame_planes(cfg, 24)["frame-00"]
    key = corpus.trial_keys(helpers.KEY, cfg)[0]
    setting = cfg.settings[0]
    rows = run.replicate_rows(cache, planes, key, setting, frame_id="frame-00", key_trial=0, cfg=cfg)
    assert [r["B"] for r in rows] == [1, 3]
    slots = [detection.replicate_scores(cache.candidate_vectors[:, s, :8], cache.clean_vectors[:, s],
                                        cache.prompt_uids, planes, key, frame_id="frame-00", key_trial=0, n=8, m=4,
                                        activation=setting.activation, radius=setting.radius, folds=cfg.folds,
                                        slot=s + 1) for s in range(3)]
    clean = detection.aggregate_scores([s.clean.scores for s in slots])
    marked = detection.aggregate_scores([s.watermarked.scores for s in slots])
    assert rows[1]["auc"] == metrics.auroc(clean, marked)
    assert rows[0]["auc"] == metrics.auroc(slots[0].clean.scores, slots[0].watermarked.scores)
    assert rows[1]["sample_count"] == 100 and len(rows[1]["roc_tpr"]) == run.ROC_GRID_POINTS
    # the threshold is frozen on clean scores only, so the achieved FPR never exceeds the target
    for target in cfg.fpr_targets:
        assert rows[1]["frozen_fpr_%s" % run.target_label(target)] <= target
    # selection lifts watermarked scores, and sets separate better than single images
    assert rows[1]["auc"] > rows[0]["auc"] > 0.5


def test_roc_grid_ends_at_the_largest_fpr_target():
    cfg = helpers.small_config()
    grid = run.roc_grid(cfg)
    assert grid[0] == 0.0 and grid[-1] == max(cfg.fpr_targets) and len(grid) == run.ROC_GRID_POINTS
    assert run.target_label(0.01) == "01" and run.target_label(0.2) == "20"


def test_summary_requires_the_complete_grid_and_reports_means(tmp_path):
    cfg = helpers.small_config()
    cache = helpers.synthetic_cache(prompts=100, slots=3, n=16)
    rows = _grid_rows(cfg, cache)
    path = tmp_path / "replicates-shard-000-of-001.jsonl"
    corpus.write_jsonl_rows(path, rows)
    loaded = summ.load_rows(tmp_path, cfg)
    summary, mean_roc = summ.summarize(loaded, cfg)
    assert len(summary) == len(cfg.settings) * len(cfg.set_sizes)
    entry = next(e for e in summary if (e["N"], e["B"]) == (8, 3))
    members = [r for r in rows if (r["N"], r["B"]) == (8, 3)]
    assert entry["replicate_count"] == 4 == len(members)
    assert entry["auc_mean"] == pytest.approx(np.mean([r["auc"] for r in members]))
    curve = next(c for c in mean_roc["ours"] if (c["N"], c["B"]) == (8, 3))
    assert np.allclose(curve["mean"], np.mean([r["roc_tpr"] for r in members], axis=0))
    assert len(mean_roc["grid"]) == len(curve["mean"])
    lines = summ.operating_points_rows(summary, cfg)
    assert lines[0].startswith("Ours $(N{=}8,M{=}4)$ & 1 &")
    table = summ.detection_table_rows(summary, cfg)
    assert len(table) == 2 and table[0].startswith("Ours $(N=8,M=4)$ & ") and table[0].count("&") == 4 and any("addlinespace" in line for line in lines)
    corpus.write_jsonl_rows(path, rows[:-1])
    with pytest.raises(ValueError):
        summ.load_rows(tmp_path, cfg)
    corpus.write_jsonl_rows(path, rows + rows[:1])
    with pytest.raises(ValueError):
        summ.load_rows(tmp_path, cfg)


def test_replicate_rows_survive_json_with_infinite_thresholds(tmp_path):
    cfg = helpers.small_config()
    cache = helpers.synthetic_cache(prompts=100, slots=3, n=16)
    planes = corpus.frame_planes(cfg, 24)
    row = run.replicate_rows(cache, planes["frame-00"], corpus.trial_keys(helpers.KEY, cfg)[0], cfg.settings[0],
                             frame_id="frame-00", key_trial=0, cfg=cfg)[0]
    path = tmp_path / "rows.jsonl"
    corpus.write_jsonl_rows(path, [row])
    from watermark.io import read_jsonl

    assert read_jsonl(path)[0]["auc"] == row["auc"]
    assert json.loads(json.dumps(row))["N"] == 8


def test_roc_figures_are_written(tmp_path):
    grid = np.linspace(0, 0.2, 201)
    data = {"grid": grid.tolist(), "ours": [
        {"N": n, "M": m, "B": b, "mean": np.clip(grid * (1 + b + n / 8), 0, 1).tolist()}
        for n, m in ((8, 4), (16, 8)) for b in (1, 5)]}
    plot_clean_roc.render(data, False, tmp_path / "linear.pdf")
    plot_clean_roc.render(data, True, tmp_path / "log.pdf")
    assert (tmp_path / "linear.pdf").stat().st_size > 1000 and (tmp_path / "log.pdf").stat().st_size > 1000
