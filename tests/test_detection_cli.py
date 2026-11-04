"""Command-line entry points end to end on a synthetic cache with a small detection config."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import detection_helpers as helpers
import _robustness_lib as _lib
import analyze_robustness
import detection_corpus as corpus
import plot_clean_roc
import render_robustness
import run_clean_detection
import run_rules
import run_sweep
import summarize_clean_detection
import summarize_robustness
import summarize_rules
import summarize_sweep


@pytest.fixture()
def files(tmp_path, monkeypatch):
    cfg = helpers.small_config(frames=2, keys=1, set_sizes=(1, 3))
    for module in (run_clean_detection, summarize_clean_detection, analyze_robustness, summarize_robustness,
                   run_rules, summarize_rules, run_sweep, summarize_sweep,
                   render_robustness):
        monkeypatch.setattr(module, "load_detection_config", lambda cfg=cfg: cfg, raising=False)
    cache = helpers.synthetic_cache(prompts=100, slots=3, n=16)
    corpus.write_cache(tmp_path / "cache.npz", cache.prompt_uids, cache.candidate_vectors, cache.clean_vectors,
                       {"encoder": "ours"})
    (tmp_path / "detector.key").write_bytes(helpers.KEY)
    (tmp_path / "prompts.jsonl").write_text("".join(
        json.dumps({"prompt_uid": uid, "text": "t"}) + "\n" for uid in cache.prompt_uids))
    return cfg, cache, tmp_path


def run(main, *argv):
    sys.argv = ["script", *map(str, argv)]
    assert main() == 0


def test_clean_detection_pipeline_with_two_shards(files):
    cfg, _, tmp = files
    for shard in range(2):
        run(run_clean_detection.main, "--cache", tmp / "cache.npz", "--key-file", tmp / "detector.key",
            "--output-dir", tmp / "clean", "--shard-index", shard, "--num-shards", 2)
    run(summarize_clean_detection.main, "--shards-dir", tmp / "clean", "--output-dir", tmp / "summary")
    run(plot_clean_roc.main, "--mean-roc", tmp / "summary" / "mean_roc.json", "--output-dir", tmp / "figures")
    assert (tmp / "summary" / "summary.csv").read_text().startswith("N,M,B,")
    assert (tmp / "figures" / "experiment2_detection_roc_log.pdf").exists()
    run(run_clean_detection.main, "--cache", tmp / "cache.npz", "--key-file", tmp / "detector.key",
        "--output-dir", tmp / "clean_workers", "--workers", 2)
    both = [r for r in map(json.loads, (tmp / "clean_workers" / "replicates-shard-000-of-001.jsonl")
                           .read_text().splitlines())]
    assert len(both) == len(cfg.settings) * cfg.frames * cfg.keys * len(cfg.set_sizes)


def test_robustness_pipeline_and_downstream_experiments(files):
    cfg, cache, tmp = files
    planes = corpus.frame_planes(cfg, 24)
    keys = corpus.trial_keys(helpers.KEY, cfg)
    sources = _lib.all_selected_sources(cache, cfg, planes, keys, ["representative", "gumbel", "tournament"])
    edits = helpers.synthetic_edits(cache, sources)
    (tmp / "edits").mkdir()
    np.savez_compressed(str(tmp / "edits" / "edits-shard-000-of-001.npz"), source_ids=np.asarray(sources), edit_ids=np.asarray(edits.edit_ids),
                        embeddings=edits.embeddings)
    run(analyze_robustness.main, "--cache", tmp / "cache.npz", "--prompts", tmp / "prompts.jsonl", "--edits",
        tmp / "edits", "--key-file", tmp / "detector.key", "--output-dir", tmp / "robustness")
    run(summarize_robustness.main, "--shards-dir", tmp / "robustness", "--output-dir", tmp / "robustness_summary")
    run(render_robustness.main, "--summary-dir", tmp / "robustness_summary", "--output-dir", tmp / "tables")
    assert "tab:robustness_detector_ablation" in (tmp / "tables" / "robustness_detector_ablation.tex").read_text()
    # sweep and sampling rules reuse the same edit embeddings
    run(run_sweep.main, "--cache", tmp / "cache.npz", "--prompts", tmp / "prompts.jsonl", "--edits", tmp / "edits",
        "--key-file", tmp / "detector.key", "--output-dir", tmp / "sweep")
    run(summarize_sweep.main, "--shards-dir", tmp / "sweep", "--output", tmp / "sweep_summary.csv")
    run(run_rules.main, "--cache", tmp / "cache.npz", "--key-file", tmp / "detector.key", "--output-dir", tmp / "rules")
    run(run_rules.main, "--cache", tmp / "cache.npz", "--prompts", tmp / "prompts.jsonl", "--edits", tmp / "edits",
        "--key-file", tmp / "detector.key", "--output-dir", tmp / "rules")
    run(summarize_rules.main, "--shards-dir", tmp / "rules", "--output-dir", tmp / "rules_tables")
    assert (tmp / "rules_tables" / "sampling_rules_edits.csv").exists()
    assert (tmp / "sweep_summary.csv").read_text().startswith("N,M,activation,radius")


def test_edits_without_prompts_are_rejected(files):
    _, _, tmp = files
    sys.argv = ["script", "--cache", str(tmp / "cache.npz"), "--edits", str(tmp), "--key-file",
                str(tmp / "detector.key"), "--output-dir", str(tmp / "x")]
    with pytest.raises(SystemExit):
        run_rules.main()
