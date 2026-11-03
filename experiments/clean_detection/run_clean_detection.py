#!/usr/bin/env python3
"""Clean detection of the watermark: ROC, AUROC and operating points per (N, M) and set size B.

For every configuration of ``configs/detection.yaml``, hyperplane frame and PRF key (the 25
replicates of the paper), each slot of the embedding cache is one independent candidate pool:
candidate selection, boundary calibration and boundary-aware scoring run per slot
(``watermark.detection.replicate_scores``); a set of B images scores the mean of the per-slot
scores of slots 1..B. The score threshold tau is frozen on the clean (unwatermarked) set scores.

    python experiments/clean_detection/run_clean_detection.py --cache cache.npz \
        --key-file keys/detector.key --output-dir clean_detection --shard-index 0 --num-shards 1

Replicates are dealt round-robin to shards; shard ``I`` writes ``replicates-shard-III-of-KKK.jsonl``
(one row per replicate and B, with the scalar metrics and the replicate's ROC resampled to an
FPR grid). ``summarize_clean_detection.py`` merges the shards.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))

from detection_corpus import (deal, frame_planes, load_cache, read_master_key, replicate_cells,  # noqa: E402
                    trial_keys, write_jsonl_rows)

from watermark.calibration import clean_threshold  # noqa: E402
from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.detection import aggregate_scores, replicate_scores  # noqa: E402
from watermark.metrics import clean_detection_roc, interpolate_roc, tpr_at_fpr  # noqa: E402

ROC_GRID_POINTS = 201


def target_label(fpr: float) -> str:
    """Column label of an FPR target: 0.01 -> ``01``, 0.2 -> ``20``."""
    return "%02d" % int(round(100 * fpr))


def roc_grid(cfg: DetectionConfig) -> np.ndarray:
    """FPR grid of the exported ROC curves: from 0 to the largest FPR target."""
    return np.linspace(0.0, max(cfg.fpr_targets), ROC_GRID_POINTS)


def replicate_rows(cache, planes, key, setting, *, frame_id: str, key_trial: int, cfg: DetectionConfig
                   ) -> List[Dict[str, Any]]:
    """Metrics of one (N, M, frame, key) replicate for every set size B."""
    n, m = setting.n, setting.m
    slots = max(cfg.set_sizes)
    per_slot = [
        replicate_scores(cache.candidate_vectors[:, s, :n], cache.clean_vectors[:, s], cache.prompt_uids, planes,
                         key, frame_id=frame_id, key_trial=key_trial, n=n, m=m, activation=setting.activation,
                         radius=setting.radius, folds=cfg.folds, slot=s + 1)
        for s in range(slots)
    ]
    grid = roc_grid(cfg)
    rows = []
    for b in cfg.set_sizes:
        used = per_slot[:b]
        clean = aggregate_scores([item.clean.scores for item in used])
        watermarked = aggregate_scores([item.watermarked.scores for item in used])
        result = clean_detection_roc(clean, watermarked, cfg.fpr_targets)
        row: Dict[str, Any] = {
            "N": n, "M": m, "B": b, "frame_id": frame_id, "key_trial": key_trial,
            "sample_count": len(cache.prompt_uids), "auc": result["auc"],
            "activation": setting.activation, "radius": setting.radius,
            "delta_mean": float(np.mean([np.mean(item.deltas) for item in used])),
            "clean_activation_rate": float(np.mean([np.mean(item.clean.neighbourhood_sizes > 1) for item in used])),
            "watermarked_activation_rate": float(np.mean([np.mean(item.watermarked.neighbourhood_sizes > 1)
                                                          for item in used])),
        }
        for target, point in zip(cfg.fpr_targets, result["fixed_fpr"]):
            label = target_label(target)
            tau, frozen_fpr = clean_threshold(clean, target)
            row["tpr_roc_%s" % label] = tpr_at_fpr(result["roc"], target)["tpr"]
            row["tpr_interp_%s" % label] = float(interpolate_roc(result["roc"], [target])[0])
            row["tau_%s" % label] = tau
            row["frozen_fpr_%s" % label] = frozen_fpr
            row["frozen_tpr_%s" % label] = float(np.mean(watermarked >= tau))
        row["roc_tpr"] = interpolate_roc(result["roc"], grid).tolist()
        rows.append(row)
    return rows


_SHARED: Dict[str, Any] = {}


def _run(task) -> List[Dict[str, Any]]:
    setting, frame_id, trial = task
    return replicate_rows(_SHARED["cache"], _SHARED["planes"][frame_id], _SHARED["keys"][trial], setting,
                          frame_id=frame_id, key_trial=trial, cfg=_SHARED["cfg"])


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="embedding cache of the evaluation corpus")
    parser.add_argument("--key-file", type=Path, required=True, help="32-byte master key")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="CPU processes within this shard")
    args = parser.parse_args()
    cache = load_cache(args.cache)
    if cache.candidate_vectors.shape[1] < max(cfg.set_sizes) \
            or cache.candidate_vectors.shape[2] < max(item.n for item in cfg.settings):
        raise SystemExit("the cache holds too few slots or candidates for the configured B and N")
    _SHARED.update(cache=cache, cfg=cfg, planes=frame_planes(cfg, cache.candidate_vectors.shape[-1]),
                   keys=trial_keys(read_master_key(args.key_file), cfg))
    tasks = deal([(setting, frame_id, trial) for setting in cfg.settings
                  for frame_id, _, trial in replicate_cells(cfg)], args.shard_index, args.num_shards)
    rows: List[Dict[str, Any]] = []
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for result in pool.imap(_run, tasks):
                rows.extend(result)
    else:
        for task in tasks:
            rows.extend(_run(task))
            print("N=%d M=%d %s key %d" % (task[0].n, task[0].m, task[1], task[2]), flush=True)
    write_jsonl_rows(args.output_dir / ("replicates-shard-%03d-of-%03d.jsonl" % (args.shard_index, args.num_shards)),
                     rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
