#!/usr/bin/env python3
"""Clean detection with one semantic encoder (appendix: semantic-encoder ablation).

The watermark is rerun unchanged except for the encoder: the same candidate images are embedded
with the encoder (``experiments/generation/embed_corpus.py --encoder NAME --slots 1``),
hyperplanes are drawn in the encoder's native dimension, and candidate selection, boundary
calibration and clean-image detection run as in ``experiments/clean_detection`` with the
single-image setting. All replicates (configuration x frame x key) of ``configs/detection.yaml``
are evaluated; the ROC of each replicate is resampled to an FPR grid on [0, 1].

    python experiments/encoder_ablation/run_ablation.py --cache cache_dinov2_base.npz \
        --encoder dinov2_base --key-file keys/detector.key --output-dir ablation \
        --shard-index 0 --num-shards 1

Shard ``I`` writes ``ablation/<encoder>/replicates-shard-III-of-KKK.jsonl``. ``plot_ablation.py``
merges the encoders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))

from detection_corpus import (ENCODERS, deal, frame_planes, load_cache, read_master_key, replicate_cells,  # noqa: E402
                    trial_keys, write_jsonl_rows)

from watermark.calibration import clean_threshold  # noqa: E402
from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.detection import replicate_scores  # noqa: E402
from watermark.metrics import clean_detection_roc, interpolate_roc, tpr_at_fpr  # noqa: E402

ROC_GRID = np.linspace(0.0, 1.0, 1001)
SLOT = 1  # single-image detection: the first slot of every prompt


def target_label(fpr: float) -> str:
    return "%02d" % int(round(100 * fpr))


def ablation_row(cache, planes, key, setting, *, frame_id: str, key_trial: int, cfg: DetectionConfig
                 ) -> Dict[str, Any]:
    """Metrics of one (N, M, frame, key) replicate with the cache's encoder."""
    n, m = setting.n, setting.m
    result = replicate_scores(cache.candidate_vectors[:, SLOT - 1, :n], cache.clean_vectors[:, SLOT - 1],
                              cache.prompt_uids, planes, key, frame_id=frame_id, key_trial=key_trial, n=n, m=m,
                              activation=setting.activation, radius=setting.radius, folds=cfg.folds)
    roc = clean_detection_roc(result.clean.scores, result.watermarked.scores, cfg.fpr_targets)
    row: Dict[str, Any] = {"N": n, "M": m, "frame_id": frame_id, "key_trial": key_trial,
                           "sample_count": len(cache.prompt_uids), "dimension": int(planes.shape[1]),
                           "auc": roc["auc"], "delta_mean": float(np.mean(result.deltas))}
    for target in cfg.fpr_targets:
        label = target_label(target)
        tau, frozen_fpr = clean_threshold(result.clean.scores, target)
        row["tpr_roc_%s" % label] = tpr_at_fpr(roc["roc"], target)["tpr"]
        row["tpr_interp_%s" % label] = float(interpolate_roc(roc["roc"], [target])[0])
        row["frozen_tpr_%s" % label] = float(np.mean(result.watermarked.scores >= tau))
        row["frozen_fpr_%s" % label] = frozen_fpr
    row["roc_tpr"] = interpolate_roc(roc["roc"], ROC_GRID).tolist()
    return row


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="embedding cache computed with --encoder")
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cache = load_cache(args.cache)
    if cache.metadata.get("encoder") != args.encoder:
        raise SystemExit("the cache was computed with encoder %r" % cache.metadata.get("encoder"))
    planes = frame_planes(cfg, cache.candidate_vectors.shape[-1])
    keys = trial_keys(read_master_key(args.key_file), cfg)
    rows: List[Dict[str, Any]] = []
    for setting, (frame_id, _, trial) in deal([(s, c) for s in cfg.settings for c in replicate_cells(cfg)],
                                              args.shard_index, args.num_shards):
        rows.append(ablation_row(cache, planes[frame_id], keys[trial], setting, frame_id=frame_id,
                                 key_trial=trial, cfg=cfg))
        print("%s N=%d M=%d %s key %d" % (args.encoder, setting.n, setting.m, frame_id, trial), flush=True)
    write_jsonl_rows(args.output_dir / args.encoder / ("replicates-shard-%03d-of-%03d.jsonl" % (
        args.shard_index, args.num_shards)), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
