#!/usr/bin/env python3
"""Detection after the 20 ordinary edits: edited TPR, edited FPR and paired retention.

Protocol (paper, Detection and Robustness Protocols): B = 1 on the prompts of ``--prompts``;
the same edit is applied to the watermarked and the unwatermarked image; delta, the Hamming
radius and the score threshold tau are fixed on the unedited images (tau at each FPR target of
``configs/detection.yaml``, from clean scores only) and are not recalibrated on edited images.
Every replicate (configuration x frame x key) is evaluated with the boundary-aware detector
and, for the detector ablation, with exact-code detection (radius 0) on the same images.

    python experiments/robustness/analyze_robustness.py --cache cache.npz --prompts prompts200.jsonl \
        --edits edits --key-file keys/detector.key --output-dir robustness \
        --shard-index 0 --num-shards 1

Shard ``I`` writes ``replicates-shard-III-of-KKK.jsonl`` with one row per
(replicate, detector, edit or ``identity``, FPR target); ``summarize_robustness.py`` merges them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _robustness_lib import SLOT, load_edit_embeddings, replicate_edit_rows  # noqa: E402
from detection_corpus import (deal, frame_planes, load_prompt_cache, read_master_key, replicate_cells,  # noqa: E402
                    trial_keys, write_jsonl_rows)

from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.detection import replicate_scores  # noqa: E402


def analyze_replicate(cache, planes, key, setting, edits, *, frame_id: str, key_trial: int,
                      cfg: DetectionConfig) -> List[Dict[str, Any]]:
    """Rows of one (configuration, frame, key) replicate for both detectors."""
    n, m = setting.n, setting.m
    candidates = cache.candidate_vectors[:, SLOT - 1, :n]
    result = replicate_scores(candidates, cache.clean_vectors[:, SLOT - 1], cache.prompt_uids, planes, key,
                              frame_id=frame_id, key_trial=key_trial, n=n, m=m, activation=setting.activation,
                              radius=setting.radius, folds=cfg.folds, slot=SLOT)
    rows = []
    for detector, radius in (("boundary", setting.radius), ("exact", 0)):
        for row in replicate_edit_rows(result, planes, candidates=candidates, edits=edits,
                                       prompt_uids=cache.prompt_uids, n=n, m=m, radius=radius,
                                       fpr_targets=cfg.fpr_targets, detector=detector):
            rows.append({**row, "frame_id": frame_id, "key_trial": key_trial})
    return rows


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True, help="directory of embed_ordinary_edits.py shards")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cache = load_prompt_cache(args.cache, args.prompts)
    edits = load_edit_embeddings(args.edits)
    planes = frame_planes(cfg, cache.candidate_vectors.shape[-1])
    keys = trial_keys(read_master_key(args.key_file), cfg)
    rows: List[Dict[str, Any]] = []
    for setting, (frame_id, _, trial) in deal([(s, c) for s in cfg.settings for c in replicate_cells(cfg)],
                                              args.shard_index, args.num_shards):
        rows.extend(analyze_replicate(cache, planes[frame_id], keys[trial], setting, edits,
                                      frame_id=frame_id, key_trial=trial, cfg=cfg))
        print("N=%d M=%d %s key %d" % (setting.n, setting.m, frame_id, trial), flush=True)
    write_jsonl_rows(args.output_dir / ("replicates-shard-%03d-of-%03d.jsonl" % (args.shard_index, args.num_shards)),
                     rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
