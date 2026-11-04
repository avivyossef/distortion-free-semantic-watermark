#!/usr/bin/env python3
"""Clean detection and detection after edits under the three distortion-free sampling rules.

``representative`` is representative sampling (the appendix alternative to the paper's Gumbel
selection); ``gumbel`` is Gumbel selection, the method of every other experiment; ``tournament`` is the
code-level non-distortionary Tournament adaptation (``watermark.selection_rules``). Every rule replays
the same candidate pools, embeddings, hyperplane frames and keys, the same per-(N, M) detector
settings and the same clean-image threshold calibration, with the rule's own keyed score
(single-image setting). Only candidate selection (and the tournament's per-code statistic)
changes.

    python experiments/sampling_rules/run_rules.py --cache cache.npz --key-file keys/detector.key \
        --output-dir rules --shard-index 0 --num-shards 1
    # optional, for the detection-after-editing comparison (embed_ordinary_edits.py --compare-rules)
    python experiments/sampling_rules/run_rules.py --cache cache.npz --prompts prompts200.jsonl \
        --edits edits --key-file keys/detector.key --output-dir rules

Without ``--edits`` the script scores clean detection on all prompts of the cache (rows
``clean-shard-III-of-KKK.jsonl``); with ``--edits`` it scores the 20 ordinary edits on the prompts
of ``--prompts`` (rows ``edits-shard-III-of-KKK.jsonl``, edited TPR and paired retention at each
FPR target). Clean detection draws the selector as the set-size experiments do (identity ``<uid>:slot:1``),
the edit comparison as the robustness experiments do (bare prompt identity). Replicates (rule x configuration x frame x key) are dealt round-robin to shards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robustness"))

from _robustness_lib import SLOT, load_edit_embeddings, replicate_edit_rows, target_label  # noqa: E402
from detection_corpus import (deal, frame_planes, load_cache, load_prompt_cache, read_master_key,  # noqa: E402
                    replicate_cells, trial_keys, write_jsonl_rows)

from watermark.calibration import clean_threshold  # noqa: E402
from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.detection import replicate_scores  # noqa: E402
from watermark.metrics import clean_detection_roc, interpolate_roc, tpr_at_fpr  # noqa: E402
from watermark.selection_rules import RULES  # noqa: E402


def _replicate(cache, planes, key, setting, rule, *, frame_id, key_trial, cfg, slot):
    """One replicate under ``rule``: ``slot=SLOT`` is the clean-detection stream, ``None`` the edit stream."""
    n = setting.n
    return replicate_scores(
        cache.candidate_vectors[:, SLOT - 1, :n], cache.clean_vectors[:, SLOT - 1], cache.prompt_uids, planes, key,
        frame_id=frame_id, key_trial=key_trial, n=n, m=setting.m, activation=setting.activation,
        radius=setting.radius, folds=cfg.folds, rule=rule, slot=slot)


def clean_row(cache, planes, key, setting, rule, *, frame_id: str, key_trial: int, cfg: DetectionConfig
              ) -> Dict[str, Any]:
    """Clean-detection metrics of one (rule, N, M, frame, key) replicate."""
    result = _replicate(cache, planes, key, setting, rule, frame_id=frame_id, key_trial=key_trial, cfg=cfg,
                        slot=SLOT)
    roc = clean_detection_roc(result.clean.scores, result.watermarked.scores, cfg.fpr_targets)
    row: Dict[str, Any] = {"rule": rule, "N": setting.n, "M": setting.m, "frame_id": frame_id,
                           "key_trial": key_trial, "sample_count": len(cache.prompt_uids), "auc": roc["auc"],
                           "mean_distinct_codes": float(np.mean(result.distinct_codes))}
    for target in cfg.fpr_targets:
        label = target_label(target)
        tau, _ = clean_threshold(result.clean.scores, target)
        row["tpr_roc_%s" % label] = tpr_at_fpr(roc["roc"], target)["tpr"]
        row["tpr_interp_%s" % label] = float(interpolate_roc(roc["roc"], [target])[0])
        row["frozen_tpr_%s" % label] = float(np.mean(result.watermarked.scores >= tau))
    return row


def edit_rows(cache, planes, key, setting, rule, edits, *, frame_id: str, key_trial: int, cfg: DetectionConfig
              ) -> List[Dict[str, Any]]:
    """Edited TPR / FPR / retention rows of one replicate (boundary-aware detector, ``rule``)."""
    result = _replicate(cache, planes, key, setting, rule, frame_id=frame_id, key_trial=key_trial, cfg=cfg,
                        slot=None)  # the selections that embed_ordinary_edits.py --compare-rules edited
    rows = replicate_edit_rows(result, planes, candidates=cache.candidate_vectors[:, SLOT - 1, :setting.n],
                               edits=edits, prompt_uids=cache.prompt_uids, n=setting.n, m=setting.m,
                               radius=setting.radius, fpr_targets=cfg.fpr_targets, detector="boundary")
    return [{**row, "rule": rule, "frame_id": frame_id, "key_trial": key_trial} for row in rows]


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, help="prompt list of the edit evaluation (with --edits)")
    parser.add_argument("--edits", type=Path, help="embed_ordinary_edits.py shards covering every rule's selections")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if (args.edits is None) != (args.prompts is None):
        raise SystemExit("--edits and --prompts go together")
    cache = load_cache(args.cache) if args.edits is None else load_prompt_cache(args.cache, args.prompts)
    edits = None if args.edits is None else load_edit_embeddings(args.edits)
    planes = frame_planes(cfg, cache.candidate_vectors.shape[-1])
    keys = trial_keys(read_master_key(args.key_file), cfg)
    cells = [(rule, s, c) for rule in RULES for s in cfg.settings for c in replicate_cells(cfg)]
    rows: List[Dict[str, Any]] = []
    for rule, setting, (frame_id, _, trial) in deal(cells, args.shard_index, args.num_shards):
        if edits is None:
            rows.append(clean_row(cache, planes[frame_id], keys[trial], setting, rule, frame_id=frame_id,
                                  key_trial=trial, cfg=cfg))
        else:
            rows.extend(edit_rows(cache, planes[frame_id], keys[trial], setting, rule, edits, frame_id=frame_id,
                                  key_trial=trial, cfg=cfg))
        print("%s N=%d M=%d %s key %d" % (rule, setting.n, setting.m, frame_id, trial), flush=True)
    name = "clean" if edits is None else "edits"
    write_jsonl_rows(args.output_dir / ("%s-shard-%03d-of-%03d.jsonl" % (name, args.shard_index, args.num_shards)),
                     rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
