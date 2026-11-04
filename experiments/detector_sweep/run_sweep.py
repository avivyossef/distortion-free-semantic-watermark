#!/usr/bin/env python3
"""Sweep of the boundary threshold delta (as an activation target) and the Hamming radius.

The paper selects delta and r_ham for each (N, M) on a development set drawn from the COCO-2017
*training* captions, considering clean detection, detection after editing and the corresponding
false-positive rate. This script scores every grid point of ``sweep_config.yaml`` (HMAC keyed
score only) on the development corpus, replicate by replicate:

- selection and unedited scores: ``watermark.detection.replicate_scores`` (Gumbel selection;
  candidates, keys and frames as in the evaluation); delta of each activation target is cross-fitted on the
  watermarked margins;
- clean detection: AUROC, TPR from the ROC at each FPR target, and the TPR/FPR at the frozen
  clean threshold tau;
- detection after editing: with delta, radius and tau fixed on the unedited images, the edited
  TPR, edited FPR and paired retention, averaged over the 20 ordinary edits.

    python experiments/detector_sweep/run_sweep.py --cache dev_cache.npz --prompts dev_prompts.jsonl \
        --edits dev_edits --key-file keys/detector.key --output-dir sweep --shard-index 0 --num-shards 1

The development corpus is built like the evaluation corpus (``experiments/generation``,
``experiments/robustness/embed_ordinary_edits.py``) from a prompt list drawn from COCO-2017 *train*;
the number of prompts must make every activation target an integer count of the calibration
folds' images. Replicates are dealt round-robin to shards; shard ``I`` writes
``sweep-shard-III-of-KKK.jsonl`` and ``summarize_sweep.py`` merges them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "robustness"))

from _robustness_lib import SLOT, edited_scores, load_edit_embeddings, source_id, target_label  # noqa: E402
from detection_corpus import (deal, frame_planes, load_prompt_cache, read_master_key, replicate_cells,  # noqa: E402
                    trial_keys, write_jsonl_rows)

from watermark.calibration import clean_threshold, crossfit_deltas  # noqa: E402
from watermark.config import DetectionConfig, load_detection_config, parse_yaml  # noqa: E402
from watermark.detection import boundary_scores, replicate_scores  # noqa: E402
from watermark.metrics import auroc, interpolate_roc, paired_retention, roc_curve  # noqa: E402


def load_sweep_config(path: Path = HERE / "sweep_config.yaml") -> Dict[str, Any]:
    raw = parse_yaml(Path(path).read_text(encoding="utf-8"))
    targets, radii = raw["activation_targets"], raw["radii"]
    if not targets or not radii or any(not 0.0 <= float(a) <= 1.0 for a in targets) \
            or any(not isinstance(r, int) or r < 0 for r in radii):
        raise ValueError("invalid sweep config")
    return {"activation_targets": [float(a) for a in targets], "radii": [int(r) for r in radii]}


def sweep_rows(cache, planes, key, setting, edits, sweep, *, frame_id: str, key_trial: int,
               cfg: DetectionConfig) -> List[Dict[str, Any]]:
    """All grid points of one (N, M, frame, key) replicate."""
    n, m = setting.n, setting.m
    candidates = cache.candidate_vectors[:, SLOT - 1, :n]
    uids = cache.prompt_uids
    base = replicate_scores(candidates, cache.clean_vectors[:, SLOT - 1], uids, planes, key, frame_id=frame_id,
                            key_trial=key_trial, n=n, m=m, activation=setting.activation, radius=setting.radius,
                            folds=cfg.folds)
    wm_edit = edits.gather([source_id(uid, int(i) + 1) for uid, i in zip(uids, base.selected_index)])
    clean_edit = edits.gather([source_id(uid, None) for uid in uids])
    rows = []
    for activation in sweep["activation_targets"]:
        deltas = crossfit_deltas(base.selected_margins, activation, cfg.folds)
        for radius in sweep["radii"]:
            clean = boundary_scores(base.clean_code_ids, base.clean_margins, base.table, deltas, radius=radius)
            marked = boundary_scores(base.selected_code_ids, base.selected_margins, base.table, deltas, radius=radius)
            wm_after = edited_scores(wm_edit, planes, m, base.table, deltas, radius)
            clean_after = edited_scores(clean_edit, planes, m, base.table, deltas, radius)
            curve = roc_curve(clean.scores, marked.scores)
            row: Dict[str, Any] = {
                "N": n, "M": m, "frame_id": frame_id, "key_trial": key_trial, "activation": activation,
                "radius": radius, "auc": auroc(clean.scores, marked.scores),
                "realized_activation": float(np.mean(marked.neighbourhood_sizes > 1))}
            for target in cfg.fpr_targets:
                label = target_label(target)
                tau, _ = clean_threshold(clean.scores, target)
                retention = [paired_retention(marked.scores, wm_after[:, e], tau)["paired_detection_retention"]
                             for e in range(wm_after.shape[1])]
                finite = [v for v in retention if v == v]
                row["tpr_interp_%s" % label] = float(interpolate_roc(curve, [target])[0])
                row["frozen_tpr_%s" % label] = float(np.mean(marked.scores >= tau))
                row["frozen_fpr_%s" % label] = float(np.mean(clean.scores >= tau))
                row["edited_tpr_%s" % label] = float(np.mean(wm_after >= tau))
                row["edited_fpr_%s" % label] = float(np.mean(clean_after >= tau))
                row["retention_%s" % label] = float(np.mean(finite)) if finite else None
            rows.append(row)
    return rows


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="embedding cache of the development corpus")
    parser.add_argument("--prompts", type=Path, required=True, help="development prompt list")
    parser.add_argument("--edits", type=Path, required=True, help="embed_ordinary_edits.py shards of the development corpus")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, default=HERE / "sweep_config.yaml")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    sweep = load_sweep_config(args.sweep_config)
    cache = load_prompt_cache(args.cache, args.prompts)
    edits = load_edit_embeddings(args.edits)
    planes = frame_planes(cfg, cache.candidate_vectors.shape[-1])
    keys = trial_keys(read_master_key(args.key_file), cfg)
    rows: List[Dict[str, Any]] = []
    for setting, (frame_id, _, trial) in deal([(s, c) for s in cfg.settings for c in replicate_cells(cfg)],
                                              args.shard_index, args.num_shards):
        rows.extend(sweep_rows(cache, planes[frame_id], keys[trial], setting, edits, sweep, frame_id=frame_id,
                               key_trial=trial, cfg=cfg))
        print("N=%d M=%d %s key %d" % (setting.n, setting.m, frame_id, trial), flush=True)
    write_jsonl_rows(args.output_dir / ("sweep-shard-%03d-of-%03d.jsonl" % (args.shard_index, args.num_shards)), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
