#!/usr/bin/env python3
"""Generative-edit results: detection after editing, paired retention and edited FPR.

Our method: the per-replicate rows of ``score_ours.py`` are averaged over the (frame, key)
replicates of each configuration. Native baselines: the scores written by ``baselines/score.py``
for the list of ``prepare_baseline_scoring.py`` give one replicate each; tau is frozen from the
unedited clean calibration images at every FPR target of ``configs/detection.yaml`` and is not
recalibrated on edits. Metrics per edit category and for all edits pooled:

* ``edited_tpr``  fraction of edited watermarked images detected,
* ``retention``   fraction of originally detected watermarked images still detected after the edit,
* ``edited_fpr``  fraction of edited unwatermarked images detected,
* ``original_tpr`` and ``calibration_fpr`` before editing.

    python experiments/generative_edits/summarize.py --panel edits/panel.jsonl \
        --ours edits/ours/replicates.csv --baseline tree_ring=edits/tree_ring_scores.jsonl \
        --baseline gaussian_shading=edits/gaussian_shading_scores.jsonl \
        --baseline seal=edits/seal_scores.jsonl --output-dir edits/summary

Writes ``summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import average_replicates, category_metrics  # noqa: E402

from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402


def read_ours(path: Path):
    """Per-replicate rows of ``score_ours.py`` with numeric fields restored."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({**row, "fpr_target": float(row["fpr_target"]), "edits": int(row["edits"]),
                         **{name: float(row[name]) for name in
                            ("original_tpr", "edited_tpr", "retention", "edited_fpr", "calibration_fpr")}})
    return rows


def baseline_rows(method: str, score_rows, panel_rows, fpr_targets):
    """Metric rows of one native baseline from its score file (``image_id``/``score`` rows)."""
    scores = {row["image_id"]: float(row["score"]) for row in score_rows}
    uids = [row["prompt_uid"] for row in panel_rows]
    calibration = np.asarray([value for key, value in sorted(scores.items())
                              if key.startswith("calibration-clean:")], dtype=np.float64)
    if not len(calibration):
        raise ValueError("%s: no calibration-clean scores" % method)
    try:
        groups = {name: np.asarray([scores["%s:%s" % (name, uid)] for uid in uids], dtype=np.float64)
                  for name in ("original", "edited-watermarked", "edited-clean")}
    except KeyError as error:
        raise ValueError("%s: missing score for %s" % (method, error)) from error
    categories = [row["edit_category"] for row in panel_rows]
    rows = []
    for fpr in fpr_targets:
        metrics = category_metrics(calibration, categories, groups["original"], groups["edited-watermarked"],
                                   groups["edited-clean"], fpr)
        for category, values in metrics.items():
            rows.append({"method": method, "fpr_target": fpr, "category": category, **values})
    return rows


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ours", type=Path, help="replicates.csv of score_ours.py")
    parser.add_argument("--baseline", action="append", default=[], metavar="METHOD=SCORES")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel_rows = read_jsonl(args.panel)
    rows = list(read_ours(args.ours)) if args.ours else []
    for item in args.baseline:
        method, _, path = item.partition("=")
        rows.extend(baseline_rows(method, read_jsonl(Path(path)), panel_rows, cfg.fpr_targets))
    summary = average_replicates(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
