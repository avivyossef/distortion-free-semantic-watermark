#!/usr/bin/env python3
"""Merge the clean-detection shards into the Table 2 rows, operating points and mean ROC curves.

    python experiments/clean_detection/summarize_clean_detection.py \
        --shards-dir clean_detection --output-dir clean_detection/summary

Writes, for ours (the native baselines come from ``experiments/baselines``):

- ``summary.csv``: one row per (N, M, B), mean and standard deviation over the replicates of the
  AUROC and, per FPR target, the TPR read from the replicate ROC curves (``tpr_interp``, the mean
  ROC curve at the target; ``tpr_roc``, the best ROC vertex not above the target) and at the
  frozen clean threshold (``frozen_tpr`` with its achieved ``frozen_fpr``).
- ``detection_table_ours.tex``: the rows of ours in the clean-detection table (AUROC and TPR at the
  largest FPR target for each set size B).
- ``operating_points_ours.tex``: the rows of ours in the clean operating-points table (TPR in
  percent at each FPR target, read from the mean ROC curve).
- ``mean_roc.json``: the mean ROC curve per (N, M, B) on a grid from 0 to the largest FPR target
  (input of ``plot_clean_roc.py``).

Fails unless every configuration x frame x key x set size replicate is present exactly once.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_corpus import replicate_cells, write_csv_rows  # noqa: E402
from run_clean_detection import roc_grid, target_label  # noqa: E402

from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402


def load_rows(shards_dir: Path, cfg: DetectionConfig) -> List[Dict[str, Any]]:
    """All replicate rows, validated to cover the complete replicate grid exactly once."""
    rows = [row for path in sorted(Path(shards_dir).glob("replicates-shard-*.jsonl")) for row in read_jsonl(path)]
    expected = {(s.n, s.m, b, frame_id, trial) for s in cfg.settings for b in cfg.set_sizes
                for frame_id, _, trial in replicate_cells(cfg)}
    seen = [(r["N"], r["M"], r["B"], r["frame_id"], r["key_trial"]) for r in rows]
    if len(set(seen)) != len(seen) or set(seen) != expected:
        raise ValueError("shards do not cover the replicate grid exactly once")
    return rows


def _mean_sd(values: List[float]) -> "tuple[float, float]":
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def summarize(rows: List[Dict[str, Any]], cfg: DetectionConfig) -> "tuple[list[dict], dict]":
    """Per (N, M, B) means over replicates, and the mean ROC curves."""
    summary, curves = [], []
    scalar = ["auc", "delta_mean", "clean_activation_rate", "watermarked_activation_rate"]
    for target in cfg.fpr_targets:
        label = target_label(target)
        scalar += ["tpr_interp_%s" % label, "tpr_roc_%s" % label, "frozen_tpr_%s" % label,
                   "frozen_fpr_%s" % label]
    for setting in cfg.settings:
        for b in cfg.set_sizes:
            members = [r for r in rows if (r["N"], r["M"], r["B"]) == (setting.n, setting.m, b)]
            entry: Dict[str, Any] = {"N": setting.n, "M": setting.m, "B": b, "replicate_count": len(members),
                                     "sample_count": members[0]["sample_count"]}
            for field in scalar:
                entry[field + "_mean"], entry[field + "_sd"] = _mean_sd([float(r[field]) for r in members])
            summary.append(entry)
            curves.append({"N": setting.n, "M": setting.m, "B": b, "replicates": len(members),
                           "sample_count": members[0]["sample_count"],
                           "mean": np.mean([r["roc_tpr"] for r in members], axis=0).tolist()})
    return summary, {"grid": roc_grid(cfg).tolist(), "ours": curves}


def operating_points_rows(summary: List[Dict[str, Any]], cfg: DetectionConfig) -> List[str]:
    """LaTeX rows of the clean operating-points table (ours): TPR (%) at each FPR target."""
    lines = []
    for index, setting in enumerate(cfg.settings):
        if index:
            lines.append("\\addlinespace")
        first = True
        for b in cfg.set_sizes:
            entry = next(r for r in summary if (r["N"], r["M"], r["B"]) == (setting.n, setting.m, b))
            values = " & ".join("%.1f" % (100 * entry["tpr_interp_%s_mean" % target_label(t)])
                                for t in cfg.fpr_targets)
            name = "Ours $(N{=}%d,M{=}%d)$" % (setting.n, setting.m) if first else ""
            lines.append("%s & %d & %s \\\\" % (name, b, values))
            first = False
    return lines


def detection_table_rows(summary: List[Dict[str, Any]], cfg: DetectionConfig) -> List[str]:
    """LaTeX rows of the clean-detection table (ours): AUROC and TPR at the largest FPR target per B."""
    label = target_label(max(cfg.fpr_targets))
    lines = []
    for setting in cfg.settings:
        cells = []
        for b in cfg.set_sizes:
            entry = next(r for r in summary if (r["N"], r["M"], r["B"]) == (setting.n, setting.m, b))
            cells += ["%.3f" % entry["auc_mean"], "%.3f" % entry["tpr_interp_%s_mean" % label]]
        lines.append("Ours $(N=%d,M=%d)$ & %s \\\\" % (setting.n, setting.m, " & ".join(cells)))
    return lines


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, mean_roc = summarize(load_rows(args.shards_dir, cfg), cfg)
    write_csv_rows(args.output_dir / "summary.csv", summary)
    (args.output_dir / "detection_table_ours.tex").write_text(
        "\n".join(detection_table_rows(summary, cfg)) + "\n", encoding="utf-8")
    (args.output_dir / "operating_points_ours.tex").write_text(
        "\n".join(operating_points_rows(summary, cfg)) + "\n", encoding="utf-8")
    (args.output_dir / "mean_roc.json").write_text(json.dumps(mean_roc, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
