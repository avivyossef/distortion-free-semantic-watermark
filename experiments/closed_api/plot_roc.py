#!/usr/bin/env python3
"""Plot the mean closed-API ROC curves with one-standard-deviation bands (appendix figure).

Reads ``mean_roc.csv`` written by ``evaluate.py``.

    python experiments/closed_api/plot_roc.py --mean-roc results/closed_api/mean_roc.csv \
        --output figures/closed_api_gpt_image_roc.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def roc_series(rows) -> list:
    by_b = defaultdict(list)
    for row in rows:
        by_b[int(row["B"])].append(row)
    if not by_b:
        raise ValueError("no ROC rows")
    return [{
        "B": b, "mean_auc": float(points[0]["mean_auc"]),
        "fpr": [float(point["fpr"]) for point in points],
        "mean_tpr": [float(point["mean_tpr"]) for point in points],
        "std_tpr": [float(point["std_tpr"]) for point in points],
    } for b, points in sorted(by_b.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mean-roc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.mean_roc.open(newline="") as handle:
        curves = roc_series(list(csv.DictReader(handle)))

    import matplotlib.pyplot as plt
    import numpy as np

    colors = ("#2563EB", "#EA580C", "#16A34A", "#9333EA")
    fig, axis = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    for color, curve in zip(colors, curves):
        fpr = np.asarray(curve["fpr"])
        mean = np.asarray(curve["mean_tpr"])
        std = np.asarray(curve["std_tpr"])
        axis.plot(fpr, mean, color=color, linewidth=2.2, label=f"B={curve['B']} (mean AUROC={curve['mean_auc']:.3f})")
        axis.fill_between(fpr, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0),
                          color=color, alpha=0.14, linewidth=0)
    axis.plot((0, 1), (0, 1), color="#6B7280", linestyle="--", linewidth=1.2, label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
