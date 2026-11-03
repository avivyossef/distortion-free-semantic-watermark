#!/usr/bin/env python3
"""Plot the clean-detection ROC curves (appendix figures: linear and logarithmic FPR axes).

    python experiments/clean_detection/plot_clean_roc.py --mean-roc clean_detection/summary/mean_roc.json \
        --output-dir figures

Writes ``experiment2_detection_roc.pdf`` (linear FPR axis) and ``experiment2_detection_roc_log.pdf``
(logarithmic axis): one panel per set size B, one mean ROC curve per configuration (N, M),
averaged over the replicates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLORS = ("#0072B2", "#D55E00", "#7B3294", "#009E73", "#CC79A7", "#E69F00")


def render(data: dict, log_axis: bool, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    sizes = sorted({row["B"] for row in data["ours"]})
    settings = sorted({(row["N"], row["M"]) for row in data["ours"]})
    top = max(data["grid"])
    fig, axes = plt.subplots(1, len(sizes), figsize=(10.5, 3.7), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=.065, right=.985, top=.88, bottom=.30, wspace=.24)
    for ax, b in zip(axes, sizes):
        start = 1 if log_axis else 0
        x = data["grid"][start:]
        for row in data["ours"]:
            if row["B"] != b:
                continue
            color = COLORS[settings.index((row["N"], row["M"])) % len(COLORS)]
            ax.plot(x, row["mean"][start:], color=color, lw=1.6, label="Ours (N=%d, M=%d)" % (row["N"], row["M"]))
        ax.plot(x, x, color="#888888", lw=.8, linestyle=":", label="Chance")
        if log_axis:
            ticks = [x[0], 1e-2, 5e-2, top]
            ax.set_xscale("log")
            ax.set_xlim(x[0], top)
            ax.set_xticks(ticks, ["%g" % tick for tick in ticks])
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ticks = [top * step / 4 for step in range(5)]
            ax.set_xlim(0, top)
            ax.set_xticks(ticks, ["%g" % tick for tick in ticks])
        ax.set_ylim(0, 1.035)
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
        ax.tick_params(labelsize=9, pad=4)
        ax.set_title("B = %d" % b, fontsize=11, pad=10)
        ax.grid(color="#e5e5e5", lw=.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("True-positive rate")
    fig.supxlabel("False-positive rate", y=.16, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.51, .01), ncol=len(labels), frameon=False,
               fontsize=9, columnspacing=1.5, handlelength=2.8)
    fig.savefig(str(output), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mean-roc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.mean_roc.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(data, False, args.output_dir / "experiment2_detection_roc.pdf")
    render(data, True, args.output_dir / "experiment2_detection_roc_log.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
