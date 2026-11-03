#!/usr/bin/env python3
"""Render the 200-prompt fixed-radius entropy curves of the three generators (appendix figure).

Inputs are the ``curves.csv`` files written by ``semantic_entropy.py``, one per generator.

    python experiments/encoder_comparison/plot_semantic_entropy_epsilon.py \
        --sd21 results/sd21/curves.csv --flux-distilled results/flux_distilled/curves.csv \
        --flux-base results/flux_base/curves.csv --output figures/semantic_entropy_epsilon.pdf
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_lib import TABLE_EPSILON  # noqa: E402

ENCODERS = {"ours": ("Ours", "#b63836"), "dinov2_base": ("DINOv2-base", "#2376a5"), "clip_vit_b32": ("CLIP ViT-B/32", "#9a6a16"), "dinov3_l16": ("DINOv3 L/16", "#528248"), "tpips_embed": ("TPIPS-Embed", "#8059a0")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sd21", required=True, type=Path)
    p.add_argument("--flux-distilled", required=True, type=Path)
    p.add_argument("--flux-base", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    panels = (("SD2.1", args.sd21), ("FLUX distilled", args.flux_distilled), ("FLUX base", args.flux_base))
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.8), sharex=True, sharey=True)
    common_grid = None
    for ax, (title, path) in zip(axes, panels):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        for key, (label, color) in ENCODERS.items():
            selected = [r for r in rows if r["encoder"] == key]
            if len(selected) != 401 or any(int(r["prompts"]) != 200 for r in selected):
                raise ValueError(f"incomplete 200-prompt curve: {title} / {key}")
            a = np.array([[float(r[k]) for k in ("epsilon", "entropy")] for r in selected])
            x, h = a.T
            if common_grid is None:
                common_grid = x
            if not np.array_equal(x, common_grid) or x[0] != 0 or x[-1] != 2 or np.any(np.diff(x) <= 0):
                raise ValueError("epsilon grids differ")
            finite_h = np.where(np.isfinite(h), h, np.nan)
            ax.plot(x, finite_h, color=color, lw=2, label=label)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.axvline(TABLE_EPSILON, color="#555555", linestyle="--", linewidth=1)
        ax.set_xlabel(r"Cosine-distance radius $\varepsilon$", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(alpha=.18, lw=.5)
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Estimated entropy $\widehat{H}_{\mathrm{sem}}^{E,\varepsilon}$", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .84))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
