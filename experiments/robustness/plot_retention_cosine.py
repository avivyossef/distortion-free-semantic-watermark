#!/usr/bin/env python3
"""Encoder similarity versus watermark retention (appendix figure) and Spearman correlations.

Each marker pairs, for one method and one of the 20 edits, the mean original-edited cosine
similarity in our encoder's space with the paired detection retention; the Spearman rank
correlation over the 20 edits is computed per method. Inputs are ``edit_results`` CSVs (see
``render_robustness.py``): ours from ``summarize_robustness.py`` (``--results``; its ``cosine``
column), the baselines' retention from ``experiments/baselines/robustness.py``
(``--baseline-per-edit NAME=per_edit.csv``) with their cosines from ``edit_cosines.py``
(``--cosines``).

    python experiments/robustness/plot_retention_cosine.py \
        --results robustness/summary/edit_results.csv \
        --baseline-per-edit "Tree-Ring=analysis/tree_ring_robustness/per_edit.csv" \
        --cosines analysis/tree_ring_cosines.csv --output-dir figures

Writes ``experiment4_retention_cosine.pdf`` and ``retention_cosine_correlations.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _robustness_lib import EDIT_IDS  # noqa: E402
from detection_corpus import write_csv_rows  # noqa: E402
from render_robustness import methods_in_order, read_baseline_per_edit, read_results  # noqa: E402

from watermark.config import load_detection_config  # noqa: E402

from watermark.attacks import EDITS  # noqa: E402

CATEGORIES = {
    "Everyday edits": ("o", {"brightness", "contrast", "gaussian_blur", "jpeg", "median"}),
    "Geometric edits": ("^", {"crop", "offcenter_crop", "translation"}),
    "Added noise": ("X", {"gaussian_noise", "salt_pepper"}),
}
COLORS = ("#0072B2", "#D55E00", "#7B4DA6", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def edit_points(rows: Sequence[Dict[str, Any]], cosines: Dict[tuple, float]) -> Dict[str, List[Dict[str, Any]]]:
    """Per method, one point per edit: ``cosine`` and ``retention`` (rows without both are dropped)."""
    points: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row["edit_id"] not in EDIT_IDS:
            continue
        cosine = cosines.get((row["method"], row["edit_id"]), row.get("cosine"))
        if cosine is None or row["retention"] is None or row["retention"] != row["retention"]:
            continue
        points.setdefault(row["method"], []).append(
            {"edit_id": row["edit_id"], "family": EDITS[row["edit_id"]].family, "cosine": cosine,
             "retention": row["retention"]})
    return points


def correlations(points: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Spearman rank correlation between cosine and retention over the edits of each method."""
    from scipy.stats import spearmanr

    return [{"method": method, "edit_count": len(items),
             "spearman_rho": float(spearmanr([p["cosine"] for p in items], [p["retention"] for p in items]).statistic)}
            for method, items in points.items()]


def plot(points: Dict[str, List[Dict[str, Any]]], rho: Dict[str, float], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    methods = list(points)
    colors = {method: COLORS[i % len(COLORS)] for i, method in enumerate(methods)}
    panels = ((True, [m for m in methods if m.startswith("Ours")], "(a) Our method"),
              (False, [m for m in methods if not m.startswith("Ours")], "(b) Baselines"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.99, bottom=.27, top=.79, wspace=.09)
    for ax, (_, names, title) in zip(axes, panels):
        handles = []
        for method in names:
            for _, (marker, families) in CATEGORIES.items():
                chosen = [p for p in points[method] if p["family"] in families]
                ax.scatter([p["cosine"] for p in chosen], [100 * p["retention"] for p in chosen], s=30, marker=marker,
                           color=colors[method], edgecolor="white", linewidth=.4, alpha=.85, zorder=3)
            handles.append(Patch(facecolor=colors[method],
                                 label=method.replace("N=", "N = ").replace("M=", "M = ")
                                 + r"  ($\rho_s=%.4f$)" % rho[method]))
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_ylim(-4, 104)
        ax.grid(color="#e5e5e5", linewidth=.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -.22), frameon=False, fontsize=9,
                  ncol=1 if len(handles) <= 3 else 2, handlelength=.9, columnspacing=.9, borderaxespad=0)
        ax.set_xlabel("Original–edited encoder cosine similarity", labelpad=8)
    axes[0].set_ylabel("Paired watermark retention (%)")
    shapes = [Line2D([], [], marker=marker, color="#444444", linestyle="None", markersize=7, label=label)
              for label, (marker, _) in CATEGORIES.items()]
    fig.legend(handles=shapes, loc="upper center", bbox_to_anchor=(.53, .965), ncol=3, frameon=False,
               columnspacing=2.5, handletextpad=.5)
    fig.text(.53, .87, "Each marker is one edit setting; color identifies the method.", ha="center", fontsize=10,
             color="#444444")
    fig.text(.08, .025, r"$\rho_s$: Spearman rank correlation across the edit settings (all categories combined).",
             fontsize=10, color="#333333")
    fig.savefig(str(output), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = load_detection_config()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-per-edit", action="append", default=[], metavar="NAME=PER_EDIT_CSV")
    parser.add_argument("--cosines", type=Path, nargs="*", default=[])
    parser.add_argument("--fpr", type=float, default=max(cfg.fpr_targets))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = []
    for path in args.results:
        rows.extend(read_results(path))
    for item in args.baseline_per_edit:
        name, _, path = item.partition("=")
        rows.extend(read_baseline_per_edit(name, Path(path), args.fpr))
    cosines: Dict[tuple, float] = {}
    for path in args.cosines:
        with path.open(newline="", encoding="utf-8") as handle:
            cosines.update({(r["method"], r["edit_id"]): float(r["cosine"]) for r in csv.DictReader(handle)})
    points = edit_points(rows, cosines)
    table = correlations(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "retention_cosine_correlations.csv", table)
    ordered = {m: points[m] for m in methods_in_order(rows) if m in points}
    plot(ordered, {row["method"]: row["spearman_rho"] for row in table}, args.output_dir / "experiment4_retention_cosine.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
