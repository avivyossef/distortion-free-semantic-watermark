#!/usr/bin/env python3
"""Distinct hash codes under perturbations and across independent generations (appendix figures).

Reconstruction from the paper's description. A group is a set of 16 embeddings: the 16
perturbed versions of one source image (ordered by increasing severity) or the 16
same-prompt generations of one prompt (in seed order). Each embedding is hashed to an
8-bit random-hyperplane code with one of 200 hyperplane draws; the same draw (same seed) is used
for every group and every encoder. For ``k = 1 .. 16`` the curve is the number of distinct
codes among the first ``k`` members, averaged over groups and draws.

    python experiments/encoder_comparison/distinct_codes.py perturbations \
        --embeddings-dir results/stability/embeddings --output-dir results/unique_codes_perturbations
    python experiments/encoder_comparison/distinct_codes.py generations \
        --panel "SD2.1=embeddings/sd21" --panel "FLUX distilled=embeddings/flux_distilled" \
        --panel "FLUX base=embeddings/flux_base" --output-dir results/unique_codes_generations
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison_lib import (  # noqa: E402
    ENCODER_COLORS, ENCODER_LABELS, ENCODERS, FAMILIES, FAMILY_LABELS, GROUP_SIZE, PROMPT_COUNT,
)
from semantic_entropy import load_embeddings  # noqa: E402
from watermark.coding import bits_to_code_ids, sample_hyperplanes  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402

CODE_BITS = 8
DRAWS = 200
DRAW_SEED_OFFSET = 1000  # hyperplane draw t uses seed frame_seed_base + DRAW_SEED_OFFSET + t


def draw_seeds(frame_seed_base: int) -> List[int]:
    """Seeds of the 200 hyperplane draws (disjoint from the detector's hyperplane frames)."""
    return [frame_seed_base + DRAW_SEED_OFFSET + draw for draw in range(DRAWS)]


def distinct_code_curve(embeddings: np.ndarray, seeds: List[int], *, bits: int = CODE_BITS) -> np.ndarray:
    """Mean number of distinct codes among the first ``k`` members, ``k = 1 .. K``.

    ``embeddings`` is ``(groups, K, d)``; the mean runs over groups and over the hyperplane
    draws given by ``seeds``.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 3 or not seeds:
        raise ValueError("expected (groups, members, dimension) embeddings and at least one seed")
    groups, members, dimension = values.shape
    total = np.zeros(members, dtype=np.float64)
    for seed in seeds:
        planes = sample_hyperplanes(seed, dimension)[:bits]
        with np.errstate(all="ignore"):
            ids = bits_to_code_ids(values @ planes.T >= 0.0)
        for k in range(1, members + 1):
            ordered = np.sort(ids[:, :k], axis=1)
            total[k - 1] += (1 + np.count_nonzero(np.diff(ordered, axis=1), axis=1)).mean()
    return total / len(seeds)


def _write_curves(path: Path, curves: Dict[Tuple[str, str], np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "encoder", "included", "mean_distinct_codes"])
        for (panel, encoder), values in curves.items():
            writer.writerows((panel, encoder, k + 1, float(value)) for k, value in enumerate(values))


def _plot(path: Path, panels: List[str], curves: Dict[Tuple[str, str], np.ndarray], *, xlabel: str,
          ylabel: str, upper_bound: bool) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), sharex=True, sharey=upper_bound,
                                squeeze=False)
    x = np.arange(1, GROUP_SIZE + 1)
    for axis, panel in zip(axes[0], panels):
        for encoder in ENCODERS:
            axis.plot(x, curves[(panel, encoder)], color=ENCODER_COLORS[encoder], lw=2, label=ENCODER_LABELS[encoder])
        if upper_bound:
            axis.plot(x, x, color="#555555", linestyle=":", lw=1.2)
        axis.set_title(panel, fontsize=11, fontweight="bold")
        axis.set_xlabel(xlabel, fontsize=10)
        axis.set_xlim(1, GROUP_SIZE)
        axis.grid(alpha=.18, lw=.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(ylabel, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(ENCODERS), frameon=False, fontsize=9,
                  bbox_to_anchor=(.5, 1.04))
    figure.tight_layout(rect=(0, 0, 1, .9))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def perturbations(args: argparse.Namespace) -> None:
    seeds = draw_seeds(load_detection_config().frame_seed_base)
    arrays = {}
    for encoder in ENCODERS:
        with np.load(args.embeddings_dir / ("%s.npz" % encoder), allow_pickle=False) as data:
            arrays[encoder] = {family: data[family] for family in FAMILIES}
    curves = {(FAMILY_LABELS[family], encoder): distinct_code_curve(arrays[encoder][family], seeds)
              for family in FAMILIES for encoder in ENCODERS}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_curves(args.output_dir / "curves.csv", curves)
    _plot(args.output_dir / "unique_codes_perturbations.pdf", [FAMILY_LABELS[f] for f in FAMILIES], curves,
          xlabel="Perturbed versions included", ylabel="Distinct %d-bit codes" % CODE_BITS, upper_bound=False)


def generations(args: argparse.Namespace) -> None:
    seeds = draw_seeds(load_detection_config().frame_seed_base)
    names, curves = [], {}
    for item in args.panel:
        name, _, directory = item.partition("=")
        _, arrays = load_embeddings(Path(directory))
        names.append(name)
        for encoder in ENCODERS:
            curves[(name, encoder)] = distinct_code_curve(arrays[encoder], seeds)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_curves(args.output_dir / "curves.csv", curves)
    _plot(args.output_dir / "unique_codes_generations.pdf", names, curves, xlabel="Independent generations included",
          ylabel="Distinct %d-bit codes" % CODE_BITS, upper_bound=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("perturbations")
    first.add_argument("--embeddings-dir", type=Path, required=True, help="stability.py output: embeddings/")
    first.add_argument("--output-dir", type=Path, required=True)
    first.set_defaults(run=perturbations)
    second = commands.add_parser("generations")
    second.add_argument("--panel", action="append", required=True, metavar="NAME=DIR",
                        help="generator name and its embeddings directory (embed_generations.py outputs)")
    second.add_argument("--output-dir", type=Path, required=True)
    second.set_defaults(run=generations)
    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
