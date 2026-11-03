#!/usr/bin/env python3
"""Same-prompt cosine and fixed-radius semantic entropy of the five encoders for one generator.

``--embeddings-dir`` holds one ``<encoder>.npz`` per encoder (``embed_generations.py``), all
over the same 200 prompts. Pooling the 120 within-prompt cosine distances of every prompt, the
estimate of the paper's semantic entropy at radius epsilon is the negative natural logarithm
of the fraction of pairs at distance at most epsilon.

Writes ``curves.csv`` (every epsilon on a 401-point grid over [0, 2]) and ``table.csv`` (mean pairwise cosine similarity and the entropy at the
table radius, one row per encoder).

    python experiments/encoder_comparison/semantic_entropy.py \
        --embeddings-dir embeddings/flux_base --output-dir results/flux_base
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_lib import (  # noqa: E402
    ENCODERS, EPSILON_GRID, GROUP_SIZE, PROMPT_COUNT, TABLE_EPSILON, estimate, mean_pairwise_cosine,
    pair_distances,
)


def load_embeddings(directory: Path):
    """``(prompt_uids, {encoder: (200, 16, d) array})``, checking that the prompts agree."""
    prompt_ids, arrays = None, {}
    for encoder in ENCODERS:
        with np.load(Path(directory) / ("%s.npz" % encoder), allow_pickle=False) as data:
            x, ids = data["embeddings"], data["prompt_uids"].tolist()
        if x.ndim != 3 or x.shape[:2] != (PROMPT_COUNT, GROUP_SIZE) or len(ids) != PROMPT_COUNT:
            raise ValueError("invalid input geometry: %s" % encoder)
        if prompt_ids is None:
            prompt_ids = ids
        if ids != prompt_ids or len(set(ids)) != len(ids):
            raise ValueError("prompt identities differ or contain duplicates")
        arrays[encoder] = x
    return prompt_ids, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prompt_ids, arrays = load_embeddings(args.embeddings_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    table = []
    with (args.output_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["encoder", "prompts", "epsilon", "p", "entropy"])
        for encoder in ENCODERS:
            distances = pair_distances(arrays[encoder])
            p, h = estimate(distances, EPSILON_GRID)
            writer.writerows((encoder, PROMPT_COUNT, *row) for row in zip(EPSILON_GRID, p, h))
            _, at_table_radius = estimate(distances, np.array([TABLE_EPSILON]))
            table.append({"encoder": encoder, "mean_cosine": mean_pairwise_cosine(arrays[encoder]),
                          "entropy_at_table_radius": float(at_table_radius[0])})
    with (args.output_dir / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["encoder", "mean_cosine", "entropy_at_table_radius"])
        writer.writeheader()
        writer.writerows(table)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "table_epsilon": TABLE_EPSILON, "grid_points": len(EPSILON_GRID),
        "pairs_per_prompt": GROUP_SIZE * (GROUP_SIZE - 1) // 2,
        "prompt_uids_sha256": hashlib.sha256("\n".join(prompt_ids).encode("utf-8")).hexdigest(),
        "curves_sha256": hashlib.sha256((args.output_dir / "curves.csv").read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
