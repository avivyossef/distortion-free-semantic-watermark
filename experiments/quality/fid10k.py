#!/usr/bin/env python3
"""Matched-population FID (generation quality table): FID between watermarked and matched clean images.

Our method, per configuration (N, M) and key ``k``: FID between the watermarked population of
every hyperplane frame (the images selected with key ``k``) and the corpus's separately
generated clean population. Both contain one image per prompt and slot. The clean population
is shared across configurations, frames and keys. The five frames are averaged within each key,
then the keys are averaged (the 95% interval is over the keys). A native baseline contributes
one FID between its watermarked images and its own clean images over the same prompts and slots.

    python experiments/quality/fid10k.py --selections selections.npz --ours-activations activations/ours \
        --native-activations tree_ring=activations/tree_ring gaussian_shading=activations/gaussian_shading \
                             seal=activations/seal --output-dir results/fid10k

CPU only; the activations come from ``inception_activations.py``. Writes ``pairs.csv`` (one FID
per pair) and ``summary.csv`` (one row per method and configuration).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402


def ours_pair_rows(candidates: np.ndarray, control: tuple, selections: dict, metadata: dict, n: int, m: int):
    """FID of every (key, frame) pair of one configuration, watermarked against the matched control."""
    watermarked = selections[lib.selection_key(n, m, "watermarked")]
    rows = []
    for key in range(watermarked.shape[1]):
        for frame, frame_id in enumerate(metadata["frame_ids"]):
            marked = lib.population_statistics(lib.gather(candidates, watermarked[frame, key]))
            rows.append({"family": "ours", "N": n, "M": m, "key": key, "frame": frame_id,
                         "fid": lib.frechet_distance(*control, *marked)})
        print("N=%d M=%d key %d" % (n, m, key), flush=True)
    return rows


def summarize_ours(rows) -> dict:
    """Mean over frames within each key, then the mean and 95% interval over the keys."""
    by_key = {}
    for row in rows:
        by_key.setdefault(row["key"], []).append(row["fid"])
    return lib.mean_interval([float(np.mean(by_key[key])) for key in sorted(by_key)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--ours-activations", type=Path, required=True)
    parser.add_argument("--native-activations", nargs="*", default=[], metavar="NAME=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    uids, selections, metadata = lib.load_selections(args.selections)
    activations = lib.load_shards(args.ours_activations, ["candidates", "cleans"], uids)
    candidates = activations["candidates"]
    cleans = activations["cleans"]
    if cleans.shape != (*candidates.shape[:2], candidates.shape[-1]):
        raise ValueError("clean and candidate activations must have matching prompts, slots and features")
    control = lib.population_statistics(cleans)
    pair_rows, summary_rows = [], []
    for n, m in metadata["configurations"]:
        rows = ours_pair_rows(candidates, control, selections, metadata, n, m)
        pair_rows.extend(rows)
        summary_rows.append({"family": "ours", "N": n, "M": m, "aggregation": "mean over frames, then keys",
                             **{"fid_" + name: value for name, value in summarize_ours(rows).items()}})
    del activations, candidates, cleans
    for name, directory in sorted(lib.parse_named_paths(args.native_activations).items()):
        shards = lib.load_shards(directory, ["watermarked", "cleans"], uids)
        fid = lib.frechet_distance(*lib.population_statistics(shards["cleans"]),
                                   *lib.population_statistics(shards["watermarked"]))
        pair_rows.append({"family": name, "fid": fid})
        summary_rows.append({"family": name, "aggregation": "one matched pair",
                             **{"fid_" + key: value for key, value in lib.mean_interval([fid]).items()}})
        print("%s done" % name, flush=True)
    lib.write_csv(args.output_dir / "pairs.csv", pair_rows)
    lib.write_csv(args.output_dir / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
