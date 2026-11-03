#!/usr/bin/env python3
"""FID against real COCO images (appendix quality table): clean and watermarked populations.

One image per validation prompt. Our clean population is the clean control of each prompt; the
watermarked population of a configuration (N, M) selects among the first N of the prompt's
candidates with hyperplane frame 0 and a key that rotates over ``--num-keys`` PRF keys (default ``_quality_lib.FID5K_NUM_KEYS``) along
the prompts (sorted by identity), so no single key defines the population. Each native
baseline contributes its watermarked and its own clean images. The report gives the FID of every
population against the real images and the relative change of each watermarked population from its
matched clean population, ``100 (FID_wm / FID_clean - 1)``.

    python experiments/quality/fid5k.py --cache cache.npz --key-file keys/detector.key \
        --ours-activations activations/ours --real-activations activations/real \
        --native-activations tree_ring=activations/tree_ring gaussian_shading=activations/gaussian_shading \
                             seal=activations/seal --output-dir results/fid5k

CPU only. ``cache.npz`` is the embedding cache (encoder: ours) of the one-slot corpus of the
5,000 panel prompts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402

from detection_corpus import frame_planes, load_cache  # noqa: E402

from watermark.config import load_detection_config  # noqa: E402


def rotating_key_positions(vectors: np.ndarray, planes: np.ndarray, keys: dict, *, frame_id: str, n: int,
                           m: int, prompt_uids) -> np.ndarray:
    """Selected positions ``(P, S)`` where prompt ``i`` uses key trial ``i mod len(keys)``."""
    positions = np.empty(vectors.shape[:2], dtype=np.int64)
    for trial in sorted(keys):
        rows = np.arange(trial, len(prompt_uids), len(keys))
        if not len(rows):
            continue
        positions[rows] = lib.select_positions(
            vectors[rows], planes, keys[trial], frame_id=frame_id, key_trial=trial, n=n, m=m,
            prompt_uids=[prompt_uids[row] for row in rows])
    return positions


def matched_changes(fids: dict, pairs: dict) -> list:
    """Relative change of each watermarked FID from its matched clean FID."""
    rows = []
    for population, clean in pairs.items():
        change = 100.0 * (fids[population] / fids[clean] - 1.0)
        rows.append({"population": population, "clean_population": clean, "fid": fids[population],
                     "clean_fid": fids[clean], "relative_change_percent": change})
    return rows


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--num-keys", type=int, default=lib.FID5K_NUM_KEYS, help="keys the prompts rotate over")
    parser.add_argument("--ours-activations", type=Path, required=True)
    parser.add_argument("--real-activations", type=Path, required=True)
    parser.add_argument("--native-activations", nargs="*", default=[], metavar="NAME=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = load_cache(args.cache)
    uids = list(cache.prompt_uids)
    frame_id = cfg.frame_ids()[0]
    planes = frame_planes(cfg, cache.candidate_vectors.shape[-1])[frame_id]
    keys = lib.trial_keys(lib.read_master_key(args.key_file), args.num_keys)
    ours = lib.load_shards(args.ours_activations, ["candidates", "cleans"], uids)
    real = lib.load_shards(args.real_activations, ["images"], uids)["images"]
    real_stats = lib.activation_statistics(real)
    populations = {"ours_clean": ours["cleans"][:, 0]}
    pairs = {}
    for setting in cfg.settings:
        positions = rotating_key_positions(cache.candidate_vectors, planes, keys, frame_id=frame_id, n=setting.n,
                                           m=setting.m, prompt_uids=uids)
        name = "ours_" + lib.config_name(setting.n, setting.m)
        populations[name] = lib.gather(ours["candidates"], positions)[:, 0]
        pairs[name] = "ours_clean"
    for name, directory in sorted(lib.parse_named_paths(args.native_activations).items()):
        shards = lib.load_shards(directory, ["watermarked", "cleans"], uids)
        populations[name], populations[name + "_clean"] = shards["watermarked"][:, 0], shards["cleans"][:, 0]
        pairs[name] = name + "_clean"
    fids = {}
    for name, values in populations.items():
        fids[name] = lib.frechet_distance(*real_stats, *lib.activation_statistics(values))
        print("%s done" % name, flush=True)
    lib.write_csv(args.output_dir / "fid.csv", [{"population": name, "fid": value} for name, value in fids.items()])
    lib.write_csv(args.output_dir / "matched_changes.csv", matched_changes(fids, pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
