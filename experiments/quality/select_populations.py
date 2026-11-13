#!/usr/bin/env python3
"""Select the watermarked and matched-random populations of the generation-quality metrics.

For every configuration (N, M) of ``configs/detection.yaml`` Gumbel selection (the paper's method)
runs on the first N candidates of every (prompt, slot) pool of the embedding cache, once per
hyperplane frame and PRF key, and a matched control draws one candidate uniformly from the
same N for every key. The watermarked population of a (frame, key) is the set of selected
images over all prompts and slots; the control of key ``k`` is the uniform draw ``k``.
Fixed-key quality is measured per key, so ``--num-keys`` keys ``derive_trial_key(master, t)``
are used (the detection experiments use the first ``replication.keys`` of them).

    python experiments/quality/select_populations.py --cache cache.npz --key-file keys/detector.key \
        --num-keys 10 --output selections.npz

``selections.npz`` holds, per configuration ``n{N}_m{M}``, ``n{N}_m{M}_watermarked`` of shape
(frames, keys, prompts, slots) and ``n{N}_m{M}_random`` of shape (keys, prompts, slots) with
1-based candidate positions. ``fid10k.py`` reads the watermarked positions. ``clipscore.py``
also reads the matched-random controls.
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


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="embedding cache (encoder: ours)")
    parser.add_argument("--key-file", type=Path, required=True, help="32-byte master key")
    parser.add_argument("--num-keys", type=int, required=True, help="number of fixed keys")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = load_cache(args.cache)
    vectors = cache.candidate_vectors
    slots = vectors.shape[1]
    planes = frame_planes(cfg, vectors.shape[-1])
    keys = lib.trial_keys(lib.read_master_key(args.key_file), args.num_keys)
    arrays = {}
    for setting in cfg.settings:
        n, m = setting.n, setting.m
        arrays[lib.selection_key(n, m, "watermarked")] = np.stack([
            np.stack([
                lib.select_positions(vectors, planes[frame_id], keys[trial], frame_id=frame_id, key_trial=trial,
                                     n=n, m=m, prompt_uids=cache.prompt_uids)
                for trial in sorted(keys)])
            for frame_id in cfg.frame_ids()])
        arrays[lib.selection_key(n, m, "random")] = np.stack([
            lib.matched_random_positions(cache.prompt_uids, replicate, n, slots) for replicate in sorted(keys)])
        print("selected N=%d M=%d" % (n, m), flush=True)
    lib.write_selections(args.output, cache.prompt_uids, arrays, {
        "frame_ids": list(cfg.frame_ids()), "num_keys": args.num_keys,
        "configurations": [[setting.n, setting.m] for setting in cfg.settings], "slots": slots,
        "cache_panel_sha256": cache.metadata.get("panel_sha256")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
