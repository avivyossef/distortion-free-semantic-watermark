#!/usr/bin/env python3
"""Score images with the native detector of Tree-Ring, Gaussian Shading or SEAL.

Run in the environment of the method (see the README). ``--images`` is a JSONL file of rows
``{image_id, path, prompt}``; the output has one row ``{image_id, method, score}`` per input.
A HIGHER SCORE MEANS MORE LIKELY WATERMARKED. The detectors need no prompt (the field is carried
for callers); ``--master-key-file`` must be the key file the images were generated with.
Thresholds are chosen by the caller, from clean-image scores only.

    python experiments/baselines/score.py --method tree_ring --official-root third_party/checkouts \
        --master-key-file key.bin --images images.jsonl --output scores.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _baselines_lib as lib
from watermark.io import read_jsonl, read_key_file, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=lib.METHODS, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--master-key-file", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    rows = [row for index, row in enumerate(read_jsonl(args.images))
            if lib.in_shard(index, args.shard_index, args.num_shards)]
    ids = [row["image_id"] for row in rows]
    if len(set(ids)) != len(ids):
        parser.error("duplicate image_id in %s" % args.images)
    import _baselines_native as native

    score = native.load_scorer(args.method, args.official_root, read_key_file(args.master_key_file))
    write_jsonl(args.output, ({"image_id": row["image_id"], "method": args.method, "score": float(score(row["path"]))}
                              for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
