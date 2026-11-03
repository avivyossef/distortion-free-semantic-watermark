#!/usr/bin/env python3
"""Generate the SDXL training sources: final images and fifth-step previews per (prompt, seed).

Input ``--prompts`` is a JSONL file with one row per prompt::

    {"prompt_uid": "p0001", "text": "...", "split": "train" | "validation"}

Seeds are allocated once for all prompts from ``--registry-seed`` (``encoder.sdxl.allocate_seeds``),
so shards never share a seed. Prompts are dealt to shards round-robin in sorted order. Each shard
writes images under ``--output-dir/<prompt_uid>/`` and its manifest rows to
``--output-dir/manifests/shard-<I>-of-<K>.jsonl``.

    python experiments/encoder_training/generate_pairs.py --prompts prompts.jsonl \
        --output-dir data/sources --registry-seed 1 --seeds-per-prompt 64 \
        --shard-index 0 --num-shards 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encoder.sdxl import allocate_seeds, load_sdxl_pipeline, write_prompt_shard  # noqa: E402
from watermark.io import read_jsonl, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-seed", type=int, required=True, help="seed of the allocator stream")
    parser.add_argument("--seeds-per-prompt", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= I < --num-shards")
    prompts = sorted(read_jsonl(args.prompts), key=lambda row: str(row["prompt_uid"]))
    for row in prompts:
        if row.get("split") not in ("train", "validation") or not str(row.get("text", "")).strip():
            raise SystemExit("invalid prompt row: %r" % (row,))
    seeds = allocate_seeds([str(row["prompt_uid"]) for row in prompts], registry_seed=args.registry_seed,
                           seeds_per_prompt=args.seeds_per_prompt)
    pipeline = load_sdxl_pipeline(args.device, cpu_offload=args.cpu_offload)
    rows = []
    for row in prompts[args.shard_index::args.num_shards]:
        uid = str(row["prompt_uid"])
        rows.extend(write_prompt_shard(
            pipeline, prompt_uid=uid, prompt_split=row["split"], text=str(row["text"]),
            seeds=seeds[uid], output_dir=args.output_dir,
        ))
        print("prompt %s: %d sources" % (uid, len(seeds[uid])), flush=True)
    write_jsonl(args.output_dir / "manifests" / ("shard-%03d-of-%03d.jsonl" % (args.shard_index, args.num_shards)), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
