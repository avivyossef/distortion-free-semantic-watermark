#!/usr/bin/env python3
"""DINOv2-base embeddings of the images of a native-baseline generation (diversity metric).

Embeds the watermarked and the clean image of every slot with the frozen DINOv2-base encoder
of ``watermark.embedding`` and writes ``<output-dir>/embeddings-shard-III-of-KKK.npz`` with
``watermarked`` and ``cleans`` of shape (p, S, 768). ``vendi.py`` reads the shards.

    python experiments/quality/embed_native.py --native-dir out/tree_ring --prompts prompts.jsonl \
        --output-dir embeddings/tree_ring --shard-index 0 --num-shards 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402

from watermark.embedding import embed_paths, load_encoder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--slots", type=int, help="use only the first slots (default: all)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    uids = [str(row["prompt_uid"]) for row in lib.deal(lib.read_prompts(args.prompts), args.shard_index,
                                                       args.num_shards)]
    encoder = load_encoder("dinov2_base", device=args.device)
    marked, cleans = [], []
    for uid in uids:
        marked_paths, clean_paths = lib.native_images(args.native_dir, uid, args.slots)
        vectors = embed_paths(encoder, list(marked_paths) + list(clean_paths))
        marked.append(vectors[:len(marked_paths)])
        cleans.append(vectors[len(marked_paths):])
        print("embedded %s" % uid, flush=True)
    lib.write_shard(args.output_dir / ("embeddings-shard-%03d-of-%03d.npz" % (args.shard_index, args.num_shards)),
                    uids, watermarked=np.stack(marked), cleans=np.stack(cleans))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
