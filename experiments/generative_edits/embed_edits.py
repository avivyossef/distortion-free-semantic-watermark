#!/usr/bin/env python3
"""Embed every edited image with the trained semantic encoder (for scoring our method).

    python experiments/generative_edits/embed_edits.py --render-dir edits/rendered \
        --checkpoint checkpoints/checkpoint-step-00019840.pt --output edits/edited_embeddings.npz

Writes an ``.npz`` with ``source_ids (E,)`` and ``vectors (E, D)`` (unit-norm float64), the
embedding of the edit of each source image. Needs a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import read_shards  # noqa: E402

from watermark.embedding import embed_paths, load_encoder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    rows = sorted(read_shards(args.render_dir, "manifest-shard-*.jsonl"), key=lambda row: row["source_id"])
    encoder = load_encoder("ours", device=args.device, checkpoint=args.checkpoint)
    vectors = embed_paths(encoder, [args.render_dir / row["edited_path"] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output), source_ids=np.asarray([row["source_id"] for row in rows]),
                        vectors=vectors)
    print("embedded %d edited images" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
