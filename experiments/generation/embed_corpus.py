#!/usr/bin/env python3
"""Embed the images of a candidate corpus with an encoder (one shard of the prompts).

Writes ``<output-dir>/embeddings-shard-III-of-KKK.npz``; ``merge_embeddings.py`` joins the shards
into the embedding cache read by every detection experiment. The paper's watermark uses
``--encoder ours`` with the trained checkpoint; the other encoders are the appendix ablation
(they embed in their native dimension).

    python experiments/generation/embed_corpus.py --corpus-dir corpus --prompts prompts.jsonl \
        --encoder ours --checkpoint checkpoints/checkpoint-step-00019840.pt \
        --output-dir embeddings --shard-index 0 --num-shards 1

``--slots`` limits the embedded slots (the ablation and the robustness experiments use slot 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_corpus import (ENCODERS, deal, panel_images, panel_manifest_sha256, read_panel,  # noqa: E402
                    read_prompts, write_cache)

from watermark.config import load_assets  # noqa: E402
from watermark.embedding import embed_paths, load_encoder  # noqa: E402


def embed_prompts(encoder, corpus_dir: Path, prompt_uids, slots):
    """Embeddings of the candidates and clean controls of ``prompt_uids``: ``(P, S, N, D)``, ``(P, S, D)``."""
    candidate_rows, clean_rows = [], []
    for uid in prompt_uids:
        candidates, cleans = panel_images(corpus_dir, read_panel(corpus_dir, uid), slots)
        flat = [path for slot in candidates for path in slot] + list(cleans)
        vectors = embed_paths(encoder, flat)
        count = sum(len(slot) for slot in candidates)
        candidate_rows.append(vectors[:count].reshape(len(candidates), len(candidates[0]), -1))
        clean_rows.append(vectors[count:])
        print("embedded %s" % uid, flush=True)
    return np.stack(candidate_rows), np.stack(clean_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, default="ours")
    parser.add_argument("--checkpoint", type=Path, help="trained-encoder checkpoint (--encoder ours)")
    parser.add_argument("--slots", type=int, help="embed only the first slots (default: all)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    uids = [str(row["prompt_uid"]) for row in read_prompts(args.prompts)]
    mine = deal(uids, args.shard_index, args.num_shards)
    encoder = load_encoder(args.encoder, device=args.device, checkpoint=args.checkpoint)
    candidates, cleans = embed_prompts(encoder, args.corpus_dir, mine, args.slots)
    metadata = {"encoder": args.encoder, "panel_sha256": panel_manifest_sha256(args.corpus_dir, mine),
                "checkpoint_sha256": load_assets()["checkpoint"]["sha256"] if args.encoder == "ours" else None}
    write_cache(args.output_dir / ("embeddings-shard-%03d-of-%03d.npz" % (args.shard_index, args.num_shards)),
                mine, candidates, cleans, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
