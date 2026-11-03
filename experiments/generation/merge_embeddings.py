#!/usr/bin/env python3
"""Join the shards written by ``embed_corpus.py`` into one embedding cache.

    python experiments/generation/merge_embeddings.py --shards-dir embeddings --output cache.npz

Fails unless the shards cover disjoint prompts, agree on the encoder and have equal geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_corpus import load_cache, write_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.shards_dir.glob("embeddings-shard-*-of-*.npz"))
    if not paths:
        raise SystemExit("no shards found")
    shards = [load_cache(path) for path in paths]
    expected = int(paths[0].stem.rsplit("-of-", 1)[1])
    if len(shards) != expected or any(int(path.stem.rsplit("-of-", 1)[1]) != expected for path in paths):
        raise SystemExit("expected exactly %d shards" % expected)
    if len({shard.metadata["encoder"] for shard in shards}) != 1 \
            or len({shard.candidate_vectors.shape[1:] for shard in shards}) != 1:
        raise SystemExit("shards disagree on encoder or geometry")
    uids = [uid for shard in shards for uid in shard.prompt_uids]
    if len(set(uids)) != len(uids):
        raise SystemExit("shards overlap")
    metadata = dict(shards[0].metadata)
    metadata.pop("version")
    metadata["panel_sha256"] = hashlib.sha256(
        "".join(sorted(shard.metadata["panel_sha256"] for shard in shards)).encode()).hexdigest()
    write_cache(args.output, uids, np.concatenate([shard.candidate_vectors for shard in shards]),
                np.concatenate([shard.clean_vectors for shard in shards]), metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
