#!/usr/bin/env python3
"""Generate the SD2.1 candidate corpus: per prompt, S slots of N candidates plus a clean control.

Every slot holds ``--candidates`` independent watermark candidates and one unwatermarked image
generated with the same model and settings; slot ``s`` serves one image of a detection set, so
``--slots`` is the largest set size B. Defaults come from ``configs/detection.yaml`` (the
largest N of the configurations and the largest set size).

``--prompts`` is a JSONL file, one row per prompt: ``{"prompt_uid": "...", "text": "..."}``,
optionally with ``"native_key_id"``, the label that seeds the prompt's clean controls (the
paper's prompt list carries it; without it every prompt uses the same default label).
Seeds derive from the 32-byte ``--seed-key-file`` (see ``new_key.py``) and the prompt, slot
and position, so a corpus is reproducible from the key and the prompt list.

    python experiments/generation/generate_corpus.py --prompts prompts.jsonl \
        --seed-key-file keys/seed.key --output-dir corpus --shard-index 0 --num-shards 1

Prompts are dealt to shards round-robin in sorted order; prompts whose output already exists
are skipped, so an interrupted shard can be restarted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_corpus import deal, read_master_key, read_prompts  # noqa: E402

from watermark.config import load_detection_config  # noqa: E402
from watermark.generation import NATIVE_KEY_ID, generate_prompt_panel, load_sd21_pipeline  # noqa: E402


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--seed-key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=max(item.n for item in cfg.settings),
                        help="candidates per slot (default: the largest N of the configurations)")
    parser.add_argument("--slots", type=int, default=max(cfg.set_sizes),
                        help="independent slots per prompt (default: the largest set size)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prompts = deal(read_prompts(args.prompts), args.shard_index, args.num_shards)
    master_key = read_master_key(args.seed_key_file)
    pipe = load_sd21_pipeline(args.device)
    for row in prompts:
        uid = str(row["prompt_uid"])
        if (args.output_dir / uid).exists():
            continue
        generate_prompt_panel(
            pipe, prompt_uid=uid, prompt=str(row["text"]), master_key=master_key, output_dir=args.output_dir,
            mode="candidates", batch_size=args.slots, candidate_count=args.candidates, device=args.device,
            native_key_id=str(row.get("native_key_id", NATIVE_KEY_ID)))
        print("prompt %s: %d slots x %d candidates" % (uid, args.slots, args.candidates), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
