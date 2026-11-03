#!/usr/bin/env python3
"""Generate native watermarked and clean SD2.1 images with Tree-Ring, Gaussian Shading or SEAL.

Run in the environment of the method (see the README). For every prompt and every slot
``1..--num-slots`` the watermarked and the clean image of that slot share one seed; the layout
written under ``--output-dir`` is documented in the README::

    <prompt_uid>/slot_001/watermarked.png, clean.png, ...
    <prompt_uid>/panel.json

    python experiments/baselines/generate.py --method tree_ring --prompts prompts.jsonl \
        --num-slots 1 --official-root third_party/checkouts --master-key-file keys/baselines.key --output-dir out/tree_ring
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _baselines_lib as lib
from watermark.io import read_key_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=lib.METHODS, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL rows {prompt_uid, text}")
    parser.add_argument("--num-slots", type=int, required=True, help="independent generations per prompt")
    parser.add_argument("--official-root", type=Path, required=True, help="directory written by third_party/clone.sh")
    parser.add_argument("--master-key-file", type=Path, required=True, help="32-byte file; fixes the method state")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_slots < 1:
        parser.error("--num-slots must be positive")

    prompts = [row for index, row in enumerate(lib.read_prompts(args.prompts))
               if lib.in_shard(index, args.shard_index, args.num_shards)]
    todo = [row for row in prompts if not (args.output_dir / row["prompt_uid"] / "panel.json").is_file()]
    if not todo:
        return 0
    import _baselines_native as native

    generator = native.load_generator(args.method, args.official_root, read_key_file(args.master_key_file))
    model = native.sd21_model()
    for row in todo:
        uid, text = row["prompt_uid"], row["text"]

        def write_images(staging: Path) -> list:
            slots = []
            for slot in range(1, args.num_slots + 1):
                watermarked, clean = generator.generate(text, lib.generation_seed(uid, slot))
                slot_dir = staging / lib.slot_dir_name(slot)
                slot_dir.mkdir()
                entry = {"slot": slot}
                for role, image in (("watermarked", watermarked), ("clean", clean)):
                    path = slot_dir / (role + ".png")
                    image.convert("RGB").save(path, format="PNG")
                    entry[role] = lib.image_record(path, staging)
                slots.append(entry)
            return slots

        lib.write_prompt_panel(args.output_dir, {
            "prompt_uid": uid, "method": args.method, "prompt": text,
            "official_commit": lib.OFFICIAL_COMMITS[args.method],
            "model_id": model["repo"], "model_revision": model["revision"],
            "image_length": native.IMAGE_LENGTH, "num_inference_steps": native.NUM_INFERENCE_STEPS,
            "guidance_scale": native.GUIDANCE_SCALE,
        }, write_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
