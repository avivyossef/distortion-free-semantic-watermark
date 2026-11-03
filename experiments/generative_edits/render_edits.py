#!/usr/bin/env python3
"""Qwen-Image-Edit renders every planned edit.

The model receives the source image and its instruction, with the fixed render settings of
``_edits_lib.EDITOR`` (seed, steps, true CFG scale, bfloat16, output size = source size).

    python experiments/generative_edits/render_edits.py --sources edits/sources.jsonl \
        --plans-dir edits/plans --output-dir edits/rendered --shard-index 0 --num-shards 1

Writes ``images/<source_id>.png`` and ``manifest-shard-III-of-KKK.jsonl``; finished edits are
skipped on rerun. The paper reviews every edited image by eye (the requested change is visible,
the main subject and scene are preserved); that is a manual step, see the README. Needs a GPU
with enough memory for the full pipeline and ``diffusers`` with ``QwenImageEditPipeline``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import EDITOR, deal, read_shards  # noqa: E402

from watermark.io import read_jsonl, sha256_file  # noqa: E402


def load_editor(device: str):
    import torch
    from diffusers import QwenImageEditPipeline

    pipe = QwenImageEditPipeline.from_pretrained(
        EDITOR["repo"], revision=EDITOR["revision"], torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, torch


def render(pipe, torch, source, instruction: str):
    """One edited image (RGB) of a source PIL image."""
    result = pipe(
        image=source, prompt=instruction, generator=torch.manual_seed(EDITOR["seed"]),
        true_cfg_scale=EDITOR["true_cfg_scale"], negative_prompt=EDITOR["negative_prompt"],
        num_inference_steps=EDITOR["num_inference_steps"], width=source.width, height=source.height,
    ).images
    if len(result) != 1:
        raise ValueError("the editor returned an unexpected number of images")
    return result[0].convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--plans-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from PIL import Image

    plans = {row["source_id"]: row for row in read_shards(args.plans_dir, "plans-shard-*.jsonl")}
    sources = deal(sorted(read_jsonl(args.sources), key=lambda row: row["source_id"]),
                   args.shard_index, args.num_shards)
    missing = [row["source_id"] for row in sources if row["source_id"] not in plans]
    if missing:
        raise SystemExit("%d sources have no plan; run plan_instructions.py for every shard first" % len(missing))
    manifest = args.output_dir / ("manifest-shard-%03d-of-%03d.jsonl" % (args.shard_index, args.num_shards))
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    done = {row["source_id"] for row in read_jsonl(manifest)} if manifest.exists() else set()
    pipe, torch = load_editor(args.device)
    with manifest.open("a", encoding="utf-8") as handle:
        for row in sources:
            if row["source_id"] in done:
                continue
            instruction = plans[row["source_id"]]["plan"]["edit_instruction"]
            with Image.open(row["path"]) as opened:
                source = opened.convert("RGB").copy()
            relative = "images/%s.png" % row["source_id"]
            render(pipe, torch, source, instruction).save(args.output_dir / relative, format="PNG")
            handle.write(json.dumps({
                "source_id": row["source_id"], "edit_category": row["edit_category"],
                "instruction": instruction, "edited_path": relative,
                "edited_file_sha256": sha256_file(args.output_dir / relative)}, sort_keys=True) + "\n")
            handle.flush()
            print("edited %s" % row["source_id"][:12], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
