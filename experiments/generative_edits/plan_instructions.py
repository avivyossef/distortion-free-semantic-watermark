#!/usr/bin/env python3
"""Qwen3-VL-8B-Instruct writes the image-specific edit instruction of every source image.

Each source image is shown to the model with the fixed prompt of its edit category (paper table
of category constraints); decoding is greedy. The response must be exactly one JSON plan of the
assigned category, otherwise the script stops (rerun to resume: finished sources are skipped).

    python experiments/generative_edits/plan_instructions.py --sources edits/sources.jsonl \
        --output-dir edits/plans --shard-index 0 --num-shards 1

Writes ``plans-shard-III-of-KKK.jsonl``. Needs a GPU, ``transformers`` with Qwen3-VL support and
the pinned model (revision in ``_edits_lib.PLANNER``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import PLANNER, deal, edit_plan_prompt, validate_edit_plan  # noqa: E402

from watermark.io import read_jsonl  # noqa: E402


def load_planner(device: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(PLANNER["repo"], revision=PLANNER["revision"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        PLANNER["repo"], revision=PLANNER["revision"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, processor, torch


def plan_image(model, processor, torch, image, category: str) -> str:
    """Greedy response of the vision-language model for one image and category."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": edit_plan_prompt(category)},
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=PLANNER["do_sample"],
                                   max_new_tokens=PLANNER["max_new_tokens"], use_cache=True)
    trimmed = [output[len(prompt):] for prompt, output in zip(inputs["input_ids"], generated)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
    if len(decoded) != 1:
        raise ValueError("the planner returned an unexpected number of responses")
    return decoded[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from PIL import Image

    sources = deal(sorted(read_jsonl(args.sources), key=lambda row: row["source_id"]),
                   args.shard_index, args.num_shards)
    output = args.output_dir / ("plans-shard-%03d-of-%03d.jsonl" % (args.shard_index, args.num_shards))
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {row["source_id"] for row in read_jsonl(output)} if output.exists() else set()
    model, processor, torch = load_planner(args.device)
    with output.open("a", encoding="utf-8") as handle:
        for row in sources:
            if row["source_id"] in done:
                continue
            with Image.open(row["path"]) as opened:
                image = opened.convert("RGB").copy()
            response = plan_image(model, processor, torch, image, row["edit_category"])
            plan = validate_edit_plan(response, expected_category=row["edit_category"])
            handle.write(json.dumps({"source_id": row["source_id"], "prompt_uid": row["prompt_uid"],
                                     "edit_category": row["edit_category"], "response": response,
                                     "plan": plan}, sort_keys=True) + "\n")
            handle.flush()
            print("planned %s" % row["source_id"][:12], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
