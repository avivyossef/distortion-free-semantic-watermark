#!/usr/bin/env python3
"""Generate 16 same-prompt FLUX.2 [klein] images for each of 200 held-out prompts.

Input ``--prompts`` is a JSONL file with one row per prompt::

    {"prompt_uid": "...", "text": "..."}

The first 200 prompt identities in sorted order are used (the same prompts as the SD2.1
panel). Image ``k`` (1..16) of prompt number ``i`` uses seed ``seed_base + stride * i + k - 1``
of the chosen variant (``comparison_lib.FLUX_VARIANTS``). Every prompt is written to
``<output-dir>/<prompt_uid>/image_NN.png`` with a ``panel.json`` record; prompts are dealt to
shards round-robin and an existing prompt directory is never overwritten.

    python experiments/encoder_comparison/generate_flux.py --prompts prompts.jsonl \
        --variant distilled --output-dir panels/flux_distilled --shard-index 0 --num-shards 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison_lib import FLUX_IMAGE_SIZE, FLUX_VARIANTS, GROUP_SIZE, flux_seeds, select_prompts  # noqa: E402
from watermark.io import canonical_json, pixel_sha256, read_jsonl, sha256_file  # noqa: E402


def load_flux_pipeline(variant: str, device: str):
    import torch
    from diffusers import Flux2KleinPipeline

    spec = FLUX_VARIANTS[variant]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return Flux2KleinPipeline.from_pretrained(spec["repo"], revision=spec["revision"], torch_dtype=dtype).to(device)


def write_prompt_panel(pipe, *, variant: str, prompt_index: int, prompt_uid: str, text: str, output_dir: Path,
                       device: str) -> None:
    import torch
    from PIL import Image

    spec = FLUX_VARIANTS[variant]
    destination = Path(output_dir) / prompt_uid
    if destination.exists():
        raise ValueError("refusing to overwrite %s" % destination)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s.staging-" % prompt_uid, dir=str(output_dir)))
    try:
        images = []
        for position, seed in enumerate(flux_seeds(variant, prompt_index), 1):
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(
                prompt=text, generator=generator, num_inference_steps=spec["steps"],
                guidance_scale=spec["guidance"], height=FLUX_IMAGE_SIZE, width=FLUX_IMAGE_SIZE,
            ).images[0].convert("RGB")
            name = "image_%02d.png" % position
            image.save(staging / name, format="PNG")
            with Image.open(staging / name) as stored:
                images.append({"candidate_position": position, "relative_path": name, "seed": seed,
                               "file_sha256": sha256_file(staging / name), "pixel_sha256": pixel_sha256(stored)})
        record = {"prompt_uid": prompt_uid, "prompt": text, "variant": variant, "model_id": spec["repo"],
                  "model_revision": spec["revision"], "num_inference_steps": spec["steps"],
                  "guidance_scale": spec["guidance"], "image_length": FLUX_IMAGE_SIZE, "images": images}
        (staging / "panel.json").write_text(canonical_json(record) + "\n", encoding="utf-8")
        os.replace(str(staging), str(destination))
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(FLUX_VARIANTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= I < --num-shards")
    prompts = select_prompts(read_jsonl(args.prompts))
    pipe = load_flux_pipeline(args.variant, args.device)
    for index in range(args.shard_index, len(prompts), args.num_shards):
        row = prompts[index]
        write_prompt_panel(pipe, variant=args.variant, prompt_index=index, prompt_uid=str(row["prompt_uid"]),
                           text=str(row["text"]), output_dir=args.output_dir, device=args.device)
        print("prompt %s: %d images" % (row["prompt_uid"], GROUP_SIZE), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
