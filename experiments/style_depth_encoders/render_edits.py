#!/usr/bin/env python3
"""Render the three InstructPix2Pix style edits of every candidate and clean image.

For the first 100 prompts of the SD2.1 panel, edits the 16 slot-1 candidates and the slot-1 clean
image (17 images per prompt). Editing every candidate lets any encoder, hyperplane frame
and key score its own selection afterwards. The spatial transformations are applied at
embedding time instead.

Layout: ``<output-dir>/<edit_id>/<prompt_uid>/<NN>.png`` with NN 00-15 the candidates in
position order and 16 the clean image, plus a ``manifest.json`` per prompt and edit. Prompts
are dealt to shards round-robin; finished prompts are skipped.

    python experiments/style_depth_encoders/render_edits.py --panel-root panels/sd21 \
        --output-dir edits --shard-index 0 --num-shards 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from style_depth_lib import (  # noqa: E402
    EDIT_SIZE, GUIDANCE, IMAGE_GUIDANCE, IP2P_MODEL, IP2P_REVISION, SLOT_CANDIDATES, STEPS, STYLE_EDITS,
    edit_seed, panel_prompt_uids, source_paths,
)
from watermark.io import canonical_json, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= I < --num-shards")
    mine = panel_prompt_uids(args.panel_root)[args.shard_index::args.num_shards]

    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from PIL import Image

    # No safety checker: it would replace flagged edits with black images.
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        IP2P_MODEL, revision=IP2P_REVISION, torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    for done, uid in enumerate(mine, 1):
        paths = source_paths(args.panel_root, uid)
        for edit_id, instruction in STYLE_EDITS.items():
            folder = args.output_dir / edit_id / uid
            if (folder / "manifest.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            records = []
            for index, source in enumerate(paths):
                with Image.open(source) as opened:
                    image = opened.convert("RGB").resize((EDIT_SIZE, EDIT_SIZE), Image.Resampling.BICUBIC)
                seed = edit_seed(uid, edit_id, index)
                result = pipe(prompt=instruction, image=image, num_inference_steps=STEPS,
                              image_guidance_scale=IMAGE_GUIDANCE, guidance_scale=GUIDANCE,
                              generator=torch.Generator(device=args.device).manual_seed(seed)).images[0]
                out = folder / ("%02d.png" % index)
                result.convert("RGB").save(out)
                records.append({"index": index, "role": "clean" if index == SLOT_CANDIDATES else "candidate",
                                "source_sha256": sha256_file(source), "output": out.name,
                                "output_sha256": sha256_file(out), "seed": seed})
            (folder / "manifest.json").write_text(canonical_json({
                "prompt_uid": uid, "edit_id": edit_id, "instruction": instruction, "model": IP2P_MODEL,
                "revision": IP2P_REVISION, "safety_checker": None, "variant": "fp16", "edit_size": EDIT_SIZE,
                "steps": STEPS, "image_guidance_scale": IMAGE_GUIDANCE, "guidance_scale": GUIDANCE,
                "records": records}) + "\n", encoding="utf-8")
        print("shard %d: %d/%d prompts" % (args.shard_index, done, len(mine)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
