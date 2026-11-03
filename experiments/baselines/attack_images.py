#!/usr/bin/env python3
"""Render the 20 ordinary edits on the watermarked and clean images of a ``generate.py`` output.

Edited PNGs are written under ``--output-dir/<edit_id>/<prompt_uid>/<role>.png`` and one row
``{image_id, path, prompt}`` per edited image is written to ``--images-output`` for ``score.py``
(the image identifier is ``<prompt_uid>|<role>|<edit_id>``). The edits are those of
``watermark.attacks``; shard over prompts with ``--shard-index/--num-shards`` and give each shard
its own ``--images-output``.

    python experiments/baselines/attack_images.py --generation-dir out/tree_ring \
        --prompts prompts.jsonl --output-dir out/tree_ring_edits --images-output edited_images.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _baselines_lib as lib
from watermark.attacks import EDIT_IDS, render_edit
from watermark.io import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    from PIL import Image

    rows = []
    for index, prompt in enumerate(lib.read_prompts(args.prompts)):
        if not lib.in_shard(index, args.shard_index, args.num_shards):
            continue
        panel = lib.read_panel(args.generation_dir, prompt["prompt_uid"])
        for role in lib.ROLES:
            with Image.open(lib.panel_image_path(args.generation_dir, panel, args.slot, role)) as source:
                source.load()
                for edit_id in EDIT_IDS:
                    target = args.output_dir / edit_id / panel["prompt_uid"] / (role + ".png")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    render_edit(source, edit_id).save(target, format="PNG")
                    rows.append({"image_id": lib.make_image_id(panel["prompt_uid"], role, edit_id),
                                 "path": str(target.resolve()), "prompt": panel["prompt"]})
    write_jsonl(args.images_output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
