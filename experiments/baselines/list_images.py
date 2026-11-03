#!/usr/bin/env python3
"""List the unedited images of a ``generate.py`` output for ``score.py``.

Writes one row ``{image_id, path, prompt}`` per prompt and role (watermarked and clean) of one
slot; the image identifier is ``<prompt_uid>|<role>|identity``. ``--prompts`` selects (and orders)
the evaluation panel.

    python experiments/baselines/list_images.py --generation-dir out/tree_ring \
        --prompts prompts.jsonl --output images.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _baselines_lib as lib
from watermark.io import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for prompt in lib.read_prompts(args.prompts):
        panel = lib.read_panel(args.generation_dir, prompt["prompt_uid"])
        for role in lib.ROLES:
            rows.append({
                "image_id": lib.make_image_id(panel["prompt_uid"], role),
                "path": str(lib.panel_image_path(args.generation_dir, panel, args.slot, role).resolve()),
                "prompt": panel["prompt"],
            })
    write_jsonl(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
