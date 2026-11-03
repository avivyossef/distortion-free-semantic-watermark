#!/usr/bin/env python3
"""Montage of one prompt's eight independent GPT Image candidates (appendix figure).

Reconstruction from the paper's caption: the eight candidates of one prompt in a 2 x 4 grid.
``--prompt-uid`` is the identity of the prompt in the registry.

    python experiments/closed_api/umbrella_grid.py --panel-root panel --prompt-uid <uid> \
        --output figures/closed_api_umbrella_grid.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

TILE = 512
COLUMNS = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--prompt-uid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.panel_root / args.prompt_uid
    record = json.loads((root / "panel.json").read_text(encoding="utf-8"))
    candidates = sorted(record["candidate_images"], key=lambda row: row["candidate_position"])
    rows = (len(candidates) + COLUMNS - 1) // COLUMNS
    canvas = Image.new("RGB", (COLUMNS * TILE, rows * TILE), "white")
    for index, item in enumerate(candidates):
        with Image.open(root / item["relative_path"]) as opened:
            tile = opened.convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)
        canvas.paste(tile, ((index % COLUMNS) * TILE, (index // COLUMNS) * TILE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
