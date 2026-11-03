#!/usr/bin/env python3
"""Embed the unedited images and the six edits of every candidate and clean image.

Reads the first 100 prompts of the SD2.1 panel (``--panel-root``), the InstructPix2Pix renders
of ``render_edits.py`` (``--edit-dir``) and applies the three spatial transformations here.
Writes ``--output`` with, for each encoder (``csd_style``, ``depth_layout``), the arrays
``<encoder>__original`` and ``<encoder>__<edit>`` of shape ``(100, 17, d)`` (rows 0-15
candidates, row 16 the clean image), and ``prompt_uids``.

    python experiments/style_depth_encoders/embed_edits.py --panel-root panels/sd21 \
        --edit-dir edits --output embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from style_depth_lib import ENCODERS, SLOT_CANDIDATES, SPATIAL_EDITS, STYLE_EDITS, panel_prompt_uids, source_paths, spatial_edit  # noqa: E402
from watermark.embedding import load_encoder  # noqa: E402
from watermark.io import sha256_file  # noqa: E402


def load_style_edit(edit_dir: Path, edit_id: str, prompt_uid: str):
    from PIL import Image

    folder = edit_dir / edit_id / prompt_uid
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    images = []
    for record in manifest["records"]:
        path = folder / record["output"]
        if sha256_file(path) != record["output_sha256"]:
            raise ValueError("edited image hash mismatch: %s" % path)
        with Image.open(path) as opened:
            images.append(opened.convert("RGB").copy())
    if len(images) != SLOT_CANDIDATES + 1:
        raise ValueError("expected %d edited images for %s" % (SLOT_CANDIDATES + 1, prompt_uid))
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--edit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    from PIL import Image

    uids = panel_prompt_uids(args.panel_root)
    encoders = {name: load_encoder(name, device=args.device) for name in ENCODERS}
    names = ("original",) + tuple(STYLE_EDITS) + SPATIAL_EDITS
    collected = {(encoder, name): [] for encoder in ENCODERS for name in names}
    for index, uid in enumerate(uids, 1):
        sources = []
        for path in source_paths(args.panel_root, uid):
            with Image.open(path) as opened:
                sources.append(opened.convert("RGB").copy())
        for name in names:
            if name == "original":
                images = sources
            elif name in STYLE_EDITS:
                images = load_style_edit(args.edit_dir, name, uid)
            else:
                images = [spatial_edit(image, name) for image in sources]
            for encoder in ENCODERS:
                collected[(encoder, name)].append(encoders[encoder].embed(images))
        if index % 10 == 0 or index == 1:
            print("embedded %d/%d prompts" % (index, len(uids)), flush=True)
    arrays = {"%s__%s" % key: np.stack(values) for key, values in collected.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prompt_uids=np.array(uids), **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
