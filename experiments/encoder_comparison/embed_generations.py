#!/usr/bin/env python3
"""Embed a same-prompt panel (200 prompts x 16 images) with one encoder.

``--panel-root`` is a directory of prompt directories, each with a ``panel.json`` (the SD2.1
corpus written by ``experiments/generation`` or the FLUX panels of ``generate_flux.py``). The
first 200 prompt identities in sorted order are embedded. ``--output`` receives
``embeddings`` (200, 16, d) and ``prompt_uids``; a ``<output>.provenance.json`` records the
encoder and the prompt identities.

    python experiments/encoder_comparison/embed_generations.py --panel-root panels/flux_base \
        --encoder ours --checkpoint checkpoints/checkpoint-step-00019840.pt \
        --output embeddings/flux_base/ours.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison_lib import ENCODERS, GROUP_SIZE, PROMPT_COUNT, panel_image_paths, panel_prompt_uids  # noqa: E402
from watermark.embedding import embed_paths, load_encoder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--checkpoint", type=Path, help="trained encoder checkpoint (needed for --encoder ours)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    uids = panel_prompt_uids(args.panel_root)
    encoder = load_encoder(args.encoder, device=args.device, checkpoint=args.checkpoint)
    values = [embed_paths(encoder, panel_image_paths(args.panel_root, uid)) for uid in uids]
    array = np.asarray(values, dtype=np.float32)
    if array.shape[:2] != (PROMPT_COUNT, GROUP_SIZE) or not np.isfinite(array).all():
        raise ValueError("invalid embeddings shape %s" % (array.shape,))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=array, prompt_uids=np.array(uids))
    provenance = {"encoder": args.encoder, "encoder_provenance": getattr(encoder, "provenance", {}),
                  "panel_root": args.panel_root.name, "prompts": PROMPT_COUNT, "generations_per_prompt": GROUP_SIZE,
                  "prompt_uids_sha256": hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest()}
    Path(str(args.output) + ".provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
