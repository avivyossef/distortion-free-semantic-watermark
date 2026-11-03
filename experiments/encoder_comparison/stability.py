#!/usr/bin/env python3
"""Encoder stability under content-preserving transformations (appendix table).

Source images are 200 ImageNette / ImageWoof images (160-pixel versions). Each is transformed
into 16 variants whose severity increases gradually, for four families: JPEG compression,
Gaussian blur, resize-resample, and mild crop with color adjustment. A source's score is the
mean pairwise cosine similarity of its 16 variants' embeddings; a family's score is the
mean over the 200 sources.

``encode`` embeds all variants with one encoder and writes ``<output-dir>/per_block/<encoder>.csv``
(one cosine per source image and family) and ``<output-dir>/embeddings/<encoder>.npz`` (the
variant embeddings, reused by ``distinct_codes.py``). ``table`` collects the encoders into
``<output-dir>/stability_table.csv``.

    python experiments/encoder_comparison/stability.py encode --encoder ours \
        --checkpoint checkpoints/checkpoint-step-00019840.pt --imagenette-root data/imagenette2-160 \
        --imagewoof-root data/imagewoof2-160 --output-dir results/stability
    python experiments/encoder_comparison/stability.py table --output-dir results/stability
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison_lib import (  # noqa: E402
    ENCODERS, FAMILIES, FAMILY_LABELS, GROUP_SIZE, PROMPT_COUNT, block_mean_cosine, build_perturbation_blocks,
    load_variants,
)


def encode(args: argparse.Namespace) -> None:
    from watermark.embedding import load_encoder

    blocks = build_perturbation_blocks(args.imagenette_root, args.imagewoof_root)
    encoder = load_encoder(args.encoder, device=args.device, checkpoint=args.checkpoint)
    rows, arrays = [], {}
    for family in FAMILIES:
        values = [encoder.embed(load_variants(block)) for block in blocks[family]]
        arrays[family] = np.asarray(values, dtype=np.float32)
        for block, block_values in zip(blocks[family], values):
            rows.append({"family": family, "block_id": block.block_id,
                         "mean_cosine": block_mean_cosine(block_values, block_size=GROUP_SIZE)})
    for directory in ("per_block", "embeddings"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_block" / ("%s.csv" % args.encoder)).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("family", "block_id", "mean_cosine"))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "embeddings" / ("%s.npz" % args.encoder), **arrays)


def table(args: argparse.Namespace) -> None:
    means = {}
    for encoder in ENCODERS:
        with (args.output_dir / "per_block" / ("%s.csv" % encoder)).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for family in FAMILIES:
            values = [float(row["mean_cosine"]) for row in rows if row["family"] == family]
            if len(values) != PROMPT_COUNT:
                raise ValueError("expected %d sources for %s / %s" % (PROMPT_COUNT, encoder, family))
            means[(family, encoder)] = float(np.mean(values))
    with (args.output_dir / "stability_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", *ENCODERS])
        for family in FAMILIES:
            writer.writerow([FAMILY_LABELS[family], *("%.3f" % means[(family, encoder)] for encoder in ENCODERS)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("encode")
    one.add_argument("--encoder", choices=ENCODERS, required=True)
    one.add_argument("--checkpoint", type=Path, help="trained encoder checkpoint (needed for --encoder ours)")
    one.add_argument("--imagenette-root", type=Path, required=True)
    one.add_argument("--imagewoof-root", type=Path, required=True)
    one.add_argument("--output-dir", type=Path, required=True)
    one.add_argument("--device", default="cuda")
    one.set_defaults(run=encode)
    collect = commands.add_parser("table")
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.set_defaults(run=table)
    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
