#!/usr/bin/env python3
"""Qualitative same-prompt example (appendix figure "Different images from one prompt").

Generates 16 SDXL images of one prompt (seeds ``--seed-base`` .. ``--seed-base + 15``) with the
pipeline of the encoder's training data, embeds them with DINOv2-base and the trained encoder, and
draws (a) the 16 images and (b) the two 16 x 16 pairwise-cosine matrices. The images are written to
``--output-dir/images``; the figure to ``--output-dir/semantic_entropy_examples_sdxl.png``.

    python experiments/encoder_comparison/entropy_example.py --prompt "a red bicycle leaning on a wall" \
        --checkpoint checkpoints/checkpoint-step-00019840.pt --output-dir figures/entropy_example
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GROUP_SIZE = 16
LABELS = (("dinov2_base", "DINOv2-base"), ("ours", "Ours"))


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarities of the rows of ``embeddings``."""
    values = np.asarray(embeddings, dtype=np.float64)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    return values @ values.T


def render_figure(images: Sequence, embeddings: dict, output: Path) -> None:
    """Draw the image grid (a) and the cosine matrices (b) of the given encoders."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10, 9))
    outer = figure.add_gridspec(2, 1, height_ratios=(1.15, 1), hspace=0.18)
    grid = outer[0].subgridspec(4, 4, hspace=0.04, wspace=0.04)
    for index, image in enumerate(images):
        axis = figure.add_subplot(grid[index // 4, index % 4])
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(str(index + 1), fontsize=7)
        if index == 0:
            axis.annotate("(a)", xy=(-0.1, 1.25), xycoords="axes fraction", fontsize=11, fontweight="bold")
    matrices = outer[1].subgridspec(1, len(embeddings), wspace=0.3)
    for column, (key, label) in enumerate(item for item in LABELS if item[0] in embeddings):
        axis = figure.add_subplot(matrices[0, column])
        shown = axis.imshow(cosine_matrix(embeddings[key]), vmin=-0.1, vmax=1.0, cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("image index")
        axis.set_ylabel("image index")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
        if column == 0:
            axis.annotate("(b)", xy=(-0.25, 1.1), xycoords="axes fraction", fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output), dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="a held-out prompt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained encoder checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from encoder.sdxl import load_sdxl_pipeline
    from watermark.embedding import load_encoder

    pipeline = load_sdxl_pipeline(args.device)
    images = [pipeline.final_and_preview(args.prompt, args.seed_base + index)[0] for index in range(GROUP_SIZE)]
    del pipeline
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(str(args.output_dir / "images" / ("seed_%02d.png" % (args.seed_base + index))))
    embeddings = {}
    for key, _ in LABELS:
        encoder = load_encoder(key, device=args.device, checkpoint=args.checkpoint)
        embeddings[key] = np.asarray(encoder.embed(images), dtype=np.float32)
        del encoder
    render_figure(images, embeddings, args.output_dir / "semantic_entropy_examples_sdxl.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
