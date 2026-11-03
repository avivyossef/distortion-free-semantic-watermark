#!/usr/bin/env python3
"""Original-edited cosine similarity of baseline images in our encoder's space (similarity figure).

For the baselines, our embedding serves only as an external diagnostic. Inputs are the image lists
of the baselines experiment: the unedited images (``experiments/baselines/list_images.py``) and
the edited images (``attack_images.py``); rows ``{"image_id": "<prompt_uid>|<role>|<edit_id>",
"path": ...}``. Each watermarked image is paired with its 20 edits.

    python experiments/robustness/edit_cosines.py --method "Tree-Ring" \
        --original-images out/tree_ring_images.jsonl --edited-images out/tree_ring_edited_images.jsonl \
        --checkpoint checkpoints/checkpoint-step-00019840.pt --output analysis/tree_ring_cosines.csv

Writes ``method, edit_id, cosine, pair_count``: the mean cosine over the prompts of each edit.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))

from detection_corpus import write_csv_rows  # noqa: E402

from watermark.embedding import embed_paths, load_encoder  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402


def watermarked_pairs(original_rows, edited_rows) -> list:
    """``(prompt_uid, edit_id, original path, edited path)`` for every edited watermarked image."""
    original = {}
    for row in original_rows:
        uid, role, edit = row["image_id"].rsplit("|", 2)
        if role == "watermarked" and edit == "identity":
            original[uid] = row["path"]
    pairs = []
    for row in edited_rows:
        uid, role, edit = row["image_id"].rsplit("|", 2)
        if role == "watermarked" and edit != "identity":
            pairs.append((uid, edit, original[uid], row["path"]))
    return pairs


def mean_cosines(method: str, edit_ids, original_vectors: np.ndarray, edited_vectors: np.ndarray) -> list:
    """Mean cosine per edit of row-aligned unit-norm embeddings."""
    cosine = np.sum(original_vectors * edited_vectors, axis=1)
    groups = defaultdict(list)
    for edit, value in zip(edit_ids, cosine):
        groups[str(edit)].append(float(np.clip(value, -1.0, 1.0)))
    return [{"method": method, "edit_id": edit, "cosine": float(np.mean(values)), "pair_count": len(values)}
            for edit, values in sorted(groups.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, help="method name written to the output")
    parser.add_argument("--original-images", type=Path, nargs="+", required=True)
    parser.add_argument("--edited-images", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pairs = watermarked_pairs([r for p in args.original_images for r in read_jsonl(p)],
                              [r for p in args.edited_images for r in read_jsonl(p)])
    encoder = load_encoder("ours", device=args.device, checkpoint=args.checkpoint)
    unique = sorted({path for pair in pairs for path in pair[2:]})
    vectors = dict(zip(unique, embed_paths(encoder, [Path(p) for p in unique])))
    write_csv_rows(args.output, mean_cosines(args.method, [pair[1] for pair in pairs],
                                             np.stack([vectors[pair[2]] for pair in pairs]),
                                             np.stack([vectors[pair[3]] for pair in pairs])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
