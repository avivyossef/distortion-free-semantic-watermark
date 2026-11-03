#!/usr/bin/env python3
"""Illustrate the six transformations of the style / depth study on one image (appendix figure).

Reconstruction from the paper's caption. Top row: the unedited image and its oil painting,
watercolor and pencil sketch renders. Bottom row: the unedited image, the 32-pixel
translation, the 75% off-center crop and the horizontal flip.

    python experiments/style_depth_encoders/edit_figure.py --panel-root panels/sd21 \
        --edit-dir edits --prompt-uid <uid> --candidate-position 1 --output figures/edits.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from style_depth_lib import EDIT_LABELS, SLOT_CANDIDATES, SPATIAL_EDITS, STYLE_EDITS, source_paths, spatial_edit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--edit-dir", type=Path, required=True)
    parser.add_argument("--prompt-uid", required=True)
    parser.add_argument("--candidate-position", type=int, default=1, help="candidate image, 1..16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.candidate_position <= SLOT_CANDIDATES:
        raise SystemExit("--candidate-position must lie in 1..%d" % SLOT_CANDIDATES)
    import matplotlib.pyplot as plt
    from PIL import Image

    index = args.candidate_position - 1
    with Image.open(source_paths(args.panel_root, args.prompt_uid)[index]) as opened:
        original = opened.convert("RGB").copy()
    top = [("Original", original)]
    for edit_id in STYLE_EDITS:
        with Image.open(args.edit_dir / edit_id / args.prompt_uid / ("%02d.png" % index)) as opened:
            top.append((EDIT_LABELS[edit_id], opened.convert("RGB").copy()))
    bottom = [("Original", original)] + [(EDIT_LABELS[edit_id], spatial_edit(original, edit_id))
                                         for edit_id in SPATIAL_EDITS]
    figure, axes = plt.subplots(2, 4, figsize=(8, 4.4))
    for row, panels in zip(axes, (top, bottom)):
        for axis, (title, image) in zip(row, panels):
            axis.imshow(image)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
