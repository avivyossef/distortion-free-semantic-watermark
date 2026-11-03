#!/usr/bin/env python3
"""Ordinary-edit robustness of one native method: edited TPR, edited FPR, paired retention.

Inputs: ``score.py`` outputs for the unedited images (``list_images.py``) and for the 20 edited
versions (``attack_images.py``). The score threshold of every FPR target in the detection config
is frozen on the clean unedited scores only and not recalibrated on edited images. Writes
``per_edit.csv`` and ``categories.csv`` (unweighted means over the settings of each category) to
``--output-dir``.

    python experiments/baselines/robustness.py --scores scores_unedited.jsonl scores_edited.jsonl \
        --output-dir analysis/tree_ring_robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _baselines_lib as lib
from watermark.config import load_detection_config
from watermark.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table = lib.collect_scores(row for path in args.scores for row in read_jsonl(path))
    per_edit, categories = lib.analyze_robustness(table, load_detection_config().fpr_targets)
    lib.write_csv(args.output_dir / "per_edit.csv", per_edit)
    lib.write_csv(args.output_dir / "categories.csv", categories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
