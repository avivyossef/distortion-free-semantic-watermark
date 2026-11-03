#!/usr/bin/env python3
"""Clean-detection ROC and operating points of one native method (single image, B = 1).

Input: ``score.py`` outputs for the unedited images listed by ``list_images.py`` (watermarked
and clean images of the same prompts). The FPR targets come from the detection config. Writes
``summary.csv`` (AUROC), ``operating_points.csv`` (TPR at each empirical FPR target, read from
the ROC curve) and ``roc.csv`` to ``--output-dir``.

    python experiments/baselines/clean_detection.py --scores scores.jsonl --output-dir analysis/tree_ring
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
    rows = [row for path in args.scores for row in read_jsonl(path)]
    table = lib.collect_scores(rows)
    marked, clean = table[(lib.IDENTITY, "watermarked")], table[(lib.IDENTITY, "clean")]
    if sorted(marked) != sorted(clean):
        parser.error("watermarked and clean scores must cover the same prompts")
    uids = sorted(marked)
    result = lib.clean_detection([clean[u] for u in uids], [marked[u] for u in uids],
                                 load_detection_config().fpr_targets)
    methods = sorted({row["method"] for row in rows})
    lib.write_csv(args.output_dir / "summary.csv", [{"method": ",".join(methods), "prompts": result["sample_count"],
                                                    "auroc": result["auc"]}])
    lib.write_csv(args.output_dir / "operating_points.csv", result["fixed_fpr"])
    lib.write_csv(args.output_dir / "roc.csv", result["roc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
