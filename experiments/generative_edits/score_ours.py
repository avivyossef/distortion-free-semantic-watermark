#!/usr/bin/env python3
"""Detection of our method after generative edits, one row per (configuration, frame, key).

For each replicate of ``configs/detection.yaml`` the detector is fixed on the unedited calibration
images: candidate selection, the per-image boundary threshold delta (cross-fitted on the selected
watermarked margins) and the score threshold tau (from the unedited clean scores at each FPR
target of the config). The edited watermarked image of each panel prompt is the edit of the image
the replicate selected; the edited unwatermarked image is the edit of the clean control. Both
are scored with the same keyed table and delta, and tau is not recalibrated.

    python experiments/generative_edits/score_ours.py --panel edits/panel.jsonl --prompts prompts.jsonl \
        --cache embeddings.npz --corpus-dir corpus --key-file keys/detector.key \
        --edited-embeddings edits/edited_embeddings.npz --output-dir edits/ours

Writes ``replicates.csv``: ``method, frame_id, key_trial, fpr_target, category`` and the metrics
(``threshold, calibration_fpr, original_tpr, edited_tpr, retention, edited_fpr, edits``).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import (category_metrics, edited_scores, load_cache, ours_panel, ours_replicates,  # noqa: E402
                        panel_rows_in_cache, read_prompts, restrict_cache)

from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_jsonl, read_key_file  # noqa: E402


def load_edited_vectors(path: Path) -> dict:
    with np.load(str(path), allow_pickle=False) as data:
        return {str(source_id): vector for source_id, vector in zip(data["source_ids"].tolist(), data["vectors"])}


def score_replicates(panel_rows, cache, corpus_dir, cfg, master_key, edited_vectors):
    """Per-replicate metric rows of our method over the panel."""
    rows = panel_rows_in_cache(panel_rows, cache)
    categories = [row["edit_category"] for row in panel_rows]
    images = [ours_panel(corpus_dir, row["prompt_uid"]) for row in panel_rows]
    clean_ids = [item["clean"][0] for item in images]
    missing = [sha for sha in clean_ids if sha not in edited_vectors]
    if missing:
        raise ValueError("%d clean controls have no edited embedding" % len(missing))
    clean_vectors = np.stack([edited_vectors[sha] for sha in clean_ids])
    output = []
    for setting, frame_id, trial, planes, scores in ours_replicates(cache, cfg, master_key):
        chosen = scores.selected_index[rows]
        wm_ids = [item["candidates"][int(index)][0] for item, index in zip(images, chosen)]
        if any(sha not in edited_vectors for sha in wm_ids):
            raise ValueError("a selected watermarked image has no edited embedding; rerun build_sources.py")
        edited_wm = edited_scores(scores, planes, setting, rows, np.stack([edited_vectors[sha] for sha in wm_ids]))
        edited_clean = edited_scores(scores, planes, setting, rows, clean_vectors)
        original = scores.watermarked.scores[rows]
        for fpr in cfg.fpr_targets:
            metrics = category_metrics(scores.clean.scores, categories, original, edited_wm, edited_clean, fpr)
            for category, values in metrics.items():
                output.append({"method": "ours_n%d_m%d" % (setting.n, setting.m), "frame_id": frame_id,
                               "key_trial": trial, "fpr_target": fpr, "category": category, **values})
        print("scored N=%d M=%d %s key %d" % (setting.n, setting.m, frame_id, trial), flush=True)
    return output


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--edited-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = restrict_cache(load_cache(args.cache), [row["prompt_uid"] for row in read_prompts(args.prompts)])
    rows = score_replicates(read_jsonl(args.panel), cache, args.corpus_dir, cfg, read_key_file(args.key_file),
                            load_edited_vectors(args.edited_embeddings))
    write_csv(args.output_dir / "replicates.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
