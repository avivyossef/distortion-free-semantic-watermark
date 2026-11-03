#!/usr/bin/env python3
"""Same-prompt diversity (generation quality and diversity table): relative Vendi score.

For every prompt the corpus holds ``S`` independent slots; one image per slot is chosen
(by the watermark selector, or uniformly for the clean reference) and the Vendi score of the
``S`` chosen images is computed from their frozen DINOv2-base embeddings, which are independent
of our semantic encoder. Our selection runs on the embeddings of our encoder, for every
configuration (N, M), hyperplane frame and PRF key of ``configs/detection.yaml``; the clean
reference is a uniform draw among the same N candidates. The relative Vendi score of a cell
(prompt, key, frame) is Vendi_wm / Vendi_clean and the reported change is
``100 (mean relative Vendi - 1)``. A native baseline contributes its watermarked and its own clean
images (one fixed key), one ratio per prompt.

    python experiments/quality/vendi.py --selection-cache cache_ours.npz --evaluation-cache cache_dino.npz \
        --key-file keys/detector.key --native-embeddings tree_ring=embeddings/tree_ring \
        gaussian_shading=embeddings/gaussian_shading seal=embeddings/seal --output-dir results/vendi

CPU only. The two caches are built from the same corpus (``experiments/generation``) with
``--encoder ours`` and ``--encoder dinov2_base``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402

from detection_corpus import frame_planes, load_cache  # noqa: E402

from watermark.config import load_detection_config  # noqa: E402


def slot_vendi(embeddings: np.ndarray, positions: np.ndarray) -> float:
    """Vendi score of one prompt's chosen images; ``embeddings`` is ``(S, N, D)``."""
    chosen = embeddings[np.arange(len(positions)), positions - 1]
    return lib.cosine_vendi_score(chosen)


def ours_cells(selection: np.ndarray, evaluation: np.ndarray, planes: dict, keys: dict, *, n: int, m: int,
               frame_id: str, prompt_uids) -> list:
    """Vendi of the selected and of the uniform images for every (prompt, key) of one frame."""
    slots = selection.shape[1]
    controls = [slot_vendi(evaluation[p], lib.uniform_positions(uid, frame_id, n, slots))
                for p, uid in enumerate(prompt_uids)]
    rows = []
    for trial in sorted(keys):
        positions = lib.select_positions(selection, planes[frame_id], keys[trial], frame_id=frame_id, key_trial=trial,
                                         n=n, m=m, prompt_uids=prompt_uids)
        for p, uid in enumerate(prompt_uids):
            vendi = slot_vendi(evaluation[p], positions[p])
            rows.append({"N": n, "M": m, "frame": frame_id, "key": trial, "prompt_uid": uid, "vendi": vendi,
                         "control_vendi": controls[p], "relative_vendi": vendi / controls[p]})
    return rows


def summarize(rows: list) -> dict:
    """Mean relative Vendi over cells and the corresponding percentage change."""
    ratio = float(np.mean([row["relative_vendi"] for row in rows]))
    return {"mean_vendi": float(np.mean([row["vendi"] for row in rows])),
            "mean_control_vendi": float(np.mean([row["control_vendi"] for row in rows])),
            "mean_relative_vendi": ratio, "change_percent": 100.0 * (ratio - 1.0), "cells": len(rows)}


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection-cache", type=Path, required=True, help="embedding cache, encoder ours")
    parser.add_argument("--evaluation-cache", type=Path, required=True, help="embedding cache, encoder dinov2_base")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--native-embeddings", nargs="*", default=[], metavar="NAME=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection, evaluation = load_cache(args.selection_cache), load_cache(args.evaluation_cache)
    uids = list(selection.prompt_uids)
    evaluation = evaluation.subset(uids)
    planes = frame_planes(cfg, selection.candidate_vectors.shape[-1])
    keys = lib.trial_keys(lib.read_master_key(args.key_file), cfg.keys)
    cell_rows, summary_rows = [], []
    for setting in cfg.settings:
        rows = []
        for frame_id in cfg.frame_ids():
            rows.extend(ours_cells(selection.candidate_vectors, evaluation.candidate_vectors, planes, keys,
                                   n=setting.n, m=setting.m, frame_id=frame_id, prompt_uids=uids))
        cell_rows.extend(rows)
        summary_rows.append({"method": "ours", "N": setting.n, "M": setting.m, **summarize(rows)})
        print("N=%d M=%d done" % (setting.n, setting.m), flush=True)
    for name, directory in sorted(lib.parse_named_paths(args.native_embeddings).items()):
        shards = lib.load_shards(directory, ["watermarked", "cleans"], uids, prefix="embeddings")
        rows = []
        for p, uid in enumerate(uids):
            vendi, control = lib.cosine_vendi_score(shards["watermarked"][p]), lib.cosine_vendi_score(shards["cleans"][p])
            rows.append({"prompt_uid": uid, "vendi": vendi, "control_vendi": control, "relative_vendi": vendi / control})
        cell_rows.extend({"method": name, **row} for row in rows)
        summary_rows.append({"method": name, **summarize(rows)})
    lib.write_csv(args.output_dir / "cells.csv", cell_rows)
    lib.write_csv(args.output_dir / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
