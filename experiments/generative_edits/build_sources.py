#!/usr/bin/env python3
"""List the distinct source images to edit: the unwatermarked and the watermarked image of every
panel prompt, for our method and for each native baseline.

For our method the watermarked image depends on the configuration, hyperplane frame and PRF key
(the paper averages five frames x five keys), so every replicate of ``configs/detection.yaml`` is
run over the calibration prompts and the image it selects for each panel prompt is added; an image
selected by several replicates is edited once. The unwatermarked control of our method is the clean
image of slot 1. For a baseline the sources are the slot-1 watermarked and clean images of its
generation run.

    python experiments/generative_edits/build_sources.py --panel edits/panel.jsonl \
        --prompts prompts.jsonl --cache embeddings.npz --corpus-dir corpus --key-file keys/detector.key \
        --baseline tree_ring=baselines/tree_ring --baseline gaussian_shading=baselines/gaussian_shading \
        --baseline seal=baselines/seal --output-dir edits

``--cache`` is the slot-1 embedding cache of the candidate corpus (``experiments/generation``).
Writes ``sources.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import (baseline_panel, load_cache, merge_sources, ours_panel, ours_replicates,  # noqa: E402
                        panel_rows_in_cache, read_prompts, restrict_cache, write_rows)

from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_jsonl, read_key_file  # noqa: E402


def ours_entries(panel_rows, corpus_dir, cache, cfg, master_key):
    """Source entries of our method: clean controls and every image selected by some replicate."""
    rows = panel_rows_in_cache(panel_rows, cache)
    images = {row["prompt_uid"]: ours_panel(corpus_dir, row["prompt_uid"]) for row in panel_rows}
    entries = []
    for row in panel_rows:
        sha, path = images[row["prompt_uid"]]["clean"]
        entries.append({**_identity(row), "source_id": sha, "path": path, "role": "ours_clean"})
    for setting, frame_id, trial, _planes, scores in ours_replicates(cache, cfg, master_key):
        for row, selected in zip(panel_rows, scores.selected_index[rows]):
            sha, path = images[row["prompt_uid"]]["candidates"][int(selected)]
            entries.append({**_identity(row), "source_id": sha, "path": path, "role": "ours_watermarked"})
        print("replicate N=%d M=%d %s key %d" % (setting.n, setting.m, frame_id, trial), flush=True)
    return entries


def baseline_entries(panel_rows, method, baseline_dir):
    entries = []
    for row in panel_rows:
        for role, (sha, path) in baseline_panel(baseline_dir, row["prompt_uid"]).items():
            entries.append({**_identity(row), "source_id": sha, "path": path, "role": "%s_%s" % (method, role)})
    return entries


def _identity(row):
    return {"prompt_uid": row["prompt_uid"], "text": row["text"], "edit_category": row["edit_category"]}


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--baseline", action="append", default=[], metavar="METHOD=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel_rows = read_jsonl(args.panel)
    cache = restrict_cache(load_cache(args.cache), [row["prompt_uid"] for row in read_prompts(args.prompts)])
    entries = ours_entries(panel_rows, args.corpus_dir, cache, cfg, read_key_file(args.key_file))
    for item in args.baseline:
        method, _, directory = item.partition("=")
        entries.extend(baseline_entries(panel_rows, method, Path(directory)))
    sources = merge_sources(entries)
    write_rows(args.output_dir / "sources.jsonl", sources)
    print("sources: %d distinct images" % len(sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
