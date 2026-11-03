#!/usr/bin/env python3
"""List the images a native baseline detector must score (input of ``baselines/score.py``).

Four groups of images of one baseline method, identified by ``image_id``:

* ``calibration-clean:<uid>``  unedited clean images of every calibration prompt (freeze tau),
* ``original:<uid>``           unedited watermarked images of the panel prompts,
* ``edited-watermarked:<uid>`` their generative edits,
* ``edited-clean:<uid>``       edits of the corresponding unwatermarked images.

    python experiments/generative_edits/prepare_baseline_scoring.py --method tree_ring \
        --baseline-dir baselines/tree_ring --prompts prompts.jsonl --panel edits/panel.jsonl \
        --render-dir edits/rendered --output edits/tree_ring_images.jsonl

``--baseline-dir`` is the ``generate.py`` output of the method over the calibration prompts.
Then run the baseline detector on the list (see the README for the environment):
``python experiments/baselines/score.py --method tree_ring ... --images edits/tree_ring_images.jsonl
--output edits/tree_ring_scores.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import baseline_panel, read_prompts, read_shards, write_rows  # noqa: E402

from watermark.io import read_jsonl  # noqa: E402


def baseline_images(method_dir, prompts, panel_rows, edited_paths):
    """Rows ``{"image_id", "path", "prompt"}`` of the four groups; ``edited_paths`` maps source id to path."""
    rows = []
    for row in prompts:
        sha, path = baseline_panel(method_dir, row["prompt_uid"])["clean"]
        rows.append({"image_id": "calibration-clean:%s" % row["prompt_uid"], "path": str(path),
                     "prompt": row["text"]})
    for row in panel_rows:
        images = baseline_panel(method_dir, row["prompt_uid"])
        uid, text = row["prompt_uid"], row["text"]
        rows.append({"image_id": "original:%s" % uid, "path": str(images["watermarked"][1]), "prompt": text})
        for group, role in (("edited-watermarked", "watermarked"), ("edited-clean", "clean")):
            sha = images[role][0]
            if sha not in edited_paths:
                raise ValueError("no edit of the %s image of %s" % (role, uid))
            rows.append({"image_id": "%s:%s" % (group, uid), "path": str(edited_paths[sha]), "prompt": text})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=("tree_ring", "gaussian_shading", "seal"))
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    edited = {row["source_id"]: args.render_dir / row["edited_path"]
              for row in read_shards(args.render_dir, "manifest-shard-*.jsonl")}
    rows = baseline_images(args.baseline_dir, read_prompts(args.prompts), read_jsonl(args.panel), edited)
    write_rows(args.output, rows)
    print("%s: %d images to score" % (args.method, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
