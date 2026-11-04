#!/usr/bin/env python3
"""Render the 20 ordinary edits of every image the detector will see, and embed the results.

The evaluated images are the clean (unwatermarked) control of each prompt and every candidate
that candidate selection picks for some replicate (configuration x hyperplane frame x PRF key);
they are recomputed here from the embedding cache with ``watermark.selection`` so the edits
match the selections the analysis replays (Gumbel selection). Only slot 1 is used (B = 1).
Edits are ``watermark.attacks.EDIT_IDS``; the encoder is the trained encoder.

    python experiments/robustness/embed_ordinary_edits.py --corpus-dir corpus --cache cache.npz \
        --prompts prompts200.jsonl --key-file keys/detector.key --checkpoint checkpoint.pt \
        --output-dir edits --shard-index 0 --num-shards 1

``--prompts`` is the prompt list of the evaluation (200 prompts in the paper; a subset of the
cache). Images are dealt round-robin to shards; shard ``I`` writes ``edits-shard-III-of-KKK.npz``
(``source_ids``, ``edit_ids`` and ``embeddings`` of shape (sources, 20, D)). ``--compare-rules``
also edits the selections of the other sampling rules (the appendix rule comparison, which scores
representative sampling, Gumbel and Tournament).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _robustness_lib import SLOT, all_selected_sources  # noqa: E402
from detection_corpus import (deal, frame_planes, load_prompt_cache, panel_images, read_master_key,  # noqa: E402
                    read_panel, trial_keys)

from watermark.attacks import EDIT_IDS, render_edit  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402
from watermark.embedding import load_encoder  # noqa: E402
from watermark.selection_rules import RULES  # noqa: E402


def _image_paths(corpus_dir: Path, sources) -> dict:
    """Path of every source image (``<uid>:clean`` or ``<uid>:candidate:<position>``)."""
    paths, panels = {}, {}
    for sid in sources:
        uid = sid.split(":")[0]
        if uid not in panels:
            panels[uid] = panel_images(corpus_dir, read_panel(corpus_dir, uid), SLOT)
        candidates, cleans = panels[uid]
        paths[sid] = cleans[0] if sid.endswith(":clean") else candidates[0][int(sid.rsplit(":", 1)[1]) - 1]
    return paths


def embed_sources(encoder, paths, sources, batch_size: int = 16) -> np.ndarray:
    """Embeddings of all 20 edits of each source: ``(len(sources), 20, D)``."""
    from PIL import Image

    out = []
    for sid in sources:
        with Image.open(paths[sid]) as opened:
            image = opened.convert("RGB").copy()
        edited = [render_edit(image, edit_id) for edit_id in EDIT_IDS]
        out.append(np.concatenate([encoder.embed(edited[i:i + batch_size])
                                   for i in range(0, len(edited), batch_size)]))
    return np.stack(out)


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare-rules", action="store_true",
                        help="edit the selections of every sampling rule (sampling-rule comparison)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cache = load_prompt_cache(args.cache, args.prompts)
    planes = frame_planes(cfg, cache.candidate_vectors.shape[-1])
    keys = trial_keys(read_master_key(args.key_file), cfg)
    sources = deal(all_selected_sources(cache, cfg, planes, keys, RULES if args.compare_rules else ("gumbel",)), args.shard_index, args.num_shards)
    encoder = load_encoder("ours", device=args.device, checkpoint=args.checkpoint)
    embeddings = embed_sources(encoder, _image_paths(args.corpus_dir, sources), sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.output_dir / ("edits-shard-%03d-of-%03d.npz" % (args.shard_index, args.num_shards))),
        source_ids=np.asarray(sources), edit_ids=np.asarray(EDIT_IDS), embeddings=embeddings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
