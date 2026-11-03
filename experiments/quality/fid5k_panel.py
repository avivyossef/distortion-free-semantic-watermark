#!/usr/bin/env python3
"""Build the COCO-2017 validation prompt panel of the FID against real images.

Every official validation image contributes exactly one caption (the one of smallest SHA-256
rank, unique across images), so the real images and the generated populations are matched
one-to-one. Writes a prompt list (``prompt_uid``, ``text`` and the real ``file_name``) that every
generation, embedding and activation script of the experiment takes as ``--prompts``.

    python experiments/quality/fid5k_panel.py --captions annotations/captions_val2017.json \
        --images-dir val2017 --output prompts_fid5k.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402

from watermark.io import write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=Path, required=True, help="COCO captions_val2017.json")
    parser.add_argument("--images-dir", type=Path, required=True, help="the validation JPEGs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = lib.build_fid5k_panel(json.loads(args.captions.read_text(encoding="utf-8")))
    missing = [row["file_name"] for row in rows if not (args.images_dir / row["file_name"]).is_file()]
    if missing:
        raise SystemExit("%d images are missing from %s, first %s" % (len(missing), args.images_dir, missing[0]))
    write_jsonl(args.output, rows)
    print("wrote %d prompts" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
