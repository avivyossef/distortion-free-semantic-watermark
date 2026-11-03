#!/usr/bin/env python3
"""Generate the GPT Image panel: N candidates and one clean control per prompt.

Input ``--prompts`` is a JSONL file with one row per prompt of the held-out registry::

    {"prompt_uid": "...", "text": "..."}

The first 50 prompts, in file order, are used. Every candidate and every clean control is a
separate API request for a single low-quality 1024 x 1024 image. N is the ``closed_api``
candidate count of ``configs/detection.yaml``. The API key is read from the
``OPENAI_API_KEY`` environment variable only. Finished images are not requested again, so an
interrupted run can be repeated. Provider request identifiers are not stored.

    export OPENAI_API_KEY=...
    python experiments/closed_api/generate_panel.py --prompts prompts.jsonl --output-dir panel
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from closed_api_lib import MODEL_ID, PROMPT_COUNT, generate_image, generate_panel  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    candidate_count = load_detection_config().exception("closed_api")["N"]
    prompts = read_jsonl(args.prompts)[:PROMPT_COUNT]
    if len(prompts) != PROMPT_COUNT:
        raise SystemExit("the prompt registry needs at least %d prompts" % PROMPT_COUNT)
    completed = 0

    def call_api(request):
        nonlocal completed
        image = generate_image(api_key=api_key, model_id=MODEL_ID, prompt=request.prompt,
                               request_uid=request.request_uid)
        completed += 1
        print("generated %d %s %s" % (completed, request.prompt_uid, request.role), file=sys.stderr)
        return image

    generate_panel(prompts=prompts, output_root=args.output_dir, model_id=MODEL_ID,
                   candidate_count=candidate_count, generate=call_api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
