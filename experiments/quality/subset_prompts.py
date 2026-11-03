#!/usr/bin/env python3
"""Take the first ``--count`` prompts, in sorted order of their identity, from a prompt list.

The same-prompt diversity panel uses a small set of the evaluation prompts; this fixes the set
deterministically from the evaluation prompt list.

    python experiments/quality/subset_prompts.py --prompts prompts.jsonl --count 60 --output prompts_diversity.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402

from watermark.io import write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = lib.read_prompts(args.prompts)
    if not 0 < args.count <= len(rows):
        raise SystemExit("--count must lie between 1 and %d" % len(rows))
    write_jsonl(args.output, rows[:args.count])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
