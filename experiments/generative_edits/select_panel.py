#!/usr/bin/env python3
"""Choose the generative-edit panel: ``--per-category`` prompts for each of the four edit categories.

Prompts are ranked by a hash of their identity (never by input order or detector outcomes), so
the panel is reproducible. ``--prompts`` is the calibration prompt list (JSONL rows
``{"prompt_uid", "text"}``): the unedited images of all its prompts calibrate the detectors, and
the edited prompts are drawn from it.

    python experiments/generative_edits/select_panel.py --prompts prompts.jsonl \
        --per-category 20 --output-dir edits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _edits_lib import balanced_panel, read_prompts, write_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--per-category", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = balanced_panel(read_prompts(args.prompts), per_category=args.per_category)
    rows = [{"prompt_uid": row["prompt_uid"], "text": row["text"], "edit_category": category}
            for row, category in panel]
    rows.sort(key=lambda row: (row["edit_category"], row["prompt_uid"]))
    write_rows(args.output_dir / "panel.jsonl", rows)
    print("panel: %d prompts" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
