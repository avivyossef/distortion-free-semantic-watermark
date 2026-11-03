#!/usr/bin/env python3
"""Draw a fresh 32-byte key file (a generation-seed key or a detector master key).

    python experiments/generation/new_key.py --output keys/detector.key

The file is secret: never commit it. Refuses to overwrite an existing file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.prf import generate_master_key  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(args.output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(generate_master_key())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
