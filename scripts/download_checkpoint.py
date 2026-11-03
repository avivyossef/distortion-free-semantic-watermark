#!/usr/bin/env python3
"""Download a released asset (the encoder checkpoint or the prompt list) and verify its SHA-256.

The URL and expected hash come from configs/assets.yaml. The file is written to
``--output-dir`` only after the hash matches. Usage::

    python scripts/download_checkpoint.py --output-dir checkpoints
    python scripts/download_checkpoint.py --asset prompts --output-dir prompts
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watermark.config import load_assets  # noqa: E402

TIMEOUT_SECONDS = 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset", choices=("checkpoint", "prompts"), default="checkpoint")
    parser.add_argument("--assets", type=Path, default=None, help="alternative assets.yaml")
    args = parser.parse_args()
    checkpoint = load_assets(args.assets)[args.asset]
    url, expected = checkpoint["url"], checkpoint["sha256"]
    if not isinstance(url, str) or not url.startswith("https://"):
        raise SystemExit("configs/assets.yaml needs an https:// %s URL (got %r)" % (args.asset, url))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / checkpoint["file"]
    handle, temporary = tempfile.mkstemp(dir=str(args.output_dir), suffix=".part")
    digest = hashlib.sha256()
    try:
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != expected:
            raise SystemExit("SHA-256 mismatch: expected %s, got %s" % (expected, digest.hexdigest()))
        os.replace(temporary, str(destination))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("wrote %s (sha256 %s)" % (destination, expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
