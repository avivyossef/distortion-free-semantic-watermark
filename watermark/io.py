"""Small file helpers: SHA-256, canonical JSON, JSONL manifests, key files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def contained_path(root: Path, relative: Any) -> Path:
    """``root / relative`` resolved, refusing absolute paths and anything outside ``root``."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("invalid relative path %r" % (relative,))
    resolved = (Path(root) / relative).resolve()
    if Path(root).resolve() not in resolved.parents:
        raise ValueError("path %r escapes its root" % (relative,))
    return resolved


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of rows (blank lines skipped)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("missing required manifest: %s" % path)
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSONL file: %s" % path) from error


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as canonical JSON lines, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def read_key_file(path: Path) -> bytes:
    """Read a 32-byte master key file."""
    key = Path(path).read_bytes()
    if len(key) != 32:
        raise ValueError("key file must contain exactly 32 bytes")
    return key


def pixel_sha256(image: Any) -> str:
    """Hash of an image's RGB pixels (size included), independent of the file encoding."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(("RGB:%dx%d\0" % (rgb.width, rgb.height)).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()
