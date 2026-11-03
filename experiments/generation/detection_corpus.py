"""Shared helpers of the detection experiments: prompts, panels, embedding caches, replicate grid.

The experiments are plain scripts. The ones that read an embedding cache add this directory
to ``sys.path`` and ``import detection_corpus as corpus``.

Embedding cache (``.npz``)::

    prompt_uids        (P,)            sorted prompt identities
    candidate_vectors  (P, S, N, D)    unit-norm float64: S slots of N candidate embeddings
    clean_vectors      (P, S, D)       the unwatermarked control of each slot
    metadata           JSON text       {"version", "encoder", "panel_sha256", "checkpoint_sha256"}
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.coding import sample_hyperplanes  # noqa: E402
from watermark.config import DetectionConfig  # noqa: E402
from watermark.io import contained_path, read_jsonl, sha256_file  # noqa: E402
from watermark.prf import derive_trial_key  # noqa: E402

CACHE_VERSION = "sd21-embedding-cache/v1"
ENCODERS = ("ours", "dinov2_base", "clip_vit_b32", "dinov3_l16", "tpips_embed")


def read_prompts(path: Path) -> List[Dict[str, Any]]:
    """Read a prompt list: JSONL rows ``{"prompt_uid": ..., "text": ...}``, sorted by identity."""
    rows = sorted(read_jsonl(Path(path)), key=lambda row: str(row["prompt_uid"]))
    uids = [str(row["prompt_uid"]) for row in rows]
    if not rows or len(set(uids)) != len(uids) or any(not str(row.get("text", "")).strip() for row in rows):
        raise ValueError("prompts must be unique and nonempty")
    return rows


def deal(items: Sequence[Any], shard_index: int, num_shards: int) -> list:
    """Round-robin shard of an ordered list."""
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= I < --num-shards")
    return list(items)[shard_index::num_shards]


def read_panel(corpus_dir: Path, prompt_uid: str) -> Dict[str, Any]:
    """The ``panel.json`` record of one prompt (see :mod:`watermark.generation`)."""
    panel = json.loads((Path(corpus_dir) / prompt_uid / "panel.json").read_text(encoding="utf-8"))
    if panel.get("mode") != "candidates" or panel.get("prompt_uid") != prompt_uid:
        raise ValueError("unexpected panel record for %s" % prompt_uid)
    return panel


def panel_images(corpus_dir: Path, panel: Dict[str, Any], slots: Optional[int] = None
                 ) -> "tuple[list[list[Path]], list[Path]]":
    """Candidate image paths per slot (ordered by position) and the clean image of each slot."""
    root = Path(corpus_dir) / panel["prompt_uid"]
    chosen = sorted(panel["batch_slots"], key=lambda slot: int(slot["slot"]))
    chosen = chosen if slots is None else chosen[:slots]
    candidates = [[contained_path(root, item["relative_path"]) for item in
                   sorted(slot["candidate_images"], key=lambda item: int(item["candidate_position"]))]
                  for slot in chosen]
    return candidates, [contained_path(root, slot["clean"]["relative_path"]) for slot in chosen]


def panel_manifest_sha256(corpus_dir: Path, prompt_uids: Iterable[str]) -> str:
    """Commit to the exact ``panel.json`` files the vectors were computed from."""
    digest = hashlib.sha256()
    for prompt_uid in sorted(prompt_uids):
        digest.update(prompt_uid.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256((Path(corpus_dir) / prompt_uid / "panel.json").read_bytes()).digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class EmbeddingCache:
    prompt_uids: Tuple[str, ...]
    candidate_vectors: np.ndarray  # (P, S, N, D)
    clean_vectors: np.ndarray  # (P, S, D)
    metadata: Dict[str, Any]

    def index_of(self, prompt_uid: str) -> int:
        try:
            return self.prompt_uids.index(prompt_uid)
        except ValueError as error:
            raise ValueError("embedding cache does not contain prompt %r" % prompt_uid) from error

    def subset(self, prompt_uids: Sequence[str]) -> "EmbeddingCache":
        """The cache restricted to ``prompt_uids`` (in the given order)."""
        rows = np.asarray([self.index_of(uid) for uid in prompt_uids], dtype=np.int64)
        return EmbeddingCache(tuple(prompt_uids), self.candidate_vectors[rows], self.clean_vectors[rows],
                              self.metadata)


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("embeddings must be finite and nonzero")
    return values / norms


def write_cache(path: Path, prompt_uids: Sequence[str], candidate_vectors: np.ndarray,
                clean_vectors: np.ndarray, metadata: Dict[str, Any]) -> None:
    """Atomically write an embedding cache (rows sorted by prompt identity)."""
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite %s" % path)
    order = np.argsort(np.asarray(prompt_uids))
    uids = np.asarray(prompt_uids)[order]
    if len(set(uids.tolist())) != len(uids):
        raise ValueError("prompt identities must be unique")
    candidates, cleans = np.asarray(candidate_vectors)[order], np.asarray(clean_vectors)[order]
    if candidates.ndim != 4 or cleans.shape != candidates.shape[:2] + candidates.shape[3:]:
        raise ValueError("candidate and clean embedding shapes disagree")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, prompt_uids=uids, candidate_vectors=_unit(candidates), clean_vectors=_unit(cleans),
                metadata=np.asarray(json.dumps({"version": CACHE_VERSION, **metadata}, sort_keys=True)))
        os.replace(temporary, str(path))
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def load_cache(path: Path) -> EmbeddingCache:
    """Load an embedding cache and validate its layout."""
    with np.load(str(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        uids = tuple(str(value) for value in data["prompt_uids"].tolist())
        candidates = np.asarray(data["candidate_vectors"], dtype=np.float64)
        cleans = np.asarray(data["clean_vectors"], dtype=np.float64)
    if metadata.get("version") != CACHE_VERSION:
        raise ValueError("not an embedding cache: %s" % path)
    if candidates.ndim != 4 or cleans.shape != candidates.shape[:2] + candidates.shape[3:] \
            or candidates.shape[0] != len(uids) or len(set(uids)) != len(uids):
        raise ValueError("inconsistent embedding cache: %s" % path)
    return EmbeddingCache(uids, candidates, cleans, metadata)


def load_prompt_cache(cache_path: Path, prompts_path: Optional[Path] = None) -> EmbeddingCache:
    """Load a cache, optionally restricted to the prompts of a prompt list."""
    cache = load_cache(cache_path)
    if prompts_path is None:
        return cache
    return cache.subset([str(row["prompt_uid"]) for row in read_prompts(prompts_path)])


def replicate_cells(cfg: DetectionConfig) -> "list[tuple[str, int, int, int]]":
    """Every (frame id, frame seed, key trial) replicate of the paper: frames x keys."""
    return [(frame_id, seed, trial)
            for frame_id, seed in zip(cfg.frame_ids(), cfg.frame_seeds())
            for trial in cfg.key_trials()]


def frame_planes(cfg: DetectionConfig, dimension: int) -> Dict[str, np.ndarray]:
    """Hyperplane frames regenerated from their seeds, keyed by frame id."""
    return {frame_id: sample_hyperplanes(seed, dimension)
            for frame_id, seed in zip(cfg.frame_ids(), cfg.frame_seeds())}


def trial_keys(master_key: bytes, cfg: DetectionConfig) -> Dict[int, bytes]:
    """PRF key of every key trial."""
    return {trial: derive_trial_key(master_key, trial) for trial in cfg.key_trials()}


def read_master_key(path: Path) -> bytes:
    key = Path(path).read_bytes()
    if len(key) != 32:
        raise ValueError("key file must contain exactly 32 bytes")
    return key


def write_jsonl_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
