"""Shared helpers of the quality experiments: population selection, FID, Vendi, activation shards.

The experiments are plain scripts that add this directory to ``sys.path`` and import this
module. Populations are described by *positions*: for a candidate corpus with ``P`` prompts,
``S`` slots and ``N`` candidates per slot, a population is an integer array of shape ``(P, S)``
holding, for every prompt and slot, the 1-based position of the selected candidate.

Shards (``<prefix>-shard-III-of-KKK.npz``) hold ``prompt_uids`` and one array per
image role, aligned to the prompts: ``candidates`` ``(p, S, N, 2048)`` and ``cleans``
``(p, S, 2048)`` for a candidate corpus; ``watermarked`` and ``cleans`` ``(p, S, 2048)`` for a
native-baseline corpus; ``images`` ``(p, 2048)`` for the real COCO images.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PUBLIC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUBLIC))
sys.path.insert(0, str(PUBLIC / "experiments" / "generation"))

from detection_corpus import deal, panel_images, read_master_key, read_panel, read_prompts  # noqa: E402,F401

from watermark.coding import bits_to_code_ids  # noqa: E402
from watermark.prf import derive_trial_key, score_table  # noqa: E402
from watermark.selection import cell_rng, select_code_ids  # noqa: E402

# Selector stream of the quality populations (matched-population FID, Vendi, FID against real images).
SELECTOR_DOMAIN = "gumbel-paper-rerun/fid10k/v1"
MATCHED_RANDOM_DOMAIN = "flattened-quality-matched-random/v1"
UNIFORM_DOMAIN = "same-prompt-vendi-uniform"
FID_DIMENSION = 2048
# Number of PRF keys over which the FID5k prompts rotate: the 5,000 prompts are spread evenly over
# 32 keys (see the appendix "Generation Quality and Prompt Alignment").
FID5K_NUM_KEYS = 32


def config_name(n: int, m: int) -> str:
    """Name of a configuration inside the selection arrays: ``n8_m4``."""
    return "n%d_m%d" % (n, m)


# ----------------------------------------------------------------------------- selection


def prompt_rng(*parts: object) -> np.random.Generator:
    """Generator seeded by the SHA-256 of a textual identity."""
    message = "\0".join(str(part) for part in parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(message).digest()[:8], "big"))


def trial_keys(master_key: bytes, count: int) -> Dict[int, bytes]:
    """PRF keys of the key trials ``0 .. count-1`` (trial 0 is the master key itself)."""
    if count <= 0:
        raise ValueError("the number of keys must be positive")
    return {trial: derive_trial_key(master_key, trial) for trial in range(count)}


def pool_identity(prompt_uid: str, slot: int) -> str:
    """Selector identity of the pool in ``slot`` (1-based): slot 1 is the bare prompt identity."""
    return str(prompt_uid) if slot == 1 else "%s:slot:%d" % (prompt_uid, slot)


def select_positions(
    candidate_vectors: np.ndarray, planes: np.ndarray, key: bytes, *, frame_id: str, key_trial: int,
    n: int, m: int, prompt_uids: Sequence[str],
) -> np.ndarray:
    """Gumbel selection on the first ``n`` candidates of every (prompt, slot) pool.

    ``candidate_vectors`` is ``(P, S, N, D)``. The selector draws are
    ``cell_rng(SELECTOR_DOMAIN, frame_id, key_trial, "hmac", n, m, pool_identity(prompt, slot))``.
    Returns the 1-based selected positions ``(P, S)``.
    """
    vectors = np.asarray(candidate_vectors)
    if vectors.ndim != 4 or vectors.shape[0] != len(prompt_uids) or vectors.shape[2] < n:
        raise ValueError("candidate vectors do not match the prompts and N")
    table = score_table(key, m)
    projection = np.einsum("psnd,md->psnm", vectors[:, :, :n], planes[:m], optimize=True)
    code_ids = bits_to_code_ids(projection >= 0.0)
    positions = np.empty(code_ids.shape[:2], dtype=np.int64)
    for p, prompt_uid in enumerate(prompt_uids):
        for s in range(code_ids.shape[1]):
            rng = cell_rng(SELECTOR_DOMAIN, frame_id, key_trial, "hmac", n, m, pool_identity(prompt_uid, s + 1))
            positions[p, s] = select_code_ids(code_ids[p, s], table, rng)[0] + 1
    return positions


def matched_random_positions(prompt_uids: Sequence[str], replicate: int, n: int, slots: int) -> np.ndarray:
    """Uniform draw among the first ``n`` candidates of every pool (the matched clean control).

    Slot 1 uses the identity of the single-slot control; later slots add the slot to it.
    Returns the 1-based positions ``(P, S)``.
    """
    prompts = [str(uid) for uid in prompt_uids]
    if not prompts or len(set(prompts)) != len(prompts) or replicate < 0 or n <= 0 or slots <= 0:
        raise ValueError("invalid matched-random configuration")
    positions = np.empty((len(prompts), slots), dtype=np.int64)
    for p, prompt_uid in enumerate(prompts):
        for slot in range(1, slots + 1):
            identity = (MATCHED_RANDOM_DOMAIN, replicate, n, prompt_uid)
            if slot != 1:
                identity = identity + ("slot", slot)
            positions[p, slot - 1] = int(prompt_rng(*identity).integers(1, n + 1))
    return positions


def uniform_positions(prompt_uid: str, frame_id: str, n: int, slots: int) -> np.ndarray:
    """Uniform control of the diversity metric: one position among ``n`` per slot ``(S,)`` (1-based)."""
    return np.asarray([
        int(prompt_rng(UNIFORM_DOMAIN, frame_id, n, prompt_uid, slot).integers(n)) + 1
        for slot in range(slots)
    ], dtype=np.int64)


def gather(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Pick ``values[p, s, positions[p, s] - 1]`` for every pool; ``values`` is ``(P, S, N, ...)``."""
    p, s = positions.shape
    if values.shape[:2] != (p, s) or positions.min() < 1 or positions.max() > values.shape[2]:
        raise ValueError("positions do not match the candidate array")
    return values[np.arange(p)[:, None], np.arange(s)[None, :], positions - 1]


# ----------------------------------------------------------------------------- FID


def activation_statistics(values: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Mean and covariance of Inception activations ``(count, 2048)``."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("FID activations must contain at least two samples")
    return np.mean(values, axis=0), np.cov(values, rowvar=False)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray,
                     eps: float = 1e-6) -> float:
    """Frechet distance between two Gaussians (the formula of ``pytorch-fid==0.3.0``)."""
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("imaginary component %s" % np.max(np.abs(covmean.imag)))
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def mean_interval(values: Sequence[float]) -> Dict[str, float]:
    """Mean, sample SD, standard error and 95% t-interval of replicate values."""
    from scipy import stats

    data = np.asarray(values, dtype=np.float64)
    mean = float(data.mean())
    if len(data) < 2:
        return {"mean": mean, "sd": 0.0, "se": 0.0, "ci95_low": mean, "ci95_high": mean, "count": len(data)}
    sd = float(data.std(ddof=1))
    se = sd / math.sqrt(len(data))
    half = float(stats.t.ppf(0.975, len(data) - 1)) * se
    return {"mean": mean, "sd": sd, "se": se, "ci95_low": mean - half, "ci95_high": mean + half,
            "count": len(data)}


def population_statistics(activations: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Statistics of a population given as ``(P, S, 2048)`` or ``(count, 2048)`` activations."""
    values = np.asarray(activations)
    return activation_statistics(values.reshape(-1, values.shape[-1]))


# ----------------------------------------------------------------------------- Vendi


def cosine_vendi_score(embeddings: np.ndarray) -> float:
    """Effective rank of the raw cosine-similarity kernel of unit-normalized embeddings."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embeddings must be nonzero")
    values = values / norms
    kernel = (values @ values.T + (values @ values.T).T) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(kernel), 0.0, None)
    total = float(eigenvalues.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("cosine kernel must have positive finite trace")
    probabilities = eigenvalues / total
    positive = probabilities[probabilities > 0.0]
    return float(math.exp(-float(np.sum(positive * np.log(positive)))))


# ----------------------------------------------------------------------------- native baselines


def read_native_panel(native_dir: Path, prompt_uid: str) -> Dict[str, Any]:
    """The ``panel.json`` of one prompt of a native-baseline generation (``baselines/generate.py``)."""
    panel = json.loads((Path(native_dir) / prompt_uid / "panel.json").read_text(encoding="utf-8"))
    if panel.get("prompt_uid") != prompt_uid or not panel.get("slots"):
        raise ValueError("unexpected native panel record for %s" % prompt_uid)
    return panel


def native_images(native_dir: Path, prompt_uid: str, slots: Optional[int] = None
                  ) -> "tuple[list[Path], list[Path]]":
    """Watermarked and clean image paths of one prompt, ordered by slot."""
    panel = read_native_panel(native_dir, prompt_uid)
    root = Path(native_dir) / prompt_uid
    chosen = sorted(panel["slots"], key=lambda item: int(item["slot"]))
    chosen = chosen if slots is None else chosen[:slots]
    if slots is not None and len(chosen) != slots:
        raise ValueError("native panel of %s has fewer than %d slots" % (prompt_uid, slots))
    return ([root / item["watermarked"]["relative_path"] for item in chosen],
            [root / item["clean"]["relative_path"] for item in chosen])


def parse_named_paths(values: Sequence[str]) -> Dict[str, Path]:
    """``name=path`` command-line pairs."""
    result: Dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("expected unique name=path arguments, got %r" % value)
        result[name] = Path(path)
    return result


# ----------------------------------------------------------------------------- activation shards


def write_shard(path: Path, prompt_uids: Sequence[str], **arrays: np.ndarray) -> None:
    """Write one shard of per-prompt arrays (float32, aligned to ``prompt_uids``)."""
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.npz" % path.stem)
    np.savez(str(temporary), prompt_uids=np.asarray(list(prompt_uids)),
             **{name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()})
    temporary.replace(path)


def load_shards(directory: Path, names: Sequence[str], prompt_uids: Sequence[str],
                prefix: str = "activations") -> Dict[str, np.ndarray]:
    """Merge the ``<prefix>-shard-*-of-*.npz`` shards of ``directory`` into arrays ordered like ``prompt_uids``."""
    paths = sorted(Path(directory).glob("%s-shard-*-of-*.npz" % prefix))
    if not paths:
        raise ValueError("no %s shards in %s" % (prefix, directory))
    expected = int(paths[0].stem.rsplit("-of-", 1)[1])
    if len(paths) != expected:
        raise ValueError("expected exactly %d %s shards in %s" % (expected, prefix, directory))
    rows: Dict[str, Dict[str, np.ndarray]] = {}
    for path in paths:
        with np.load(str(path), allow_pickle=False) as data:
            uids = [str(uid) for uid in data["prompt_uids"].tolist()]
            for index, uid in enumerate(uids):
                if uid in rows:
                    raise ValueError("prompt %s appears in two shards" % uid)
                rows[uid] = {name: data[name][index] for name in names}
    missing = [uid for uid in prompt_uids if uid not in rows]
    if missing:
        raise ValueError("activation shards lack %d prompts, first %s" % (len(missing), missing[0]))
    return {name: np.stack([rows[str(uid)][name] for uid in prompt_uids]) for name in names}


# ----------------------------------------------------------------------------- selections file


def selection_key(n: int, m: int, kind: str) -> str:
    return "%s_%s" % (config_name(n, m), kind)


def write_selections(path: Path, prompt_uids: Sequence[str], arrays: Mapping[str, np.ndarray],
                     metadata: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.npz" % path.stem)
    np.savez_compressed(str(temporary), prompt_uids=np.asarray(list(prompt_uids)),
                        metadata=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
                        **{name: np.asarray(value, dtype=np.uint8) for name, value in arrays.items()})
    temporary.replace(path)


def load_selections(path: Path) -> "tuple[list[str], Dict[str, np.ndarray], Dict[str, Any]]":
    with np.load(str(path), allow_pickle=False) as data:
        uids = [str(uid) for uid in data["prompt_uids"].tolist()]
        metadata = json.loads(str(data["metadata"].item()))
        arrays = {name: np.asarray(data[name], dtype=np.int64) for name in data.files
                  if name not in ("prompt_uids", "metadata")}
    return uids, arrays, metadata


# ----------------------------------------------------------------------------- COCO panel

_DASH_TRANSLATION = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                                   "—": "-", "−": "-"})
PANEL_VERSION = "coco-fid5k-panel/v1"
CAPTION_RANK_DOMAIN = "coco-fid5k-caption/v1"


def normalize_prompt(text: str) -> str:
    """NFKC, dash unification, control characters to spaces, whitespace collapsed."""
    if not isinstance(text, str):
        raise TypeError("prompt text must be a string")
    value = unicodedata.normalize("NFKC", text).translate(_DASH_TRANSLATION)
    value = "".join(
        character if not unicodedata.category(character).startswith("C") or character in "\t\n\r" else " "
        for character in value
    )
    return " ".join(value.split()).strip()


def build_fid5k_panel(captions: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One hash-ranked caption per official COCO validation image, unique across images.

    ``captions`` is ``captions_val2017.json``. Images are visited in order of their numeric
    ID; each takes the caption of smallest SHA-256 rank whose case-folded text no earlier
    image has taken.
    """
    images, annotations = captions.get("images"), captions.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list) or not images:
        raise ValueError("the captions file must contain images and annotations")
    by_id: Dict[int, Mapping[str, Any]] = {}
    for image in images:
        image_id = int(image["id"])
        file_name = image.get("file_name")
        if image_id in by_id or not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError("invalid or duplicate COCO image record %d" % image_id)
        by_id[image_id] = image
    ranked: Dict[int, List[Tuple[str, int, str]]] = {image_id: [] for image_id in by_id}
    seen_annotations: set = set()
    for annotation in annotations:
        annotation_id, image_id = int(annotation["id"]), int(annotation["image_id"])
        if annotation_id in seen_annotations or image_id not in ranked:
            raise ValueError("invalid or duplicate COCO annotation %d" % annotation_id)
        seen_annotations.add(annotation_id)
        caption = normalize_prompt(str(annotation["caption"]))
        if not caption:
            raise ValueError("empty caption for annotation %d" % annotation_id)
        payload = "%s\0%d\0%d\0%s" % (CAPTION_RANK_DOMAIN, image_id, annotation_id, caption)
        ranked[image_id].append((hashlib.sha256(payload.encode("utf-8")).hexdigest(), annotation_id, caption))
    taken: set = set()
    rows: List[Dict[str, Any]] = []
    for index, image_id in enumerate(sorted(by_id)):
        choice = next(((rank, annotation_id, caption) for rank, annotation_id, caption in sorted(ranked[image_id])
                       if caption.casefold() not in taken), None)
        if choice is None:
            raise ValueError("no unused caption for image %d" % image_id)
        _, annotation_id, caption = choice
        taken.add(caption.casefold())
        payload = "%s\0%d\0%d\0%s" % (PANEL_VERSION, image_id, annotation_id, caption)
        rows.append({
            "prompt_uid": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "text": caption,
            "panel_index": index, "image_id": image_id, "annotation_id": annotation_id,
            "file_name": str(by_id[image_id]["file_name"]),
        })
    return rows


# ----------------------------------------------------------------------------- output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dictionaries as CSV (columns in order of first appearance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
