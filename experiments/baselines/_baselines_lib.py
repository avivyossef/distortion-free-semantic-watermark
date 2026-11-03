"""Shared, GPU-free helpers of the native-baseline experiment.

Seeds, fixed method state, the on-disk image layout written by ``generate.py`` and read by
the quality experiment, image identifiers, and the clean-detection and robustness
aggregation. Importing this module needs only numpy and the public ``watermark`` package;
Pillow is imported inside the functions that read images.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.attacks import EDIT_GROUPS, EDIT_IDS  # noqa: E402
from watermark.calibration import clean_threshold  # noqa: E402
from watermark.io import canonical_json, pixel_sha256, sha256_file  # noqa: E402
from watermark.metrics import clean_detection_roc, paired_retention  # noqa: E402
from watermark.prf import derive_digest, derive_seed  # noqa: E402

METHODS = ("tree_ring", "gaussian_shading", "seal")
# Directory names written by third_party/clone.sh and the commits it pins.
REPOSITORY_DIRS = {"tree_ring": "tree-ring-watermark", "gaussian_shading": "Gaussian-Shading", "seal": "SEAL"}
OFFICIAL_COMMITS = {
    "tree_ring": "3015283d9cf82e90b628f02ad2121bd37408ca9a",
    "gaussian_shading": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc",
    "seal": "92d31b31b93a6e373fe88a584c640beafb68c6fa",
}
NATIVE_KEY_ID = "key-00"
ROLES = ("watermarked", "clean")
IDENTITY = "identity"  # edit identifier of the unedited image


# --------------------------------------------------------------------------- seeds


def native_seed(prompt_uid: str, key_id: str = NATIVE_KEY_ID) -> int:
    """Per-identity seed of slot 1; the same function seeds the clean control of the SD2.1 corpus."""
    digest = hashlib.sha256(("sd21-semantic-baseline/v1\0%s\0%s" % (prompt_uid, key_id)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def extension_slot_seed(prompt_uid: str, slot: int) -> int:
    """Seed of slots 2 and above."""
    digest = hashlib.sha256(("fid10k-slot-seed/v1\0%s\0%d" % (prompt_uid, slot)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def generation_seed(prompt_uid: str, slot: int) -> int:
    """Seed shared by the watermarked and clean image of ``(prompt, slot)``; slots count from 1."""
    if not isinstance(slot, int) or slot < 1:
        raise ValueError("slot must be a positive integer")
    return native_seed(prompt_uid) if slot == 1 else extension_slot_seed(prompt_uid, slot)


# --------------------------------------------------------------------------- fixed method state


def tree_ring_seed(master_key: bytes) -> int:
    """Seed of the Tree-Ring pattern: one pattern for every image of a run."""
    return derive_seed(master_key, "baseline/tree-ring/pattern") % (2 ** 32)


def gaussian_key_nonce(master_key: bytes) -> "tuple[bytes, bytes]":
    """ChaCha20 key (32 bytes) and nonce (12 bytes) of the single Gaussian Shading state."""
    return (derive_digest(master_key, "baseline/gaussian-shading/key"),
            derive_digest(master_key, "baseline/gaussian-shading/nonce")[:12])


def gaussian_watermark_bits(master_key: bytes, channels: int, size: int) -> np.ndarray:
    """The fixed Gaussian Shading message, a ``(1, channels, size, size)`` int64 array of bits."""
    rng = np.random.default_rng(derive_seed(master_key, "baseline/gaussian-shading/message"))
    return rng.integers(0, 2, size=(1, channels, size, size)).astype(np.int64)


# --------------------------------------------------------------------------- environment


def verify_checkout(repo: Path, method: str) -> None:
    """Require the official checkout at the commit pinned by ``third_party/clone.sh``."""
    if not Path(repo).is_dir():
        raise SystemExit("official repository is missing: %s (run third_party/clone.sh)" % repo)
    try:
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit("cannot verify the pinned commit of %s with git: %s" % (repo, error))
    if head != OFFICIAL_COMMITS[method]:
        raise SystemExit("%s is not at the pinned commit %s" % (repo, OFFICIAL_COMMITS[method]))
    if dirty:
        raise SystemExit("%s has uncommitted changes; refusing to run unverified code" % repo)


def in_shard(index: int, shard_index: int, num_shards: int) -> bool:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard")
    return index % num_shards == shard_index


# --------------------------------------------------------------------------- image layout


def image_record(path: Path, root: Path) -> Dict[str, str]:
    """``{relative_path, file_sha256, pixel_sha256}`` of one written PNG."""
    from PIL import Image

    with Image.open(path) as image:
        pixels = pixel_sha256(image)
    return {"relative_path": str(Path(path).relative_to(root)), "file_sha256": sha256_file(path),
            "pixel_sha256": pixels}


def slot_dir_name(slot: int) -> str:
    return "slot_%03d" % slot


def write_prompt_panel(output_dir: Path, record: Mapping[str, Any], write_images) -> None:
    """Publish one prompt atomically: ``write_images(staging_dir) -> slot records`` fills a staging
    directory, ``panel.json`` is written last and the directory is renamed into place."""
    destination = Path(output_dir) / record["prompt_uid"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s.staging-" % record["prompt_uid"], dir=str(output_dir)))
    try:
        panel = dict(record)
        panel["slots"] = write_images(staging)
        (staging / "panel.json.tmp").write_text(canonical_json(panel) + "\n", encoding="utf-8")
        os.replace(str(staging / "panel.json.tmp"), str(staging / "panel.json"))
        os.replace(str(staging), str(destination))
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise


def read_panel(output_dir: Path, prompt_uid: str) -> Dict[str, Any]:
    import json

    return json.loads((Path(output_dir) / prompt_uid / "panel.json").read_text(encoding="utf-8"))


def panel_image_path(output_dir: Path, panel: Mapping[str, Any], slot: int, role: str) -> Path:
    """Path of the ``role`` image (``watermarked`` or ``clean``) of one slot."""
    for item in panel["slots"]:
        if item["slot"] == slot:
            return Path(output_dir) / panel["prompt_uid"] / item[role]["relative_path"]
    raise KeyError("slot %d is not in the panel of %s" % (slot, panel["prompt_uid"]))


def make_image_id(prompt_uid: str, role: str, edit_id: str = IDENTITY) -> str:
    if role not in ROLES or "|" in prompt_uid:
        raise ValueError("invalid image identity")
    return "%s|%s|%s" % (prompt_uid, role, edit_id)


def parse_image_id(image_id: str) -> "tuple[str, str, str]":
    prompt_uid, role, edit_id = image_id.rsplit("|", 2)
    return prompt_uid, role, edit_id


def read_prompts(path: Path) -> List[Dict[str, str]]:
    from watermark.io import read_jsonl

    rows = read_jsonl(path)
    uids = [row["prompt_uid"] for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate prompt_uid in %s" % path)
    return rows


# --------------------------------------------------------------------------- scores


def collect_scores(rows: Iterable[Mapping[str, Any]]) -> Dict["tuple[str, str]", Dict[str, float]]:
    """``{(edit_id, role): {prompt_uid: score}}`` from score.py rows of this layout."""
    table: Dict["tuple[str, str]", Dict[str, float]] = {}
    for row in rows:
        prompt_uid, role, edit_id = parse_image_id(row["image_id"])
        value = float(row["score"])
        if not math.isfinite(value):
            raise ValueError("non-finite score for %s" % row["image_id"])
        bucket = table.setdefault((edit_id, role), {})
        if prompt_uid in bucket:
            raise ValueError("duplicate score for %s" % row["image_id"])
        bucket[prompt_uid] = value
    return table


def clean_detection(clean: Sequence[float], watermarked: Sequence[float], fpr_targets: Sequence[float]) -> Dict[str, Any]:
    """AUROC, ROC and the operating points of one method at the configured empirical FPR targets."""
    return clean_detection_roc(list(clean), list(watermarked), tuple(fpr_targets))


def analyze_robustness(
    table: Mapping["tuple[str, str]", Mapping[str, float]], fpr_targets: Sequence[float]
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Per-edit and per-category robustness of one native method.

    The threshold tau of each FPR target is frozen on the clean unedited scores only. For every
    edit: TPR on the edited watermarked images, FPR on the edited clean images, and paired
    retention (the fraction of unedited detections that stay detected). A category is the
    unweighted mean over its edit settings.
    """
    for role in ROLES:
        if (IDENTITY, role) not in table:
            raise ValueError("scores of the unedited %s images are required" % role)
    uids = sorted(table[(IDENTITY, "watermarked")])
    for key, bucket in table.items():
        if sorted(bucket) != uids:
            raise ValueError("every image set must cover the same prompts: %s" % (key,))
    for edit_id in EDIT_IDS:
        for role in ROLES:
            if (edit_id, role) not in table:
                raise ValueError("missing scores for edit %s (%s)" % (edit_id, role))

    def vector(edit_id: str, role: str) -> np.ndarray:
        return np.asarray([table[(edit_id, role)][uid] for uid in uids], dtype=np.float64)

    per_edit: List[Dict[str, Any]] = []
    categories: List[Dict[str, Any]] = []
    for target in fpr_targets:
        tau, empirical = clean_threshold(vector(IDENTITY, "clean"), float(target))
        original = vector(IDENTITY, "watermarked")
        rows = []
        for edit_id in EDIT_IDS:
            marked, clean = vector(edit_id, "watermarked"), vector(edit_id, "clean")
            retention = paired_retention(original, marked, tau)
            rows.append({
                "requested_fpr": float(target), "threshold": tau, "calibration_fpr": empirical,
                "edit_id": edit_id,
                "category": next(name for name, members in EDIT_GROUPS.items() if edit_id in members),
                "unedited_tpr": float(np.mean(original >= tau)),
                "edited_tpr": float(np.mean(marked >= tau)),
                "edited_fpr": float(np.mean(clean >= tau)),
                "retention": retention["paired_detection_retention"],
            })
        per_edit.extend(rows)
        for name, members in EDIT_GROUPS.items():
            own = [row for row in rows if row["edit_id"] in members]
            categories.append({
                "requested_fpr": float(target), "category": name, "settings": len(own),
                "edited_tpr": float(np.mean([row["edited_tpr"] for row in own])),
                "edited_fpr": float(np.mean([row["edited_fpr"] for row in own])),
                "retention": _mean_defined([row["retention"] for row in own]),
            })
    return per_edit, categories


def _mean_defined(values: Sequence[float]) -> float:
    """Mean over the settings where retention is defined (NaN when nothing was detected before the edit)."""
    defined = [value for value in values if not math.isnan(value)]
    return float(np.mean(defined)) if defined else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
