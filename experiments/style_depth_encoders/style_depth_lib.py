"""Shared pieces of the style / depth encoder study: panel, edits, per-replicate scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

from watermark import attacks
from watermark.calibration import clean_threshold
from watermark.config import load_assets
from watermark.detection import replicate_scores, score_embeddings
from watermark.metrics import paired_retention

PROMPT_COUNT = 100  # the first 100 prompts of the encoder-ablation panel
SLOT_CANDIDATES = 16  # slot-1 candidates per prompt; the slot-1 clean image follows as row 16
ENCODERS = ("csd_style", "depth_layout")
ENCODER_LABELS = {"csd_style": "CSD style", "depth_layout": "Depth layout"}

IP2P_MODEL = load_assets()["models"]["instruct_pix2pix"]["repo"]
IP2P_REVISION = load_assets()["models"]["instruct_pix2pix"]["revision"]
EDIT_SIZE = 512
STEPS, IMAGE_GUIDANCE, GUIDANCE = 30, 1.5, 7.5
STYLE_EDITS = {
    "ip2p_oil_painting": "Make it an oil painting",
    "ip2p_watercolor": "Turn it into a watercolor painting",
    "ip2p_pencil_sketch": "Turn it into a pencil sketch",
}
SPATIAL_EDITS = ("translate_32", "offcenter_crop_75", "hflip")
EDIT_LABELS = {
    "ip2p_oil_painting": "Oil painting", "ip2p_watercolor": "Watercolor", "ip2p_pencil_sketch": "Pencil sketch",
    "translate_32": "Translation, 32 px", "offcenter_crop_75": "Off-center crop, 75%", "hflip": "Horizontal flip",
}
EDITS = tuple(STYLE_EDITS) + SPATIAL_EDITS


def panel_prompt_uids(panel_root: Path) -> List[str]:
    """The first ``PROMPT_COUNT`` prompt identities of an SD2.1 panel directory, in sorted order."""
    uids = sorted(path.parent.name for path in Path(panel_root).glob("*/panel.json"))
    if len(uids) < PROMPT_COUNT:
        raise ValueError("panel has %d prompts, expected at least %d" % (len(uids), PROMPT_COUNT))
    return uids[:PROMPT_COUNT]


def source_paths(panel_root: Path, prompt_uid: str) -> List[Path]:
    """Slot-1 candidate images in position order (rows 0..15), then the slot-1 clean image (row 16)."""
    root = Path(panel_root) / prompt_uid
    record = json.loads((root / "panel.json").read_text(encoding="utf-8"))
    slots = [slot for slot in record["batch_slots"] if slot.get("slot") == 1]
    if len(slots) != 1:
        raise ValueError("prompt %s does not contain exactly one slot 1" % prompt_uid)
    images = sorted(slots[0]["candidate_images"], key=lambda value: value["candidate_position"])
    if [value["candidate_position"] for value in images] != list(range(1, SLOT_CANDIDATES + 1)):
        raise ValueError("prompt %s does not contain candidate positions 1..%d" % (prompt_uid, SLOT_CANDIDATES))
    paths = [(root / value["relative_path"]).resolve() for value in images]
    paths.append((root / slots[0]["clean"]["relative_path"]).resolve())
    if any(root.resolve() not in path.parents or not path.is_file() for path in paths):
        raise ValueError("prompt %s contains an invalid image path" % prompt_uid)
    return paths


def edit_seed(prompt_uid: str, edit_id: str, index: int) -> int:
    digest = hashlib.sha256(f"semantic-axis-edit|{edit_id}|{prompt_uid}|{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def spatial_edit(image: Image.Image, edit_id: str) -> Image.Image:
    """One of the three spatial transformations of the study."""
    if edit_id == "hflip":
        return ImageOps.mirror(image)
    if edit_id not in SPATIAL_EDITS:
        raise ValueError("unknown spatial edit: %s" % edit_id)
    return attacks.render_edit(image, edit_id)


def cosine_distance(original: np.ndarray, edited: np.ndarray) -> float:
    """Mean ``1 - <z, z'>`` over all candidates and clean controls of normalized embeddings."""
    return float(1.0 - np.sum(original * edited, axis=-1).mean())


def score_replicate(
    original: np.ndarray, edited: Mapping[str, np.ndarray], prompt_uids: Sequence[str], planes: np.ndarray,
    key: bytes, *, frame_id: str, key_trial: int, n: int, m: int, activation: float, radius: int, folds: int,
    fpr: float,
) -> Dict[str, Dict[str, Any]]:
    """Score one (frame, key) replicate on unedited and edited images.

    ``original`` and every ``edited[name]`` are ``(prompts, n + 1, D)`` unit-norm embeddings:
    rows ``0..n-1`` the candidates and row ``n`` the clean control. Candidate selection and the
    boundary thresholds delta use the unedited embeddings only. The score threshold targets
    ``fpr`` on the unedited clean images that are also evaluated, and stays fixed after
    editing. Returns, for ``"original"`` and every edit, the detection rate (``tpr``), the clean
    false-positive rate and, for edits, the retention (fraction of originally detected images
    still detected; NaN when nothing was detected).
    """
    rep = replicate_scores(
        original[:, :n], original[:, n], prompt_uids, planes, key, frame_id=frame_id, key_trial=key_trial,
        n=n, m=m, activation=activation, radius=radius, folds=folds,
    )
    tau, clean_fpr = clean_threshold(rep.clean.scores, fpr)
    rows = np.arange(len(prompt_uids))
    result: Dict[str, Dict[str, Any]] = {
        "original": {"tpr": float(np.mean(rep.watermarked.scores >= tau)), "clean_fpr": float(clean_fpr),
                     "retention": float("nan")}
    }
    for name, values in edited.items():
        watermarked = score_embeddings(values[rows, rep.selected_index], planes, m, rep.table, rep.deltas,
                                       radius=radius).scores
        clean = score_embeddings(values[:, n], planes, m, rep.table, rep.deltas, radius=radius).scores
        result[name] = {
            "tpr": float(np.mean(watermarked >= tau)), "clean_fpr": float(np.mean(clean >= tau)),
            "retention": paired_retention(rep.watermarked.scores, watermarked, tau)["paired_detection_retention"],
        }
    return result
