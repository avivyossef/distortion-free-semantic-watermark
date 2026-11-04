"""Shared helpers of the generative-edit experiment (paper: robustness to generative edits).

Pure numpy/stdlib code; torch, transformers and diffusers are imported only by the GPU scripts.

Layout of the files exchanged between the scripts:

``panel.jsonl``            ``{"prompt_uid", "text", "edit_category"}``, one row per edited prompt
``sources.jsonl``          one row per distinct source image to edit: ``{"source_id", "prompt_uid",
                           "text", "edit_category", "roles", "path"}`` (``source_id`` is the SHA-256 of the
                           image file)
``plans-shard-*.jsonl``    ``{"source_id", "prompt_uid", "edit_category", "response", "plan"}``
``manifest-shard-*.jsonl`` ``{"source_id", "edit_category", "instruction", "edited_path",
                           "edited_file_sha256"}`` (``edited_path`` relative to the render directory)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.calibration import clean_threshold  # noqa: E402
from watermark.coding import sample_hyperplanes  # noqa: E402
from watermark.config import load_assets  # noqa: E402
from watermark.detection import replicate_scores, score_embeddings  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402
from watermark.metrics import paired_retention  # noqa: E402
from watermark.prf import derive_trial_key  # noqa: E402

# Edit categories of the paper (appendix "Detection and Robustness Protocols"). The identifiers
# are the ones that seed the hash ordering of the panel.
CATEGORIES = ("attribute_edit", "background_cleanup", "cat_attack", "camera_viewpoint_change")

# Fixed category constraints given to the vision-language model (paper Table "Fixed prompt constraint").
CATEGORY_CONSTRAINTS = {
    "attribute_edit": (
        "Modify only the colour or material of one visibly present, non-salient secondary object. "
        "Preserve the main subject, main action, object inventory, and scene. "
        "Do not target faces, text, logos, or watermarks."
    ),
    "background_cleanup": (
        "Remove only one tiny, non-salient background clutter item. "
        "Preserve the main subject, main action, salient objects, and scene. "
        "Do not target faces, text, logos, or watermarks."
    ),
    "cat_attack": (
        "Add one unrelated small realistic cat in the background. "
        "The cat must be clearly visible but must not replace or occlude the main subject. "
        "Do not target faces, text, logos, or watermarks."
    ),
    "camera_viewpoint_change": (
        "Change only the camera viewpoint to a clearly different three-quarter or slightly elevated viewpoint. "
        "Preserve the same main subject, objects, actions, scene, colours, and lighting. "
        "Do not target faces, text, logos, or watermarks."
    ),
}

# Pinned models (configs/assets.yaml) and the render settings used for the paper (Qwen planner decode and editor settings).
_MODELS = load_assets()["models"]
PLANNER = {
    "repo": _MODELS["qwen3_vl_planner"]["repo"],
    "revision": _MODELS["qwen3_vl_planner"]["revision"],
    "do_sample": False,
    "max_new_tokens": 192,
}
EDITOR = {
    "repo": _MODELS["qwen_image_edit"]["repo"],
    "revision": _MODELS["qwen_image_edit"]["revision"],
    "seed": 20260830,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
}
PLAN_FIELDS = ("visible_subject", "context", "edit_category", "edit_instruction")


# --------------------------------------------------------------------------- panel and prompts


def edit_plan_prompt(edit_category: str) -> str:
    """The request sent with the source image to the vision-language model."""
    if edit_category not in CATEGORIES:
        raise ValueError("unsupported edit category")
    return (
        "Inspect the provided image and prepare one feasible Qwen-Image-Edit instruction. "
        "Assigned category: %s. %s "
        "Return only one valid JSON object with exactly these string fields: visible_subject, context, "
        "edit_category, edit_instruction. The edit_category must be exactly %s."
        % (edit_category, CATEGORY_CONSTRAINTS[edit_category], edit_category)
    )


def validate_edit_plan(payload: str, *, expected_category: str) -> Dict[str, str]:
    """Fail closed unless the model produced exactly the typed plan of its assigned category."""
    if expected_category not in CATEGORIES:
        raise ValueError("unsupported expected category")
    if not isinstance(payload, str) or payload != payload.strip():
        raise ValueError("plan must be valid JSON without surrounding text")
    try:
        plan = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("plan must be valid JSON") from error
    if not isinstance(plan, dict) or set(plan) != set(PLAN_FIELDS):
        raise ValueError("plan must contain exactly the required fields")
    if any(not isinstance(plan[field], str) or not plan[field].strip() or plan[field] != plan[field].strip()
           for field in PLAN_FIELDS):
        raise ValueError("plan fields must be nonempty trimmed strings")
    if plan["edit_category"] != expected_category:
        raise ValueError("plan edit category does not match the assigned category")
    return {field: plan[field] for field in PLAN_FIELDS}


def balanced_panel(rows: Sequence[Mapping[str, Any]], *, per_category: int) -> List[Tuple[Dict[str, Any], str]]:
    """Hash-ordered, category-balanced panel that does not depend on input order or outcomes.

    Within each category (in :data:`CATEGORIES` order) rows are ranked by the SHA-256 of
    ``category:prompt_uid:key_id`` (``key_id`` is empty for rows without one) and the first
    ``per_category`` rows not used by an earlier category are taken.
    """
    if not isinstance(per_category, int) or per_category <= 0:
        raise ValueError("per_category must be a positive integer")
    candidates = [dict(row) for row in rows]
    selected: List[Tuple[Dict[str, Any], str]] = []
    used = set()
    for category in CATEGORIES:
        ordered = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                ("%s:%s:%s" % (category, row.get("prompt_uid"), row.get("key_id", ""))).encode("utf-8")
            ).hexdigest(),
        )
        chosen = 0
        for row in ordered:
            identity = (str(row.get("prompt_uid")), str(row.get("key_id", "")))
            if identity in used:
                continue
            selected.append((row, category))
            used.add(identity)
            chosen += 1
            if chosen == per_category:
                break
        if chosen != per_category:
            raise ValueError("the prompt list does not contain enough unique rows for a balanced panel")
    return selected


def read_prompts(path: Path) -> List[Dict[str, Any]]:
    """Prompt list: JSONL rows ``{"prompt_uid", "text"}``, sorted by identity."""
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


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_shards(directory: Path, pattern: str) -> List[Dict[str, Any]]:
    """Concatenate the rows of every ``pattern`` file of a directory (sorted by file name)."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(directory).glob(pattern)):
        rows.extend(read_jsonl(path))
    return rows


# --------------------------------------------------------------------------- source images


def ours_panel(corpus_dir: Path, prompt_uid: str) -> Dict[str, Any]:
    """Slot-1 image records of one prompt of the SD2.1 candidate corpus (``watermark.generation`` layout).

    Returns ``{"candidates": [(file_sha256, path), ...]`` ordered by position, ``"clean": (sha, path)}``.
    """
    root = Path(corpus_dir) / prompt_uid
    record = json.loads((root / "panel.json").read_text(encoding="utf-8"))
    if record.get("mode") != "candidates" or record.get("prompt_uid") != prompt_uid:
        raise ValueError("unexpected panel record for %s" % prompt_uid)
    slot = sorted(record["batch_slots"], key=lambda item: int(item["slot"]))[0]
    items = sorted(slot["candidate_images"], key=lambda item: int(item["candidate_position"]))
    return {
        "candidates": [(item["file_sha256"], root / item["relative_path"]) for item in items],
        "clean": (slot["clean"]["file_sha256"], root / slot["clean"]["relative_path"]),
    }


def baseline_panel(baseline_dir: Path, prompt_uid: str) -> Dict[str, Tuple[str, Path]]:
    """Slot-1 watermarked and clean images of one prompt of a native-baseline run."""
    root = Path(baseline_dir) / prompt_uid
    record = json.loads((root / "panel.json").read_text(encoding="utf-8"))
    if record.get("prompt_uid") != prompt_uid:
        raise ValueError("unexpected baseline panel record for %s" % prompt_uid)
    slot = sorted(record["slots"], key=lambda item: int(item["slot"]))[0]
    return {role: (slot[role]["file_sha256"], root / slot[role]["relative_path"])
            for role in ("watermarked", "clean")}


def merge_sources(entries: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Distinct source images (by file hash), each with the roles it plays; sorted by ``source_id``."""
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        source_id = str(entry["source_id"])
        row = merged.setdefault(source_id, {
            "source_id": source_id, "prompt_uid": entry["prompt_uid"], "text": entry["text"],
            "edit_category": entry["edit_category"], "roles": [], "path": str(entry["path"]),
        })
        if row["prompt_uid"] != entry["prompt_uid"] or row["edit_category"] != entry["edit_category"]:
            raise ValueError("one image file cannot belong to two prompts or categories")
        if entry["role"] not in row["roles"]:
            row["roles"].append(entry["role"])
    return [merged[key] for key in sorted(merged)]


# --------------------------------------------------------------------------- ours: replicates


def load_cache(path: Path) -> Dict[str, Any]:
    """Embedding cache of the detection experiments (``.npz``): prompts, slot-1 candidates and cleans.

    Layout: ``prompt_uids (P,)``, ``candidate_vectors (P, S, N, D)``, ``clean_vectors (P, S, D)``, all
    unit-norm. Only slot 1 is used.
    """
    with np.load(str(path), allow_pickle=False) as data:
        uids = [str(value) for value in data["prompt_uids"].tolist()]
        candidates = np.asarray(data["candidate_vectors"], dtype=np.float64)
        cleans = np.asarray(data["clean_vectors"], dtype=np.float64)
    if candidates.ndim != 4 or cleans.shape != candidates.shape[:2] + candidates.shape[3:] \
            or candidates.shape[0] != len(uids) or len(set(uids)) != len(uids):
        raise ValueError("inconsistent embedding cache: %s" % path)
    return {"prompt_uids": uids, "candidates": candidates[:, 0], "cleans": cleans[:, 0]}


def restrict_cache(cache: Mapping[str, Any], prompt_uids: Sequence[str]) -> Dict[str, Any]:
    """The cache restricted to ``prompt_uids`` in the given order."""
    index = {uid: row for row, uid in enumerate(cache["prompt_uids"])}
    missing = [uid for uid in prompt_uids if uid not in index]
    if missing:
        raise ValueError("embedding cache lacks %d calibration prompts" % len(missing))
    rows = np.asarray([index[uid] for uid in prompt_uids], dtype=np.int64)
    return {"prompt_uids": list(prompt_uids), "candidates": cache["candidates"][rows],
            "cleans": cache["cleans"][rows]}


def panel_rows_in_cache(panel_rows: Sequence[Mapping[str, Any]], cache: Mapping[str, Any]) -> np.ndarray:
    """Cache row of every panel prompt; the panel must lie inside the calibration population."""
    position = {uid: row for row, uid in enumerate(cache["prompt_uids"])}
    missing = [row["prompt_uid"] for row in panel_rows if row["prompt_uid"] not in position]
    if missing:
        raise ValueError("%d panel prompts are not in the calibration prompt list" % len(missing))
    return np.asarray([position[row["prompt_uid"]] for row in panel_rows], dtype=np.int64)


def ours_replicates(cache: Mapping[str, Any], cfg: Any, master_key: bytes):
    """Every (configuration, frame, key) replicate of the paper over the calibration population.

    Yields ``(setting, frame_id, key_trial, planes, scores)``; ``scores`` is the
    :class:`watermark.detection.ReplicateScores` of the unedited calibration images (delta is
    cross-fitted on the selected watermarked margins, the clean scores give the frozen tau).
    """
    dimension = cache["candidates"].shape[-1]
    frames = {frame_id: sample_hyperplanes(seed, dimension)
              for frame_id, seed in zip(cfg.frame_ids(), cfg.frame_seeds())}
    keys = {trial: derive_trial_key(master_key, trial) for trial in cfg.key_trials()}
    for setting in cfg.settings:
        for frame_id in cfg.frame_ids():
            for trial in cfg.key_trials():
                scores = replicate_scores(
                    cache["candidates"][:, :setting.n], cache["cleans"], cache["prompt_uids"],
                    frames[frame_id], keys[trial], frame_id=frame_id, key_trial=trial, n=setting.n,
                    m=setting.m, activation=setting.activation, radius=setting.radius, folds=cfg.folds)
                yield setting, frame_id, trial, frames[frame_id], scores


def edited_scores(scores: Any, planes: np.ndarray, setting: Any, rows: np.ndarray,
                  vectors: np.ndarray) -> np.ndarray:
    """Boundary scores of edited images of calibration rows ``rows`` under one replicate.

    The replicate's keyed table and each prompt's delta come from the unedited images and stay
    frozen; ``vectors`` are the unit-norm embeddings of the edited images.
    """
    return score_embeddings(vectors, planes, setting.m, scores.table, scores.deltas[rows],
                            radius=setting.radius).scores


# --------------------------------------------------------------------------- metrics


METRICS = ("original_tpr", "edited_tpr", "retention", "edited_fpr", "calibration_fpr")


def detection_metrics(calibration_clean: np.ndarray, original: np.ndarray, edited: np.ndarray,
                      edited_clean: np.ndarray, fpr: float) -> Dict[str, float]:
    """Frozen-threshold detection rates of one method (or replicate) over one set of edits.

    ``tau`` is frozen from the unedited clean calibration scores at the requested FPR. A score is
    detected when it is at least tau. ``original`` and ``edited`` are the scores of the same
    watermarked images before and after editing (paired); ``edited_clean`` are the scores of the
    edited unwatermarked images.
    """
    tau, calibration_fpr = clean_threshold(np.asarray(calibration_clean, dtype=np.float64), fpr)
    original, edited, edited_clean = (np.asarray(v, dtype=np.float64) for v in (original, edited, edited_clean))
    retention = paired_retention(original, edited, tau)["paired_detection_retention"]
    return {
        "threshold": float(tau),
        "calibration_fpr": float(calibration_fpr),
        "original_tpr": float(np.mean(original >= tau)),
        "edited_tpr": float(np.mean(edited >= tau)),
        "retention": float(retention),
        "edited_fpr": float(np.mean(edited_clean >= tau)),
    }


def category_metrics(calibration_clean: np.ndarray, categories: Sequence[str], original: np.ndarray,
                     edited: np.ndarray, edited_clean: np.ndarray, fpr: float) -> Dict[str, Dict[str, float]]:
    """:func:`detection_metrics` for every category and for all edits pooled (key ``"all"``)."""
    labels = np.asarray(categories)
    result = {}
    for category in (*CATEGORIES, "all"):
        keep = np.ones(len(labels), dtype=bool) if category == "all" else labels == category
        if not keep.any():
            raise ValueError("no edits in category %s" % category)
        result[category] = detection_metrics(
            calibration_clean, np.asarray(original)[keep], np.asarray(edited)[keep],
            np.asarray(edited_clean)[keep], fpr)
        result[category]["edits"] = int(keep.sum())
    return result


def average_replicates(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Mean over the (frame, key) replicates of per-replicate metric rows.

    Rows carry ``method``, ``fpr_target``, ``category`` and :data:`METRICS`; retention is averaged
    over the replicates where it is defined.
    """
    groups: Dict[Tuple[str, float, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), float(row["fpr_target"]), str(row["category"])), []).append(row)
    result = []
    for (method, fpr, category), members in sorted(groups.items()):
        out: Dict[str, Any] = {"method": method, "fpr_target": fpr, "category": category,
                               "replicates": len(members), "edits": int(members[0]["edits"])}
        for name in METRICS:
            values = np.asarray([float(member[name]) for member in members], dtype=np.float64)
            out[name] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        result.append(out)
    return result
