"""Shared pieces of the robustness experiments (also used by the sampling-rule and sweep experiments)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark import selection_rules as rules  # noqa: E402
from watermark.attacks import EDIT_GROUPS, EDIT_IDS  # noqa: E402
from watermark.calibration import clean_threshold  # noqa: E402
from watermark.coding import bits_to_code_ids  # noqa: E402
from watermark.detection import ReplicateScores, boundary_scores, score_embeddings, selector_rng  # noqa: E402
from watermark.metrics import paired_retention  # noqa: E402
from watermark.prf import score_table  # noqa: E402

SLOT = 1  # robustness uses B = 1: the first slot of every prompt


def target_label(fpr: float) -> str:
    return "%02d" % int(round(100 * fpr))


def source_id(prompt_uid: str, position: Optional[int]) -> str:
    """Identity of an image that is edited: a candidate (1-based position) or the clean control."""
    return "%s:clean" % prompt_uid if position is None else "%s:candidate:%02d" % (prompt_uid, position)


def selected_indices(candidates: np.ndarray, prompt_uids: Sequence[str], planes: np.ndarray, key: bytes, *,
                     frame_id: str, key_trial: int, n: int, m: int, rule: str = "gumbel") -> np.ndarray:
    """Index (0-based) of the candidate selected per prompt (Gumbel selection, the edit stream).

    Equals ``replicate_scores(..., rule=rule).selected_index`` with ``slot=None``; ``rule`` is
    other than ``"gumbel"`` only in the sampling-rule comparison.
    """
    with np.errstate(all="ignore"):
        projection = np.einsum("pnd,md->pnm", candidates[:, :n], planes[:m], optimize=True)
    code_ids = bits_to_code_ids(projection >= 0.0)
    table = score_table(key, m)
    layers = rules.tournament_layer_tables(key, m=m, layers=rules.tournament_layers(n)) \
        if rule == "tournament" else None
    selected = np.empty(len(prompt_uids), dtype=np.int64)
    for row, uid in enumerate(prompt_uids):
        selected[row], _ = rules.select(
            rule, code_ids[row], scores=table, layer_tables=layers,
            rng=selector_rng(rule, frame_id, key_trial, n, m, uid))
    return selected


def all_selected_sources(cache, cfg, planes: Mapping[str, np.ndarray], keys: Mapping[int, bytes],
                         rules_used: Sequence[str] = ("gumbel",)) -> "list[str]":
    """Sorted identities of every image to edit: the clean controls and each candidate ever selected.

    ``rules_used`` is ``("gumbel",)`` except for the sampling-rule comparison, which edits the
    selections of every rule.
    """
    from detection_corpus import replicate_cells

    uids = cache.prompt_uids
    sources = {source_id(uid, None) for uid in uids}
    candidates = cache.candidate_vectors[:, SLOT - 1]
    for rule in rules_used:
        for setting in cfg.settings:
            for frame_id, _, trial in replicate_cells(cfg):
                chosen = selected_indices(candidates, uids, planes[frame_id], keys[trial], frame_id=frame_id,
                                          key_trial=trial, n=setting.n, m=setting.m, rule=rule)
                sources.update(source_id(uid, int(index) + 1) for uid, index in zip(uids, chosen))
    return sorted(sources)


class EditEmbeddings:
    """Embeddings of the edited images, ``(sources, edits, D)``, looked up by source identity."""

    def __init__(self, source_ids: Sequence[str], edit_ids: Sequence[str], embeddings: np.ndarray) -> None:
        if tuple(edit_ids) != tuple(EDIT_IDS):
            raise ValueError("edit embeddings do not cover the 20 edits in canonical order")
        self.index = {sid: row for row, sid in enumerate(source_ids)}
        if len(self.index) != len(source_ids) or embeddings.shape[:2] != (len(source_ids), len(edit_ids)):
            raise ValueError("inconsistent edit embeddings")
        self.edit_ids, self.embeddings = tuple(edit_ids), embeddings

    def gather(self, ids: Iterable[str]) -> np.ndarray:
        """Edited embeddings of ``ids``: ``(len(ids), edits, D)``."""
        return self.embeddings[[self.index[sid] for sid in ids]]


def load_edit_embeddings(directory: Path) -> EditEmbeddings:
    """Merge the ``edits-shard-*.npz`` files written by ``embed_ordinary_edits.py``."""
    paths = sorted(Path(directory).glob("edits-shard-*-of-*.npz"))
    if not paths or any(int(p.stem.rsplit("-of-", 1)[1]) != len(paths) for p in paths):
        raise ValueError("edit shards are missing")
    ids: List[str] = []
    blocks = []
    edit_ids = None
    for path in paths:
        with np.load(str(path), allow_pickle=False) as data:
            ids.extend(str(value) for value in data["source_ids"].tolist())
            blocks.append(np.asarray(data["embeddings"], dtype=np.float64))
            shard_edits = tuple(str(value) for value in data["edit_ids"].tolist())
        if edit_ids not in (None, shard_edits):
            raise ValueError("edit shards disagree on the edit list")
        edit_ids = shard_edits
    return EditEmbeddings(ids, edit_ids, np.concatenate(blocks))


def edited_scores(embeddings: np.ndarray, planes: np.ndarray, m: int, table: np.ndarray, deltas: np.ndarray,
                  radius: int) -> np.ndarray:
    """Boundary-aware scores of edited images: ``(prompts, edits, D)`` -> ``(prompts, edits)``."""
    prompts, edits = embeddings.shape[:2]
    flat = embeddings.transpose(1, 0, 2).reshape(prompts * edits, -1)
    scores = score_embeddings(flat, planes, m, table, np.tile(deltas, edits), radius=radius).scores
    return scores.reshape(edits, prompts).T


def replicate_edit_rows(result: ReplicateScores, planes: np.ndarray, *, candidates: np.ndarray,
                        edits: EditEmbeddings, prompt_uids: Sequence[str], n: int, m: int, radius: int,
                        fpr_targets: Sequence[float], detector: str) -> "list[dict[str, Any]]":
    """Edited TPR, edited FPR, paired retention and cosine per edit and FPR target for one replicate.

    ``radius`` is the detector's Hamming radius (the configured one for the boundary-aware
    detector, 0 for exact-code detection); delta, tau and the scores of the unedited images
    come from the same replicate. tau is frozen on the unedited clean scores.
    """
    rows_index = np.arange(len(prompt_uids))
    wm_ids = [source_id(uid, int(i) + 1) for uid, i in zip(prompt_uids, result.selected_index)]
    clean_ids = [source_id(uid, None) for uid in prompt_uids]
    wm_edit, clean_edit = edits.gather(wm_ids), edits.gather(clean_ids)
    clean_scores = boundary_scores(result.clean_code_ids, result.clean_margins, result.table, result.deltas,
                                   radius=radius).scores
    wm_scores = boundary_scores(result.selected_code_ids, result.selected_margins, result.table, result.deltas,
                                radius=radius).scores
    wm_edit_scores = edited_scores(wm_edit, planes, m, result.table, result.deltas, radius)
    clean_edit_scores = edited_scores(clean_edit, planes, m, result.table, result.deltas, radius)
    selected = candidates[rows_index, result.selected_index]
    cosine = np.einsum("pd,ped->pe", selected, wm_edit)
    rows = []
    for target in fpr_targets:
        tau, _ = clean_threshold(clean_scores, target)
        common = {"N": n, "M": m, "detector": detector, "radius": radius, "fpr_target": target}
        rows.append({**common, "edit_id": "identity", "tpr": float(np.mean(wm_scores >= tau)),
                     "fpr": float(np.mean(clean_scores >= tau)), "retention": None, "cosine": 1.0})
        for e, edit_id in enumerate(edits.edit_ids):
            kept = paired_retention(wm_scores, wm_edit_scores[:, e], tau)
            retention = kept["paired_detection_retention"]
            rows.append({**common, "edit_id": edit_id, "tpr": float(np.mean(wm_edit_scores[:, e] >= tau)),
                         "fpr": float(np.mean(clean_edit_scores[:, e] >= tau)),
                         "retention": None if retention != retention else retention,
                         "identity_detected": kept["identity_detected_count"],
                         "retained": kept["attacked_retained_count"], "cosine": float(np.mean(cosine[:, e]))})
    return rows
