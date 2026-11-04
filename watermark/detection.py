"""Boundary-aware detection (paper Algorithms "Per-image score computation" and "Detection from multiple images") and per-replicate scoring.

A replicate is one (hyperplane frame, PRF key) pair. :func:`replicate_scores` runs the
whole clean-detection pass for one replicate and one pool of candidate embeddings:
selection, delta calibration on the watermarked margins, and boundary-aware scores of
both the clean controls and the selected watermarked images. Set scores are the mean of
the per-image scores (:func:`aggregate_scores`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from . import selection_rules as rules
from .calibration import crossfit_deltas
from .coding import bits_to_code_ids
from .prf import score_table
from .selection import cell_rng


def boundary_max_score(
    code_id: int, margins: np.ndarray, scores: np.ndarray, *, delta: float, radius: int
) -> "tuple[float, int, int]":
    """Largest keyed score over the boundary-restricted Hamming neighbourhood.

    ``J_delta`` is the set of bits whose margin is at most ``delta``; the neighbourhood
    holds the observed code and every code obtained by flipping at most ``radius`` bits
    of ``J_delta``. Returns ``(score, neighbourhood_size, |J_delta|)``.
    """
    uncertain = tuple(int(index) for index in np.flatnonzero(margins <= delta))
    candidate_ids = [int(code_id)]
    for flips in range(1, min(radius, len(uncertain)) + 1):
        for chosen in combinations(uncertain, flips):
            candidate = int(code_id)
            for index in chosen:
                candidate ^= 1 << (len(margins) - index - 1)
            candidate_ids.append(candidate)
    return float(np.max(scores[candidate_ids])), len(candidate_ids), len(uncertain)


@dataclass(frozen=True)
class BoundaryScores:
    scores: np.ndarray  # (images,)
    neighbourhood_sizes: np.ndarray  # (images,) codes scored; > 1 means the boundary was activated
    uncertain_bits: np.ndarray  # (images,) |J_delta|


def boundary_scores(
    code_ids: np.ndarray, margins: np.ndarray, table: np.ndarray, deltas: np.ndarray, *, radius: int
) -> BoundaryScores:
    """Boundary-aware score of every image given its code ID, margins and delta."""
    count = len(code_ids)
    scores = np.empty(count, dtype=np.float64)
    sizes = np.empty(count, dtype=np.int64)
    uncertain = np.empty(count, dtype=np.int64)
    for row in range(count):
        scores[row], sizes[row], uncertain[row] = boundary_max_score(
            int(code_ids[row]), margins[row], table, delta=float(deltas[row]), radius=int(radius)
        )
    return BoundaryScores(scores, sizes, uncertain)


def score_embeddings(
    embeddings: np.ndarray, planes: np.ndarray, m: int, table: np.ndarray, deltas: np.ndarray, *, radius: int
) -> BoundaryScores:
    """Boundary-aware score of unit-normalized embeddings (e.g. edited images) ``(images, D)``."""
    with np.errstate(all="ignore"):
        projection = np.asarray(embeddings, dtype=np.float64) @ planes[:m].T
    return boundary_scores(
        bits_to_code_ids(projection >= 0.0), np.abs(projection), table, deltas, radius=radius
    )


def aggregate_scores(per_image_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Set score of ``B`` images: the mean of their per-image scores, element-wise over sets."""
    return np.mean(np.stack([np.asarray(item, dtype=np.float64) for item in per_image_scores]), axis=0)


@dataclass(frozen=True)
class ReplicateScores:
    """Scores of one replicate over a pool of prompts (one image per prompt and role)."""

    clean: BoundaryScores
    watermarked: BoundaryScores
    deltas: np.ndarray
    selected_index: np.ndarray
    selected_code_ids: np.ndarray
    clean_code_ids: np.ndarray
    selected_margins: np.ndarray  # (prompts, M)
    clean_margins: np.ndarray  # (prompts, M)
    distinct_codes: np.ndarray
    table: np.ndarray  # per-code detector statistic used for scoring


DETECTION_STREAM = "selection-rule-comparison/v1"
EDIT_STREAM = "selection-rule-comparison/attacks/v1"


def selector_rng(rule: str, frame_id: str, key_trial: int, n: int, m: int, uid: str,
                 slot: Optional[int] = None) -> np.random.Generator:
    """Selector RNG of one prompt, with the stream identities of the paper's runs.

    ``slot=None`` is the stream of the experiments that edit or re-embed the selected images
    (robustness, detector sweep and ablation, encoder ablation, CSD / depth, generative edits):
    ``EDIT_STREAM`` with the bare prompt identity. ``slot=s`` is the stream of the set-size
    experiments (clean detection, rule comparison): ``DETECTION_STREAM`` with the identity
    ``<uid>:slot:<s>`` of the s-th independent pool of the prompt. ``rule`` is part of the Gumbel and
    Tournament identity; representative sampling keeps the identity of the sweep it was run in.
    """
    identity = uid if slot is None else "%s:slot:%d" % (uid, slot)
    if rule == "representative":
        return cell_rng("candidate-pool-sweep", frame_id, key_trial, "hmac", n, m, identity)
    return cell_rng(EDIT_STREAM if slot is None else DETECTION_STREAM, rule, frame_id, key_trial, n, m, identity)


def replicate_scores(
    candidates: np.ndarray, cleans: np.ndarray, prompt_uids: Sequence[str], planes: np.ndarray,
    key: bytes, *, frame_id: str, key_trial: int, n: int, m: int,
    activation: float, radius: int, folds: int, rule: str = "gumbel", slot: Optional[int] = None,
) -> ReplicateScores:
    """Select, calibrate and score one (frame, key) replicate.

    ``candidates`` is ``(prompts, N, D)`` and ``cleans`` ``(prompts, D)``, both
    unit-normalized embeddings. ``slot`` selects the selector stream
    (see :func:`selector_rng`): ``None`` for the experiments that edit the selected images, a slot
    number when several independent pools per prompt are scored (set sizes ``B > 1``). ``rule`` is
    ``"gumbel"``, the paper's method, except in the rule comparison. Delta is
    cross-fitted over ``folds`` folds on the margins of the selected watermarked images,
    the paper's activation population, and applied to the clean controls of the same
    prompts.
    """
    prompts = tuple(str(uid) for uid in prompt_uids)
    if candidates.ndim != 3 or candidates.shape[:2] != (len(prompts), n):
        raise ValueError("candidate embeddings do not match prompt count and N")
    if cleans.shape != (len(prompts), candidates.shape[2]):
        raise ValueError("clean embeddings do not match candidates")
    if not 1 <= m <= planes.shape[0] or len(set(prompts)) != len(prompts):
        raise ValueError("invalid M or prompt identities")
    with np.errstate(all="ignore"):
        candidate_projection = np.einsum("pnd,md->pnm", candidates, planes[:m], optimize=True)
        clean_projection = cleans @ planes[:m].T
    candidate_ids = bits_to_code_ids(candidate_projection >= 0.0)
    clean_ids = bits_to_code_ids(clean_projection >= 0.0)
    scores = score_table(key, m)
    layer_tables = (
        rules.tournament_layer_tables(key, m=m, layers=rules.tournament_layers(n))
        if rule == "tournament" else None
    )
    table = rules.detector_table(rule, scores=scores, layer_tables=layer_tables)

    selected = np.empty(len(prompts), dtype=np.int64)
    distinct = np.empty(len(prompts), dtype=np.int64)
    for row, prompt_uid in enumerate(prompts):
        selected[row], distinct[row] = rules.select(
            rule, candidate_ids[row], scores=scores, layer_tables=layer_tables,
            rng=selector_rng(rule, frame_id, key_trial, n, m, prompt_uid, slot),
        )
    rows = np.arange(len(prompts))
    selected_ids = candidate_ids[rows, selected]
    selected_margins = np.abs(candidate_projection[rows, selected, :m])
    clean_margins = np.abs(clean_projection[:, :m])
    deltas = crossfit_deltas(selected_margins, activation, folds)
    return ReplicateScores(
        clean=boundary_scores(clean_ids, clean_margins, table, deltas, radius=radius),
        watermarked=boundary_scores(selected_ids, selected_margins, table, deltas, radius=radius),
        deltas=deltas, selected_index=selected, selected_code_ids=selected_ids, clean_code_ids=clean_ids,
        selected_margins=selected_margins, clean_margins=clean_margins, distinct_codes=distinct, table=table,
    )
