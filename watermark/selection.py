"""Distortion-free candidate selection (the paper's Algorithm 1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .prf import pack_code, score_code


@dataclass(frozen=True)
class SelectionTrace:
    """Selection outcome together with the collision partition it used."""

    selected_index: int
    selected_score: float
    branch: str  # "unique_max" or "duplicate_uniform"
    representatives: "tuple[int, ...]"
    duplicate_indices: "tuple[int, ...]"
    distinct_code_count: int
    max_tie_count: int


def cell_rng(*parts: object) -> np.random.Generator:
    """Selector RNG derived from a textual identity (frame, key trial, N, M, prompt, ...)."""
    message = "\0".join(str(part) for part in parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(message).digest()[:8], "big"))


def select_candidate(
    codes: Sequence[Sequence[int]], key: bytes, rng: np.random.Generator
) -> SelectionTrace:
    """Choose an original candidate index without changing its marginal law.

    ``codes`` are the candidates' bit tuples. One uniform representative is drawn per
    occupied code; with probability ``|U| / N`` the highest-scoring representative is
    returned (ties uniform), otherwise a uniform non-representative.
    """
    if not codes:
        raise ValueError("codes must contain at least one candidate")

    validated_codes = tuple(tuple(code) for code in codes)
    for code in validated_codes:
        pack_code(code)

    groups: "dict[tuple[int, ...], list[int]]" = {}
    for index, code in enumerate(validated_codes):
        groups.setdefault(code, []).append(index)
    representatives = tuple(int(rng.choice(group)) for group in groups.values())
    representative_set = set(representatives)
    duplicates = tuple(index for index in range(len(codes)) if index not in representative_set)

    if not duplicates or rng.random() < len(representatives) / len(codes):
        scores = {index: score_code(key, validated_codes[index]) for index in representatives}
        maximum_score = max(scores.values())
        maxima = [index for index, value in scores.items() if value == maximum_score]
        selected = int(rng.choice(maxima))
        branch = "unique_max"
        max_tie_count = len(maxima)
        selected_score = scores[selected]
    else:
        selected = int(rng.choice(duplicates))
        branch = "duplicate_uniform"
        max_tie_count = 0
        selected_score = score_code(key, validated_codes[selected])

    return SelectionTrace(
        selected_index=selected,
        selected_score=selected_score,
        branch=branch,
        representatives=representatives,
        duplicate_indices=duplicates,
        distinct_code_count=len(groups),
        max_tie_count=max_tie_count,
    )


def select_code_ids(
    code_ids: np.ndarray, scores: np.ndarray, rng: np.random.Generator
) -> "tuple[int, str, int]":
    """Algorithm 1 over packed integer code IDs and a precomputed keyed score table.

    Returns ``(selected_index, branch, distinct_code_count)``. Equivalent to
    :func:`select_candidate` given the same RNG state, but vectorised over a score table.
    """
    groups: "dict[int, list[int]]" = {}
    for index, code_id in enumerate(np.asarray(code_ids, dtype=np.int64).tolist()):
        groups.setdefault(int(code_id), []).append(index)
    representatives = tuple(int(rng.choice(group)) for group in groups.values())
    representative_set = set(representatives)
    duplicates = tuple(index for index in range(len(code_ids)) if index not in representative_set)
    if not duplicates or float(rng.random()) < len(representatives) / len(code_ids):
        best_score = max(float(scores[int(code_ids[index])]) for index in representatives)
        maxima = [index for index in representatives if float(scores[int(code_ids[index])]) == best_score]
        return int(rng.choice(maxima)), "unique_max", len(groups)
    return int(rng.choice(duplicates)), "duplicate_uniform", len(groups)
