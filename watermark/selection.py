"""Distortion-free candidate selection by Gumbel selection (the method of every experiment).

Candidates sharing an ``M``-bit code ``c`` share the keyed score ``F_kappa(c)``. With
``I_c`` the candidates of code ``c`` and ``n_c = |I_c|``, the selector returns a uniform member of
``I_{c*}`` for ``c* = argmax_c log F_kappa(c) / n_c`` (ties uniform). Averaged over keys, code
``c`` wins with probability ``n_c / N``, so every candidate is returned with probability ``1/N``.
The paper's alternative rules (representative sampling, Tournament) live in
:mod:`watermark.selection_rules` and are used only by the rule comparison.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import numpy as np

from .prf import pack_code, score_code


@dataclass(frozen=True)
class SelectionTrace:
    """Selection outcome of one candidate pool."""

    selected_index: int
    selected_score: float
    distinct_code_count: int
    tie_count: int  # codes tied for the maximum of log F / n_c (1 unless scores collide)


def cell_rng(*parts: object) -> np.random.Generator:
    """Selector RNG derived from a textual identity (frame, key trial, N, M, prompt, ...)."""
    message = "\0".join(str(part) for part in parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(message).digest()[:8], "big"))


def _race(groups: "Sequence[list[int]]", scores: "Sequence[float]", rng: np.random.Generator) -> "tuple[int, int]":
    """Exponential race: the group maximizing ``log(score) / size`` wins; a uniform member returns.

    ``groups`` lists the candidate indices of every occupied code in order of first occurrence and
    ``scores`` the keyed score of each group's code. Returns ``(selected_index, tie_count)``.
    """
    values = [float(np.log(score)) / len(members) for score, members in zip(scores, groups)]
    best = max(values)
    winners = [position for position, value in enumerate(values) if value == best]
    winner = winners[0] if len(winners) == 1 else int(rng.choice(winners))
    return int(rng.choice(groups[winner])), len(winners)


def select_code_ids(
    code_ids: np.ndarray, scores: Union[np.ndarray, Mapping[int, float]], rng: np.random.Generator
) -> "tuple[int, int]":
    """Gumbel selection over packed integer code IDs and a keyed score table.

    ``scores`` maps a code ID to its keyed score (an array indexed by code ID, e.g.
    :func:`watermark.prf.score_table`, or a mapping). Returns ``(selected_index,
    distinct_code_count)``.
    """
    groups: "dict[int, list[int]]" = {}
    for index, code_id in enumerate(np.asarray(code_ids, dtype=np.int64).tolist()):
        groups.setdefault(int(code_id), []).append(index)
    selected, _ = _race(list(groups.values()), [scores[code] for code in groups], rng)
    return selected, len(groups)


def select_candidate(
    codes: Sequence[Sequence[int]], key: bytes, rng: np.random.Generator
) -> SelectionTrace:
    """Gumbel selection over the candidates' bit tuples (any code length up to 64 bits).

    Same draws as :func:`select_code_ids` on the packed codes and the scores
    :func:`watermark.prf.score_code` gives each code under ``key``.
    """
    if not codes:
        raise ValueError("codes must contain at least one candidate")
    validated = tuple(tuple(code) for code in codes)
    for code in validated:
        pack_code(code)
    groups: "dict[tuple[int, ...], list[int]]" = {}
    for index, code in enumerate(validated):
        groups.setdefault(code, []).append(index)
    group_scores = [score_code(key, code) for code in groups]
    selected, ties = _race(list(groups.values()), group_scores, rng)
    return SelectionTrace(
        selected_index=selected,
        selected_score=score_code(key, validated[selected]),
        distinct_code_count=len(groups),
        tie_count=ties,
    )
