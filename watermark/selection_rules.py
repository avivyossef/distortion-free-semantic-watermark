"""Distortion-free candidate-selection rules compared in the paper's appendix.

Three rules share one interface: given a pool's packed M-bit code IDs, keyed score
tables, and a selector RNG, each returns one original candidate index. Averaged over
keys, every rule selects each candidate with probability exactly 1/N for every fixed
pool. Only the rule comparison (``experiments/sampling_rules``) chooses among them;
every other experiment selects with :func:`watermark.selection.select_code_ids`.

- ``representative``: representative sampling (the paper's appendix Algorithm
  "Representative sampling", :func:`select_representative`): one uniform representative per
  code, the best-scoring representative with probability ``|U| / N``, otherwise a uniform
  non-representative.
- ``gumbel``: Gumbel selection, the method of the paper (:func:`watermark.selection.select_code_ids`):
  ``c* = argmax_c log F(c) / n_c``, then uniform within ``c*``. ``Pr[c* = c] = n_c / N`` over keys.
- ``tournament``: SynthID-style knockout over a uniform random bracket with an
  independent keyed score table per layer; ties (including same-code matches) break
  uniformly. Its detector statistic is the mean of the per-layer scores.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional

import numpy as np

from .selection import select_code_ids as select_gumbel

RULES = ("representative", "gumbel", "tournament")
TOURNAMENT_LAYER_DOMAIN = b"distortion-free-watermark/tournament-layer-score/v1\0"


def _groups(code_ids: np.ndarray) -> "dict[int, list[int]]":
    groups: "dict[int, list[int]]" = {}
    for index, code_id in enumerate(np.asarray(code_ids, dtype=np.int64).tolist()):
        groups.setdefault(int(code_id), []).append(index)
    return groups


def tournament_layer_tables(key: bytes, *, m: int, layers: int) -> np.ndarray:
    """Return ``layers x 2**m`` independent keyed midpoint scores in ``(0, 1)``."""
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    if not isinstance(m, int) or not 1 <= m <= 20 or not isinstance(layers, int) or layers < 1:
        raise ValueError("M must lie in [1, 20] and layers must be positive")
    size = 1 << m
    tables = np.empty((layers, size), dtype=np.float64)
    width = (m + 7) // 8
    for layer in range(layers):
        prefix = TOURNAMENT_LAYER_DOMAIN + m.to_bytes(2, "big") + layer.to_bytes(2, "big")
        for code_id in range(size):
            digest = hmac.new(key, prefix + code_id.to_bytes(width, "big"), hashlib.sha256).digest()
            tables[layer, code_id] = ((int.from_bytes(digest[:7], "big") >> 3) + 0.5) / (1 << 53)
    return tables


def tournament_layers(n: int) -> int:
    """Number of knockout layers, ``log2 N`` (N must be a power of two)."""
    if not isinstance(n, int) or n < 2 or n & (n - 1):
        raise ValueError("tournament selection requires N to be a power of two")
    return n.bit_length() - 1


def select_representative(
    code_ids: np.ndarray, scores: np.ndarray, rng: np.random.Generator
) -> "tuple[int, str, int]":
    """Representative sampling over packed code IDs and a keyed score table.

    One uniform representative is drawn per occupied code; with probability ``|U| / N`` the
    highest-scoring representative is returned (ties uniform), otherwise a uniform
    non-representative. Returns ``(selected_index, branch, distinct_code_count)`` with
    ``branch`` ``"unique_max"`` or ``"duplicate_uniform"``.
    """
    groups = _groups(code_ids)
    representatives = tuple(int(rng.choice(group)) for group in groups.values())
    representative_set = set(representatives)
    duplicates = tuple(index for index in range(len(code_ids)) if index not in representative_set)
    if not duplicates or float(rng.random()) < len(representatives) / len(code_ids):
        best_score = max(float(scores[int(code_ids[index])]) for index in representatives)
        maxima = [index for index in representatives if float(scores[int(code_ids[index])]) == best_score]
        return int(rng.choice(maxima)), "unique_max", len(groups)
    return int(rng.choice(duplicates)), "duplicate_uniform", len(groups)


def select_tournament(
    code_ids: np.ndarray, layer_tables: np.ndarray, rng: np.random.Generator
) -> "tuple[int, int]":
    """Knockout over a uniform bracket; layer l compares ``layer_tables[l]`` scores."""
    ids = np.asarray(code_ids, dtype=np.int64)
    layers = tournament_layers(len(ids))
    if layer_tables.shape[0] != layers:
        raise ValueError("layer tables do not match log2(N)")
    alive = rng.permutation(len(ids)).tolist()
    for layer in range(layers):
        table = layer_tables[layer]
        survivors = []
        for left, right in zip(alive[0::2], alive[1::2]):
            a, b = float(table[ids[left]]), float(table[ids[right]])
            if a > b:
                survivors.append(left)
            elif b > a:
                survivors.append(right)
            else:
                survivors.append(left if rng.random() < 0.5 else right)
        alive = survivors
    return int(alive[0]), len(set(ids.tolist()))


def select(
    rule: str, code_ids: np.ndarray, *, scores: np.ndarray, layer_tables: Optional[np.ndarray],
    rng: np.random.Generator,
) -> "tuple[int, int]":
    """Dispatch one selection; returns ``(selected_index, distinct_code_count)``."""
    if rule == "representative":
        selected, _, distinct = select_representative(code_ids, scores, rng)
        return selected, distinct
    if rule == "gumbel":
        return select_gumbel(code_ids, scores, rng)
    if rule == "tournament":
        if layer_tables is None:
            raise ValueError("tournament selection requires layer tables")
        return select_tournament(code_ids, layer_tables, rng)
    raise ValueError("unknown selection rule: %s" % rule)


def detector_table(rule: str, *, scores: np.ndarray, layer_tables: Optional[np.ndarray]) -> np.ndarray:
    """Per-code detector statistic: F_K for representative/gumbel, mean layer score for tournament."""
    if rule in ("representative", "gumbel"):
        return np.asarray(scores, dtype=np.float64)
    if rule == "tournament":
        if layer_tables is None:
            raise ValueError("tournament detection requires layer tables")
        return np.mean(layer_tables, axis=0)
    raise ValueError("unknown selection rule: %s" % rule)
