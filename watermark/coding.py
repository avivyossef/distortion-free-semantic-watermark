"""Stable random-hyperplane coding for normalized embeddings."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .config import HYPERPLANE_COUNT


def sample_hyperplanes(seed: int, dimension: int = 256) -> np.ndarray:
    """Create the experiment's fixed, row-normalized float64 hyperplanes."""
    if not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(dimension, Integral) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    planes = generator.standard_normal((HYPERPLANE_COUNT, int(dimension)), dtype=np.float64)
    return planes / np.linalg.norm(planes, axis=1, keepdims=True)


def validate_hyperplanes(hyperplanes: np.ndarray, dimension: "int | None" = None) -> np.ndarray:
    """Validate normalized float64 hyperplanes and return the unchanged matrix."""
    if not isinstance(hyperplanes, np.ndarray) or hyperplanes.dtype != np.float64:
        raise ValueError("hyperplanes must be a float64 numpy array")
    if hyperplanes.ndim != 2 or hyperplanes.shape[0] == 0:
        raise ValueError("hyperplanes must be a nonempty matrix")
    if dimension is not None and hyperplanes.shape[1] != dimension:
        raise ValueError("hyperplane dimension does not match embedding")
    if not np.isfinite(hyperplanes).all():
        raise ValueError("hyperplanes must be finite")
    norms = np.linalg.norm(hyperplanes, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("hyperplanes must have unit row norms")
    return hyperplanes


def code_embedding(embedding: np.ndarray, hyperplanes: np.ndarray) -> "tuple[int, ...]":
    """Return nonnegative-dot-product bits for one finite nonzero embedding."""
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("embedding must be a finite one-dimensional vector")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("embedding must have nonzero norm")
    planes = validate_hyperplanes(hyperplanes, dimension=vector.shape[0])
    normalized = vector / norm
    with np.errstate(all="ignore"):  # Apple Accelerate raises false FP-status warnings in matmul
        projection = planes @ normalized
    return tuple(int(value >= 0.0) for value in projection)


def bits_to_code_ids(bits: np.ndarray) -> np.ndarray:
    """Pack a final axis of up to 63 booleans into network-order integer code IDs."""
    values = np.asarray(bits)
    if values.ndim == 0 or values.shape[-1] == 0 or values.shape[-1] > 63:
        raise ValueError("bits must have a nonempty final axis of length at most 63")
    if values.dtype != np.bool_:
        raise ValueError("bits must be a boolean array")
    weights = np.left_shift(np.int64(1), np.arange(values.shape[-1] - 1, -1, -1, dtype=np.int64))
    return np.sum(values.astype(np.int64) * weights, axis=-1, dtype=np.int64)


def project(embeddings: np.ndarray, hyperplanes: np.ndarray, m: int) -> np.ndarray:
    """Signed projections ``<r_j, z>`` on the first ``m`` hyperplanes (last axis of the result).

    ``embeddings`` is ``(..., D)`` and unit-normalized; the sign gives the code bit, the
    absolute value the boundary margin used by the detector.
    """
    planes = validate_hyperplanes(hyperplanes, dimension=np.asarray(embeddings).shape[-1])
    if not isinstance(m, Integral) or not 1 <= m <= planes.shape[0]:
        raise ValueError("m must select existing hyperplanes")
    with np.errstate(all="ignore"):
        return np.asarray(embeddings, dtype=np.float64) @ planes[: int(m)].T
