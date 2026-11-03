"""Exact HMAC-SHA256 scoring for binary watermark codes, plus key/seed derivation."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from numbers import Integral
from typing import Sequence

import numpy as np


SCORE_DOMAIN = b"distortion-free-watermark/score/v1\0"
DERIVE_DOMAIN = b"distortion-free-watermark/derive/v1\0"
TRIAL_KEY_DOMAIN = b"distortion-free-watermark/candidate-pool-sweep/key-trial/v1\0"
_MAX_CODE_LENGTH = 64


def _validate_code(code: Sequence[int]) -> "tuple[int, ...]":
    try:
        bits = tuple(code)
    except TypeError as error:
        raise ValueError("code must be a binary sequence") from error
    if not bits or len(bits) > _MAX_CODE_LENGTH:
        raise ValueError("code length must be between 1 and 64")
    if any(not isinstance(bit, Integral) or int(bit) not in (0, 1) for bit in bits):
        raise ValueError("code must contain only binary bits")
    return tuple(int(bit) for bit in bits)


def pack_code(code: Sequence[int]) -> bytes:
    """Pack binary bits in network order with zero-filled trailing low bits."""
    bits = _validate_code(code)
    packed = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        packed[index // 8] |= bit << (7 - (index % 8))
    return bytes(packed)


def score_code(key: bytes, code: Sequence[int]) -> float:
    """Score a binary code as an exact 53-bit uniform midpoint in ``(0, 1)``."""
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    bits = _validate_code(code)
    message = SCORE_DOMAIN + len(bits).to_bytes(2, "big") + pack_code(bits)
    digest = hmac.new(key, message, hashlib.sha256).digest()
    z = int.from_bytes(digest[:7], "big") >> 3
    return (z + 0.5) / (1 << 53)


def code_bits(code_id: int, m: int) -> "tuple[int, ...]":
    """Unpack a packed integer code ID into its ``m`` network-order bits."""
    return tuple((int(code_id) >> (m - index - 1)) & 1 for index in range(m))


def score_table(key: bytes, m: int) -> np.ndarray:
    """Keyed score of every M-bit code ID, indexed by packed integer code."""
    return np.asarray([score_code(key, code_bits(code_id, m)) for code_id in range(1 << m)])


def generate_master_key() -> bytes:
    """Draw a fresh 32-byte master key. Keep it secret; never commit it."""
    return secrets.token_bytes(32)


def derive_trial_key(master: bytes, trial: int) -> bytes:
    """Key for replicate ``trial``: trial 0 is the master key, later trials are HMAC-derived."""
    if not isinstance(master, bytes) or len(master) != 32:
        raise ValueError("master key must contain exactly 32 bytes")
    if not isinstance(trial, int) or isinstance(trial, bool) or trial < 0:
        raise ValueError("key trial must be a nonnegative integer")
    if trial == 0:
        return master
    return hmac.new(master, TRIAL_KEY_DOMAIN + trial.to_bytes(4, "big"), hashlib.sha256).digest()


def derive_digest(master_seed: bytes, *parts: str) -> bytes:
    """Return a domain-separated digest for a fixed experiment purpose."""
    if (
        not isinstance(master_seed, bytes)
        or len(master_seed) != 32
        or any(not isinstance(part, str) or "\0" in part for part in parts)
    ):
        raise ValueError("invalid derivation input")
    message = DERIVE_DOMAIN + b"\0".join(part.encode("utf-8") for part in parts)
    return hmac.new(master_seed, message, hashlib.sha256).digest()


def derive_seed(master_seed: bytes, *parts: str) -> int:
    """Derive an unsigned 64-bit generation seed."""
    return int.from_bytes(derive_digest(master_seed, *parts)[:8], "big", signed=False)
