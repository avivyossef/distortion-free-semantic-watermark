"""Synthetic corpora and a small detection config for the detection-experiment tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
for name in ("generation", "robustness", "clean_detection", "detector_sweep", "encoder_ablation", "sampling_rules"):
    sys.path.insert(0, str(EXPERIMENTS / name))

from watermark.config import DetectionConfig, DetectorSetting  # noqa: E402

KEY = bytes(range(32))


def small_config(frames: int = 2, keys: int = 2, set_sizes=(1, 3)) -> DetectionConfig:
    """Two settings, two frames and two keys: a few replicates instead of the paper's grid."""
    return DetectionConfig(
        prf="hmac-sha256", settings=(DetectorSetting(8, 4, 0.25, 1), DetectorSetting(16, 8, 0.5, 1)),
        activation_population="clean_watermarked", calibration_population="clean", folds=5,
        fpr_targets=(0.05, 0.2), frames=frames, keys=keys, frame_seed_base=2026090200,
        set_sizes=tuple(set_sizes), aggregation="mean", exceptions={})


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def synthetic_cache(prompts: int = 100, slots: int = 3, n: int = 16, dim: int = 24, seed: int = 0):
    """An EmbeddingCache of clustered candidates around a per-prompt center, plus a clean control per slot."""
    from detection_corpus import EmbeddingCache

    rng = np.random.default_rng(seed)
    base = rng.normal(size=(prompts, 1, 1, dim))
    candidates = unit(base + 0.6 * rng.normal(size=(prompts, slots, n, dim)))
    cleans = unit(base[:, :, 0] + 0.6 * rng.normal(size=(prompts, slots, dim)))
    uids = tuple("p%04d" % i for i in range(prompts))
    return EmbeddingCache(uids, candidates, cleans, {"encoder": "ours"})


def synthetic_edits(cache, sources, strength: float = 0.05, seed: int = 1):
    """Edited embeddings of ``sources`` (``uid:clean`` or ``uid:candidate:NN``): small perturbations."""
    from _robustness_lib import EditEmbeddings

    from watermark.attacks import EDIT_IDS

    rng = np.random.default_rng(seed)
    base = []
    for sid in sources:
        uid = sid.split(":")[0]
        row = cache.index_of(uid)
        base.append(cache.clean_vectors[row, 0] if sid.endswith(":clean")
                    else cache.candidate_vectors[row, 0, int(sid.rsplit(":", 1)[1]) - 1])
    base = np.stack(base)
    noise = rng.normal(size=(len(sources), len(EDIT_IDS), base.shape[1]))
    return EditEmbeddings(list(sources), EDIT_IDS, unit(base[:, None, :] + strength * noise))
