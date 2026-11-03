"""CPU test of the matrix and plotting part of the same-prompt example figure."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "encoder_comparison"))

import entropy_example as ex  # noqa: E402


def test_cosine_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    matrix = ex.cosine_matrix(rng.normal(size=(16, 8)) * 3.0)
    assert matrix.shape == (16, 16)
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1.0)
    assert matrix.min() >= -1.0 - 1e-9 and matrix.max() <= 1.0 + 1e-9


def test_render_figure_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8) for _ in range(ex.GROUP_SIZE)]
    embeddings = {"dinov2_base": rng.normal(size=(16, 8)), "ours": rng.normal(size=(16, 8))}
    output = tmp_path / "out" / "figure.png"
    ex.render_figure(images, embeddings, output)
    assert output.stat().st_size > 0
