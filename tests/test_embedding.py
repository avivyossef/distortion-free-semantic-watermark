from __future__ import annotations

import numpy as np
import pytest
import torch

from watermark import embedding


def test_normalize_returns_unit_rows_and_rejects_bad_output():
    out = embedding.normalize(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    assert out.dtype == np.float64 and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[0], [0.6, 0.8])
    with pytest.raises(ValueError):
        embedding.normalize(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        embedding.normalize(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        embedding.normalize(np.ones(3))


def test_global_embedding_selection_handles_tensor_and_wrapped_outputs():
    class Wrapped:
        def __init__(self, pooled=None, hidden=None):
            self.pooler_output, self.last_hidden_state = pooled, hidden

    pooled = torch.ones(2, 5)
    assert embedding._select_embedding_output(pooled) is pooled
    assert embedding._select_embedding_output(Wrapped(pooled=pooled)) is pooled
    hidden = torch.arange(24.0).reshape(2, 3, 4)
    assert torch.equal(embedding._select_embedding_output(hidden), hidden[:, 0, :])
    assert torch.equal(embedding._select_embedding_output(Wrapped(hidden=hidden)), hidden[:, 0, :])
    with pytest.raises(ValueError):
        embedding._select_embedding_output(object())


def test_depth_descriptor_pools_to_a_zero_mean_grid():
    depth = torch.arange(2 * 32 * 32, dtype=torch.float32).reshape(2, 32, 32)
    descriptor = embedding.depth_descriptor(depth)
    assert descriptor.shape == (2, 256) and np.allclose(descriptor.mean(axis=1), 0.0, atol=1e-9)
    # A per-image scale and shift of relative depth does not change the normalized descriptor.
    shifted = embedding.normalize(embedding.depth_descriptor(depth * 3.0 + 7.0))
    assert np.allclose(shifted, embedding.normalize(descriptor))


def test_embed_paths_batches_and_preserves_order(tmp_path):
    from PIL import Image

    paths = []
    for index in range(5):
        path = tmp_path / ("%d.png" % index)
        Image.new("RGB", (4, 4), (index * 40, 0, 0)).save(path)
        paths.append(path)

    class MeanRed:
        calls = []

        def embed(self, images):
            self.calls.append(len(images))
            return embedding.normalize(np.array([[np.asarray(i)[0, 0, 0] + 1.0, 1.0] for i in images]))

    encoder = MeanRed()
    out = embedding.embed_paths(encoder, paths, batch_size=2)
    assert encoder.calls == [2, 2, 1] and out.shape == (5, 2)
    assert np.all(np.diff(out[:, 0]) > 0)


def test_load_encoder_rejects_unknown_names_and_missing_checkpoint():
    with pytest.raises(ValueError):
        embedding.load_encoder("resnet", device="cpu")
    with pytest.raises(ValueError):
        embedding.load_encoder("ours", device="cpu")
    assert set(embedding.ENCODER_NAMES) == {"ours", "dinov2_base", "clip_vit_b32", "dinov3_l16",
                                            "tpips_embed", "csd_style", "depth_layout"}


def test_trained_encoder_checkpoint_must_match_the_released_hash(tmp_path):
    path = tmp_path / "other.pt"
    path.write_bytes(b"not the released checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        embedding.load_encoder("ours", device="cpu", checkpoint=path)


def test_trained_encoder_checkpoint_is_checked_against_a_given_hash(tmp_path):
    path = tmp_path / "own.pt"
    path.write_bytes(b"a locally trained checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        embedding.load_encoder("ours", device="cpu", checkpoint=path, checkpoint_sha256="0" * 64)
