from __future__ import annotations

import pytest
import torch
from torch import nn

from encoder import model as model_module
from encoder.model import DinoLoRAMetric, ModelConfigurationError, pair_loss
from encoder_helpers import REVISION, TinyDino, tiny_model


def test_output_is_unit_normalized_256_dimensional_and_head_is_trainable():
    model = tiny_model()
    z = model(pixel_values=torch.randn(3, 3, 8, 8))
    assert z.shape == (3, 256) and torch.allclose(z.norm(dim=1), torch.ones(3), atol=1e-5)
    assert any(p.requires_grad for p in model.metric_head.parameters())
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_head_architectures_match_the_variants():
    mlp, linear = tiny_model("lora_mlp"), tiny_model("lora_linear_head")
    assert [type(m).__name__ for m in mlp.metric_head] == ["Linear", "GELU", "Dropout", "Linear"]
    assert (mlp.metric_head[0].out_features, mlp.metric_head[2].p, mlp.metric_head[3].out_features) == (512, 0.1, 256)
    assert isinstance(linear.metric_head, nn.Linear) and linear.metric_head.out_features == 256
    assert mlp.checkpoint_config()["head_architecture"] == "cls_768_512_gelu_dropout01_256_l2"
    assert linear.checkpoint_config()["head_architecture"] == "cls_768_256_l2"


def test_only_the_last_four_blocks_query_value_projections_are_adapted():
    model = tiny_model()
    assert model.target_modules == tuple(
        "encoder.layer.%d.attention.attention.%s" % (block, name) for block in (8, 9, 10, 11) for name in ("query", "value"))
    assert model.checkpoint_config()["adapted_blocks"] == [8, 9, 10, 11]
    with pytest.raises(ModelConfigurationError, match="exactly"):
        DinoLoRAMetric.from_backbone(TinyDino(), lora_rank=4, adapted_blocks=(0, 8, 9, 10), model_revision=REVISION,
                                     adapter_factory=lambda b, **_: b)
    broken = TinyDino()
    broken.encoder.layer[10].attention.attention.value = nn.Identity()
    with pytest.raises(ModelConfigurationError, match="target set"):
        DinoLoRAMetric.from_backbone(broken, lora_rank=4, model_revision=REVISION, adapter_factory=lambda b, **_: b)


def test_only_rank_four_or_eight_and_pinned_revisions_are_accepted():
    with pytest.raises(ModelConfigurationError):
        DinoLoRAMetric.from_backbone(TinyDino(), lora_rank=16, model_revision=REVISION, adapter_factory=lambda b, **_: b)
    with pytest.raises(ModelConfigurationError):
        DinoLoRAMetric.from_backbone(TinyDino(), lora_rank=8, model_revision="main", adapter_factory=lambda b, **_: b)
    with pytest.raises(ModelConfigurationError):
        DinoLoRAMetric.from_backbone(TinyDino(), lora_rank=8, model_revision=REVISION, variant="full",
                                     adapter_factory=lambda b, **_: b)


def test_trainable_state_is_head_and_adapter_only_and_hashes_its_bytes():
    model = tiny_model()
    state = model.trainable_state_dict()
    assert state and all(name.startswith("metric_head.") for name in state)
    digest = model_module.trainable_state_sha256(state)
    assert digest == model_module.trainable_state_sha256(dict(reversed(list(state.items()))))
    model.metric_head[0].weight.data += 1.0
    assert digest != model_module.trainable_state_sha256(model.trainable_state_dict())


def test_paper_loss_is_alignment_for_positives_and_max_zero_cosine_for_negatives():
    assert model_module.NEGATIVE_CAP == 0.0
    assert pair_loss(torch.tensor([1.0]), positive=True).item() == 0
    assert pair_loss(torch.tensor([0.4]), positive=True).item() == pytest.approx(0.6)
    assert pair_loss(torch.tensor([-0.3]), positive=False).item() == 0
    assert pair_loss(torch.tensor([0.4]), positive=False).item() == pytest.approx(0.4)
    assert pair_loss(torch.tensor([0.8]), positive=False, negative_cap=0.25).item() == pytest.approx(0.55)
    with pytest.raises(ValueError):
        pair_loss(torch.tensor([0.1]), positive=False, negative_cap=2.0)
