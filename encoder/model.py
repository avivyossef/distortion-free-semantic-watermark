"""The semantic encoder: frozen DINOv2-base, rank-r LoRA on the query/value projections of
the last four blocks, and a 768-512-256 MLP head with GELU and dropout 0.1, followed by
L2 normalization.

The optional PEFT dependency is imported only when a real LoRA adapter is requested, which
keeps data tests CPU-only and makes a missing dependency an explicit error rather than a
silent head-only fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import hashlib
import re
from typing import Any, Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DINO_MODEL_ID = "facebook/dinov2-base"
LORA_RANK_CANDIDATES = (4, 8)
DEFAULT_ADAPTED_BLOCKS = (8, 9, 10, 11)
NEGATIVE_CAP = 0.0
MetricVariant = Literal["head_only", "lora_linear_head", "lora_mlp"]


class ModelConfigurationError(ValueError):
    """Raised before an adapter can be attached to an unexpected backbone."""


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def trainable_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash the exact ordered adapter/head tensor bytes for artifact binding."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("trainable state must map names to tensors")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _expected_target_names(adapted_blocks: Iterable[int]) -> tuple[str, ...]:
    blocks = tuple(adapted_blocks)
    if blocks != DEFAULT_ADAPTED_BLOCKS:
        raise ModelConfigurationError(
            "DINOv2 LoRA must adapt exactly blocks (8, 9, 10, 11) query/value targets"
        )
    return tuple(
        f"encoder.layer.{block}.attention.attention.{projection}"
        for block in blocks
        for projection in ("query", "value")
    )


def discover_qv_targets(backbone: nn.Module, *, adapted_blocks: Iterable[int]) -> tuple[str, ...]:
    """Return the exact query/value Linear modules allowed to receive LoRA.

    DINOv2-base exposes ``encoder.layer.<i>.attention.attention.{query,value}``.
    Checking both module type and complete name prevents a PEFT suffix match
    from broadening adaptation to another layer or projection.
    """
    expected = _expected_target_names(adapted_blocks)
    linear_names = {name for name, module in backbone.named_modules() if isinstance(module, nn.Linear)}
    discovered = tuple(name for name in expected if name in linear_names)
    if set(discovered) != set(expected):
        missing = sorted(set(expected) - set(discovered))
        unexpected = sorted(
            name for name in linear_names
            if name.endswith(".query") or name.endswith(".value")
            if any(f"encoder.layer.{block}." in name for block in tuple(adapted_blocks))
            and name not in expected
        )
        raise ModelConfigurationError(
            f"DINOv2 LoRA target set differs from configured set; missing={missing}, unexpected={unexpected}"
        )
    return discovered


def _peft_adapter_factory(backbone: nn.Module, *, target_modules: tuple[str, ...], lora_rank: int) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:  # pragma: no cover - depends on local optional package state
        raise RuntimeError(
            "DINOv2 LoRA requires peft>=0.15.0; install the project dependencies (pip install peft transformers)"
        ) from error
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(target_modules),
    )
    return get_peft_model(backbone, config)


class DinoLoRAMetric(nn.Module):
    """Frozen DINOv2 with optional Q/V LoRA and a normalized 256-D metric."""

    def __init__(
        self,
        backbone: nn.Module,
        metric_head: nn.Module,
        *,
        target_modules: tuple[str, ...],
        variant: MetricVariant,
        lora_rank: int,
        model_id: str,
        model_revision: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.metric_head = metric_head
        self.target_modules = target_modules
        self.variant = variant
        self.lora_rank = lora_rank
        self.model_id = model_id
        self.model_revision = model_revision

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        *,
        lora_rank: int = 4,
        adapted_blocks: Iterable[int] = DEFAULT_ADAPTED_BLOCKS,
        variant: MetricVariant = "lora_mlp",
        model_id: str = DINO_MODEL_ID,
        model_revision: str | None = None,
        adapter_factory: Callable[..., nn.Module] | None = None,
    ) -> "DinoLoRAMetric":
        if variant not in {"head_only", "lora_linear_head", "lora_mlp"}:
            raise ModelConfigurationError(f"unsupported metric variant: {variant}")
        if lora_rank not in LORA_RANK_CANDIDATES:
            raise ModelConfigurationError(f"lora rank must be one of {LORA_RANK_CANDIDATES}")
        if model_id != DINO_MODEL_ID:
            raise ModelConfigurationError(f"The semantic encoder only permits the pinned model id {DINO_MODEL_ID}")
        if not isinstance(model_revision, str) or not _IMMUTABLE_REVISION.fullmatch(model_revision):
            raise ModelConfigurationError("DINOv2 model_revision must be an immutable 40-character lowercase commit")
        hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ModelConfigurationError("backbone must expose a positive config.hidden_size")

        # Freeze the pretrained DINO parameters before PEFT adds new adapter
        # parameters.  Adapter weights remain the only trainable encoder state.
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        blocks = tuple(adapted_blocks)
        _expected_target_names(blocks)
        target_modules: tuple[str, ...] = ()
        if variant != "head_only":
            target_modules = discover_qv_targets(backbone, adapted_blocks=blocks)
            factory = adapter_factory or _peft_adapter_factory
            backbone = factory(backbone, target_modules=target_modules, lora_rank=lora_rank)
            for name, parameter in backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)
                elif parameter.requires_grad:
                    # A custom adapter factory is not allowed to unfreeze DINO.
                    parameter.requires_grad_(False)
        if variant == "lora_linear_head":
            metric_head: nn.Module = nn.Linear(hidden_size, 256)
        else:
            metric_head = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
            )
        return cls(
            backbone,
            metric_head,
            target_modules=target_modules,
            variant=variant,
            lora_rank=lora_rank,
            model_id=model_id,
            model_revision=model_revision,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = DINO_MODEL_ID,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "DinoLoRAMetric":
        """Load the pinned DINOv2-base backbone from the Hugging Face Hub.

        ``revision`` defaults to the revision pinned in ``configs/assets.yaml``.
        Transformers is imported here so CPU tests need no model download.
        """
        if revision is None:
            from watermark.config import load_assets

            revision = load_assets()["models"]["dinov2_base"]["revision"]
        if model_id != DINO_MODEL_ID:
            raise ModelConfigurationError(f"The semantic encoder only permits the pinned model id {DINO_MODEL_ID}")
        if not isinstance(revision, str) or not _IMMUTABLE_REVISION.fullmatch(revision):
            raise ModelConfigurationError("DINOv2 revision must be an immutable 40-character lowercase commit")
        try:
            from transformers import Dinov2Model
        except ImportError as error:  # pragma: no cover - local dependency gate
            raise RuntimeError(
                "DINOv2 loading requires transformers; install the project dependencies (pip install peft transformers)"
            ) from error
        backbone = Dinov2Model.from_pretrained(model_id, revision=revision)
        return cls.from_backbone(
            backbone,
            model_id=model_id,
            model_revision=revision,
            **kwargs,
        )

    def forward(self, *, pixel_values: Tensor, **kwargs: Any) -> Tensor:
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise RuntimeError("DINOv2 backbone did not return last_hidden_state [batch, tokens, hidden]")
        return F.normalize(self.metric_head(hidden[:, 0]), p=2, dim=1)

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Return only adapter/head state; frozen base-DINO is always reloaded pinned."""
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name.startswith("metric_head.") or ".lora_" in name
        }

    def checkpoint_config(self) -> dict[str, Any]:
        """Immutable model and adapter/head architecture used for safe resume."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "variant": self.variant,
            "lora_rank": self.lora_rank,
            "adapted_blocks": list(DEFAULT_ADAPTED_BLOCKS),
            "head_architecture": (
                "cls_768_256_l2"
                if self.variant == "lora_linear_head"
                else "cls_768_512_gelu_dropout01_256_l2"
            ),
        }


def pair_loss(cosine: Tensor, *, positive: bool, negative_cap: float = NEGATIVE_CAP) -> Tensor:
    """The training objective: ``1 - cos`` for positive pairs, ``max(0, cos - cap)`` for negatives.

    The paper's negative loss is ``max(0, cos)``, i.e. ``negative_cap = 0``.
    """
    if not 0.0 <= negative_cap <= 1.0:
        raise ValueError("negative_cap must lie in [0, 1]")
    if positive:
        return (1.0 - cosine).mean()
    return torch.relu(cosine - negative_cap).mean()
