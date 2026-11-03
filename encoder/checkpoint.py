"""Loading the trained semantic encoder from a checkpoint file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .model import DinoLoRAMetric
from .training import load_checkpoint_file, restore_exact_trainable_state


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(
    checkpoint: Path, *, expected_sha256: str, device: "str | torch.device" = "cpu",
    model: Optional[DinoLoRAMetric] = None, processor: Any = None,
) -> Tuple[DinoLoRAMetric, Any, Dict[str, Any]]:
    """Load ``(model, image_processor, provenance)`` from a checkpoint file.

    The checkpoint is a pickle (it also stores optimizer and RNG state), so
    ``expected_sha256`` is required: the hash is verified before the file is deserialized. The frozen DINOv2-base
    backbone is fetched from the Hugging Face Hub at the revision recorded in the
    checkpoint; ``model`` and ``processor`` may be supplied prebuilt (tests).
    """
    path = Path(checkpoint)
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError("checkpoint SHA-256 does not match the expected value")
    saved = load_checkpoint_file(path)
    config = saved.config
    if model is None:
        model = DinoLoRAMetric.from_pretrained(
            model_id=config["model_id"], revision=config["model_revision"],
            variant=config["variant"], lora_rank=config["lora_rank"],
        )
    if model.checkpoint_config() != config:
        raise ValueError("checkpoint configuration does not match the model")
    restore_exact_trainable_state(model, saved.model_state)
    if processor is None:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    provenance = {"checkpoint_sha256": digest, **{key: config[key] for key in ("model_id", "model_revision")},
                  "variant": config["variant"], "lora_rank": config["lora_rank"],
                  "global_update": saved.global_update}
    return model.to(device).eval(), processor, provenance
