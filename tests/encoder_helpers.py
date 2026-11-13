"""Tiny CPU stand-ins for DINOv2 and its image processor, and synthetic source images."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from encoder.model import DinoLoRAMetric

REVISION = "a" * 40


class _Attention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.attention = _Attention(hidden)


class TinyDino(nn.Module):
    """Public-contract substitute for ``Dinov2Model``: ``encoder.layer.i.attention.attention.{query,value}``."""

    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.proj = nn.Linear(48, hidden)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Block(hidden) for _ in range(12)])

    def forward(self, pixel_values: torch.Tensor, **_):
        cls = self.proj(torch.nn.functional.adaptive_avg_pool2d(pixel_values, 4).flatten(1))
        return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))


def tiny_model(variant: str = "lora_mlp", rank: int = 4) -> DinoLoRAMetric:
    torch.manual_seed(0)
    return DinoLoRAMetric.from_backbone(
        TinyDino(), lora_rank=rank, variant=variant, model_revision=REVISION,
        adapter_factory=lambda backbone, **_: backbone,
    )


class FakeProcessor:
    """Resizes to 8x8 and rescales to [-1, 1], like an image processor's ``pixel_values``."""

    def __call__(self, images, return_tensors="pt"):
        arrays = [np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 127.5 - 1.0 for image in images]
        return {"pixel_values": torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()}


def write_sources(root, prompts, seeds_per_prompt: int, *, size: int = 32, split_of=None):
    """Write synthetic final PNGs and return their manifest rows."""
    rng = np.random.default_rng(0)
    rows = []
    seed = 16
    for prompt_uid in prompts:
        split = split_of(prompt_uid) if split_of else "train"
        for _ in range(seeds_per_prompt):
            final = Path(root) / prompt_uid / "final" / ("seed_%02d.png" % seed)
            final.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)).save(final)
            rows.append({
                "source_uid": "%s:seed_%02d" % (prompt_uid, seed), "prompt_uid": prompt_uid, "prompt_split": split,
                "seed": seed, "final_path": final.relative_to(root).as_posix(),
            })
            seed += 1
    return rows

