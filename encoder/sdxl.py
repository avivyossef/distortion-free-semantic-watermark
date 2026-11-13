"""SDXL training sources: random seeds and final images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.0
IMAGE_SIZE = 1024
SEED_MINIMUM = 16
SEED_EXCLUSIVE_MAXIMUM = 2 ** 63


def allocate_seeds(
    prompt_uids: Sequence[str], *, registry_seed: int, seeds_per_prompt: int = 64
) -> Dict[str, List[int]]:
    """Disjoint random SDXL seeds, ``seeds_per_prompt`` per prompt, in allocation order.

    One PCG64 stream drawn for the prompts in sorted order guarantees that no seed is
    reused across prompts. The first ``k`` seeds of a prompt are a prefix of the first
    ``2k``, so a stage-wise curriculum (8, 16, 32, 64 seeds per prompt) only appends.
    """
    if len(set(prompt_uids)) != len(prompt_uids) or not prompt_uids:
        raise ValueError("prompt identities must be unique and nonempty")
    if not 0 <= registry_seed < SEED_EXCLUSIVE_MAXIMUM or seeds_per_prompt < 1:
        raise ValueError("invalid registry seed or seeds per prompt")
    generator = np.random.Generator(np.random.PCG64(registry_seed))
    allocated = set()
    result: Dict[str, List[int]] = {}
    for prompt_uid in sorted(prompt_uids):
        seeds: List[int] = []
        while len(seeds) < seeds_per_prompt:
            seed = int(generator.integers(SEED_MINIMUM, SEED_EXCLUSIVE_MAXIMUM, endpoint=False, dtype=np.int64))
            if seed not in allocated:
                allocated.add(seed)
                seeds.append(seed)
        result[prompt_uid] = seeds
    return result


def source_uid(prompt_uid: str, seed: int) -> str:
    return "%s:seed_%02d" % (prompt_uid, seed)


class SDXLSourcePipeline:
    """Generate one final image from a prompt and seed."""

    def __init__(self, pipeline: Any) -> None:
        import torch

        self.pipeline = pipeline
        self._torch = torch

    def generate(self, prompt: str, seed: int) -> Image.Image:
        generator = self._torch.Generator(device="cpu").manual_seed(seed)
        result = self.pipeline(
            prompt=prompt, generator=generator, height=IMAGE_SIZE, width=IMAGE_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE, output_type="pil",
        )
        final = result.images[0].convert("RGB")
        if final.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("generated image size does not match the pinned settings")
        return final


def load_sdxl_pipeline(device: str = "cuda", *, cpu_offload: bool = False) -> SDXLSourcePipeline:
    """Load the pinned SDXL base model (fp16 weights, Euler discrete scheduler)."""
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from watermark.config import load_assets

    spec = load_assets()["models"]["sdxl"]
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        spec["repo"], revision=spec["revision"], torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    if cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    return SDXLSourcePipeline(pipeline)


def write_prompt_shard(
    pipeline: SDXLSourcePipeline, *, prompt_uid: str, prompt_split: str, text: str, seeds: Sequence[int],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Generate and write one prompt's sources; returns its manifest rows.

    Images go to ``<output_dir>/<prompt_uid>/final/seed_<seed>.png``; paths in
    the rows are relative to ``output_dir``. Already-written seeds are not regenerated.
    """
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        final_path = Path(prompt_uid) / "final" / ("seed_%02d.png" % seed)
        target = Path(output_dir) / final_path
        if not target.is_file():
            final = pipeline.generate(text, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".png")
            os.close(handle)
            temporary_path = Path(temporary)
            try:
                final.save(temporary_path, format="PNG")
                temporary_path.replace(target)
            finally:
                temporary_path.unlink(missing_ok=True)
        rows.append({
            "source_uid": source_uid(prompt_uid, seed), "prompt_uid": prompt_uid, "prompt_split": prompt_split,
            "seed": seed, "final_path": final_path.as_posix(),
        })
    return rows
