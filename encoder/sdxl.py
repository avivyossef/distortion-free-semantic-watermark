"""SDXL training sources: random seeds, final images and fifth-step previews.

For every (prompt, seed) the pinned SDXL pipeline (Euler discrete scheduler, 30 steps,
guidance 7, 1024 x 1024, fp16) produces the final image and, from the same run, a
*preview*: the denoiser's predicted clean sample at the fifth scheduler call, decoded by
the VAE. Previews are noisy low-detail versions of the final image used as positive pairs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

STEP_INDEX = 4  # zero-based scheduler call whose predicted clean sample is decoded
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


class PreviewPipeline:
    """One SDXL pipeline whose scheduler is observed to capture the fifth prediction."""

    def __init__(self, pipeline: Any) -> None:
        import torch

        self.pipeline = pipeline
        self._torch = torch

    def _decode(self, latents: Any) -> Image.Image:
        pipeline, torch = self.pipeline, self._torch
        vae = pipeline.vae
        original_dtype = getattr(vae, "dtype", None)
        force_upcast = bool(getattr(vae.config, "force_upcast", False)) and original_dtype == torch.float16
        if force_upcast:
            pipeline.upcast_vae()
            post_quant_conv = getattr(vae, "post_quant_conv", None)
            if post_quant_conv is not None:
                parameter = next(post_quant_conv.parameters(), None)
                if parameter is not None:
                    latents = latents.to(dtype=parameter.dtype)
        config = vae.config
        if getattr(config, "latents_mean", None) is not None and getattr(config, "latents_std", None) is not None:
            kwargs = {"device": latents.device, "dtype": latents.dtype}
            mean = torch.tensor(config.latents_mean, **kwargs).view(1, 4, 1, 1)
            std = torch.tensor(config.latents_std, **kwargs).view(1, 4, 1, 1)
            latents = latents * std / config.scaling_factor + mean
        else:
            latents = latents / config.scaling_factor
        try:
            with torch.no_grad():
                decoded = vae.decode(latents, return_dict=False)[0]
        finally:
            if force_upcast and original_dtype is not None:
                vae.to(dtype=original_dtype)
        return pipeline.image_processor.postprocess(decoded, output_type="pil")[0].convert("RGB")

    def final_and_preview(self, prompt: str, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Run the full 30-step pipeline once; return ``(final, preview)`` PIL images."""
        scheduler = self.pipeline.scheduler
        original_step = scheduler.step
        state = getattr(scheduler, "__dict__", {})
        had_instance_step = "step" in state
        saved_instance_step = state.get("step")
        calls = 0
        prediction: Any = None

        def observed_step(model_output: Any, timestep: Any, sample: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal calls, prediction
            index = calls
            return_dict = kwargs.get("return_dict", True)
            call_kwargs = dict(kwargs)
            if return_dict is False:
                # The denoising loop asks for a tuple, which drops pred_original_sample.
                call_kwargs["return_dict"] = True
            output = original_step(model_output, timestep, sample, *args, **call_kwargs)
            calls += 1
            if index == STEP_INDEX:
                candidate = getattr(output, "pred_original_sample", None)
                if candidate is None:
                    raise RuntimeError("scheduler output lacks pred_original_sample")
                prediction = candidate.detach().clone()
            return (output.prev_sample,) if return_dict is False else output

        scheduler.step = observed_step
        try:
            generator = self._torch.Generator(device="cpu").manual_seed(seed)
            result = self.pipeline(
                prompt=prompt, generator=generator, height=IMAGE_SIZE, width=IMAGE_SIZE,
                num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE, output_type="pil",
            )
        finally:
            if had_instance_step:
                scheduler.step = saved_instance_step
            else:
                delattr(scheduler, "step")
        if prediction is None or calls != NUM_INFERENCE_STEPS:
            raise RuntimeError("pipeline did not follow the pinned 30-step timetable")
        final = result.images[0].convert("RGB")
        preview = self._decode(prediction).convert("RGB")
        if final.size != (IMAGE_SIZE, IMAGE_SIZE) or preview.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("generated image size does not match the pinned settings")
        return final, preview


def load_sdxl_pipeline(device: str = "cuda", *, cpu_offload: bool = False) -> PreviewPipeline:
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
    return PreviewPipeline(pipeline)


def write_prompt_shard(
    pipeline: PreviewPipeline, *, prompt_uid: str, prompt_split: str, text: str, seeds: Sequence[int],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Generate and write one prompt's sources; returns its manifest rows.

    Images go to ``<output_dir>/<prompt_uid>/{final,previews}/seed_<seed>.png``; paths in
    the rows are relative to ``output_dir``. Already-written seeds are not regenerated.
    """
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        final_path = Path(prompt_uid) / "final" / ("seed_%02d.png" % seed)
        preview_path = Path(prompt_uid) / "previews" / ("seed_%02d.png" % seed)
        if not (Path(output_dir) / final_path).is_file() or not (Path(output_dir) / preview_path).is_file():
            final, preview = pipeline.final_and_preview(text, seed)
            for relative, image in ((final_path, final), (preview_path, preview)):
                target = Path(output_dir) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".png")
                os.close(handle)
                image.save(temporary, format="PNG")
                os.replace(temporary, str(target))
        rows.append({
            "source_uid": source_uid(prompt_uid, seed), "prompt_uid": prompt_uid, "prompt_split": prompt_split,
            "seed": seed, "final_path": final_path.as_posix(), "preview_path": preview_path.as_posix(),
        })
    return rows
