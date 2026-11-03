"""Stable Diffusion 2.1 candidate generation with the seed policy of the paper corpus.

Per prompt the corpus holds, for ``mode="candidates"``, ``batch_size`` independent slots,
each with ``candidate_count`` candidate images plus one clean (unwatermarked) control, and,
for ``mode="references"``, ``candidate_count`` independent reference images. A prompt's
images are written to ``<output_dir>/<prompt_uid>/`` with a ``panel.json`` record::

    {"prompt_uid", "mode", "prompt", "model_id", "model_revision", "num_inference_steps",
     "guidance_scale", "image_length", "batch_slots": [{"slot", "candidate_images":
     [{"candidate_position", "relative_path", "file_sha256", "pixel_sha256"}], "clean":
     {"relative_path", ...}}]}        # references: "reference_images" instead of slots

Seeds are derived from a 32-byte master key (:func:`watermark.prf.derive_seed`) and the
position index alone, so raising ``candidate_count`` leaves the seeds of earlier positions
untouched (``candidate_start`` generates only the new positions).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import load_assets
from .io import canonical_json, pixel_sha256, sha256_file
from .prf import derive_seed

SELECTION_KEY_ID = "key-01"
NATIVE_KEY_ID = "key-00"
IMAGE_LENGTH = 512
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
GENERATION_BATCH_SIZE = 4


def _native_clean_seed(prompt_uid: str, key_id: str) -> int:
    digest = hashlib.sha256(
        ("sd21-semantic-baseline/v1\0%s\0%s" % (prompt_uid, key_id)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def panel_seed_rows(
    *, prompt_uid: str, master_key: bytes, mode: str, batch_size: int = 1, candidate_count: int = 16,
    candidate_start: int = 1, include_clean: bool = True,
    selection_key_id: str = SELECTION_KEY_ID, native_key_id: str = NATIVE_KEY_ID,
) -> "list[dict[str, Any]]":
    """Exact candidate/clean or independent-reference seeds for one prompt.

    The clean control of slot 1 uses the native baseline seed (a function of the prompt
    and ``native_key_id`` only); later slots derive theirs from ``master_key``.
    ``include_clean=False`` suppresses clean images so an extension run cannot emit a
    second, conflicting clean identity.
    """
    if (not prompt_uid or not native_key_id or not selection_key_id or len(master_key) != 32
            or not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError("invalid panel seed inputs")
    if not isinstance(candidate_count, int) or candidate_count <= 0:
        raise ValueError("candidate_count must be a positive integer")
    if not isinstance(candidate_start, int) or not 1 <= candidate_start <= candidate_count:
        raise ValueError("candidate_start must be within 1..candidate_count")
    if mode == "candidates":
        rows: "list[dict[str, Any]]" = []
        for slot in range(1, batch_size + 1):
            for position in range(candidate_start, candidate_count + 1):
                domain = "pmark-sd21-candidate/v1" if slot == 1 else "pmark-sd21-candidate-batch/v1"
                parts = (prompt_uid, selection_key_id, str(position)) if slot == 1 else (
                    prompt_uid, selection_key_id, str(slot), str(position)
                )
                rows.append({
                    "role": "candidate", "batch_slot": slot, "position": position,
                    "key_id": selection_key_id, "seed": derive_seed(master_key, domain, *parts),
                })
            if not include_clean:
                continue
            rows.append({
                "role": "clean", "batch_slot": slot, "position": candidate_count + 1, "key_id": native_key_id,
                "seed": (
                    _native_clean_seed(prompt_uid, native_key_id)
                    if slot == 1 else derive_seed(
                        master_key, "pmark-sd21-clean-batch/v1", prompt_uid, native_key_id, str(slot)
                    )
                ),
            })
    elif mode == "references":
        rows = [
            {
                "role": "reference", "batch_slot": 1, "position": position, "key_id": selection_key_id,
                "seed": derive_seed(
                    master_key, "pmark-sd21-reference/v1", prompt_uid, selection_key_id, str(position)
                ),
            }
            for position in range(candidate_start, candidate_count + 1)
        ]
    else:
        raise ValueError("mode must be 'candidates' or 'references'")
    for row in rows:
        row["torch_seed"] = int(row["seed"]) % (2 ** 32)
    if len({int(row["torch_seed"]) for row in rows}) != len(rows):
        raise ValueError("panel seed policy produced a collision")
    return rows


def verified_sd21_snapshot(model: Dict[str, Any]) -> Path:
    """Resolve the pinned SD2.1 snapshot and hash its UNet pickle *before* anything unpickles it."""
    import huggingface_hub

    snapshot = Path(huggingface_hub.snapshot_download(
        repo_id=model["repo"], revision=model["revision"],
        allow_patterns=["unet/diffusion_pytorch_model.bin"],
    ))
    unet = snapshot / "unet" / "diffusion_pytorch_model.bin"
    if not unet.is_file() or sha256_file(unet) != model["unet_sha256"]:
        raise ValueError("SD2.1 UNet does not match the pinned official weight hash")
    return snapshot


def load_sd21_pipeline(device: str = "cuda") -> Any:
    """Load the pinned SD2.1 pipeline (fp16, DPM-Solver++ scheduler, no safety checker)."""
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    model = load_assets()["models"]["sd21"]
    snapshot = verified_sd21_snapshot(model)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(snapshot, subfolder="scheduler", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        snapshot, torch_dtype=torch.float16, use_safetensors=False, scheduler=scheduler,
        local_files_only=True,
    ).to(device)
    pipe.safety_checker = None
    return pipe


def generate_images(
    pipe: Any, prompt: str, torch_seeds: Sequence[int], *, device: str = "cuda",
    batch_size: int = GENERATION_BATCH_SIZE,
) -> list:
    """Generate individually seeded 512x512 images in small inference batches."""
    import torch

    if not torch_seeds or batch_size <= 0:
        raise ValueError("seeds and batch_size must be positive")
    images: list = []
    for start in range(0, len(torch_seeds), batch_size):
        group = list(torch_seeds[start:start + batch_size])
        generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in group]
        result = pipe(
            [prompt] * len(group), generator=generators, height=IMAGE_LENGTH, width=IMAGE_LENGTH,
            num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
        ).images
        if len(result) != len(group):
            raise ValueError("SD2.1 returned an unexpected image batch length")
        images.extend(image.convert("RGB") for image in result)
    return images


def _save_png(path: Path, image: Any) -> "dict[str, str]":
    from PIL import Image

    image.convert("RGB").save(path, format="PNG")
    with Image.open(path) as loaded:
        return {"file_sha256": sha256_file(path), "pixel_sha256": pixel_sha256(loaded)}


def generate_prompt_panel(
    pipe: Any, *, prompt_uid: str, prompt: str, master_key: bytes, output_dir: Path, mode: str,
    batch_size: int = 1, candidate_count: int = 16, candidate_start: int = 1, include_clean: bool = True,
    device: str = "cuda", generation_batch_size: int = GENERATION_BATCH_SIZE,
    native_key_id: str = NATIVE_KEY_ID,
) -> "dict[str, Any]":
    """Generate and write one prompt's panel; returns the ``panel.json`` record.

    ``native_key_id`` labels the clean controls' seeds; the paper's prompt list gives one
    per prompt, so its clean images can be regenerated exactly.
    """
    destination = Path(output_dir) / prompt_uid
    if destination.exists():
        raise ValueError("refusing to overwrite %s" % destination)
    rows = panel_seed_rows(
        prompt_uid=prompt_uid, master_key=master_key, mode=mode, batch_size=batch_size,
        candidate_count=candidate_count, candidate_start=candidate_start, include_clean=include_clean,
        native_key_id=native_key_id,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s.staging-" % prompt_uid, dir=str(output_dir)))
    model = load_assets()["models"]["sd21"]
    try:
        record: "dict[str, Any]" = {"prompt_uid": prompt_uid, "mode": mode, "prompt": prompt}
        if mode == "candidates":
            slots: "list[dict[str, Any]]" = []
            for slot in range(1, batch_size + 1):
                slot_dir = staging / ("slot_%03d" % slot)
                slot_dir.mkdir()
                slot_rows = [item for item in rows if int(item["batch_slot"]) == slot]
                images = generate_images(
                    pipe, prompt, [int(item["torch_seed"]) for item in slot_rows],
                    device=device, batch_size=generation_batch_size,
                )
                candidate_images: "list[dict[str, Any]]" = []
                clean: Optional["dict[str, Any]"] = None
                for row, image in zip(slot_rows, images):
                    if row["role"] == "clean":
                        clean = {"relative_path": "slot_%03d/clean.png" % slot,
                                 **_save_png(slot_dir / "clean.png", image)}
                    else:
                        name = "candidate_%02d.png" % int(row["position"])
                        candidate_images.append({
                            "candidate_position": int(row["position"]),
                            "relative_path": "slot_%03d/%s" % (slot, name),
                            **_save_png(slot_dir / name, image),
                        })
                slot_record: "dict[str, Any]" = {"slot": slot, "candidate_images": candidate_images}
                if include_clean:
                    slot_record["clean"] = clean
                slots.append(slot_record)
            record["batch_slots"] = slots
            record["candidate_start"] = candidate_start
        else:
            references = []
            images = generate_images(
                pipe, prompt, [int(item["torch_seed"]) for item in rows],
                device=device, batch_size=generation_batch_size,
            )
            for row, image in zip(rows, images):
                name = "reference_%02d.png" % int(row["position"])
                references.append({"reference_position": int(row["position"]), "relative_path": name,
                                   **_save_png(staging / name, image)})
            record["reference_images"] = references
        record.update({
            "candidate_count": candidate_count, "model_id": model["repo"],
            "model_revision": model["revision"], "image_length": IMAGE_LENGTH,
            "num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": GUIDANCE_SCALE,
        })
        (staging / "panel.json").write_text(canonical_json(record) + "\n", encoding="utf-8")
        os.replace(str(staging), str(destination))
        return record
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
