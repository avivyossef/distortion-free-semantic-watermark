"""Image encoders used by the paper, behind one ``embed(images) -> (n, d)`` interface.

Every encoder returns float64 rows with unit L2 norm. Heavy dependencies (torch,
transformers, open_clip, tpips) are imported inside the constructors, so importing this
module is cheap. Pretrained models are loaded from the Hugging Face Hub at the revisions
pinned in ``configs/assets.yaml``.

``load_encoder(name, ...)`` accepts: ``ours`` (the trained semantic encoder, needs
``checkpoint``), ``dinov2_base``, ``clip_vit_b32``, ``dinov3_l16``, ``tpips_embed``, and the
two black-box encoders of the style/depth study, ``csd_style`` and ``depth_layout``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

from .config import load_assets

ENCODER_NAMES = ("ours", "dinov2_base", "clip_vit_b32", "dinov3_l16", "tpips_embed", "csd_style", "depth_layout")
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
DEPTH_GRID = 16


def normalize(values: Any) -> np.ndarray:
    """Convert model output to a finite float64 matrix with unit-norm rows."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "float"):
        values = values.float()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    matrix = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("encoder returned invalid embeddings")
    return matrix / norms


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[list]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def _select_embedding_output(value: Any) -> Any:
    """Extract a two-dimensional global embedding, including nested HF wrappers."""
    if getattr(value, "ndim", None) == 2:
        return value
    if getattr(value, "ndim", None) == 3:
        return value[:, 0, :]
    for attribute in ("pooler_output", "last_hidden_state"):
        candidate = getattr(value, attribute, None)
        if candidate is not None and candidate is not value:
            return _select_embedding_output(candidate)
    raise ValueError("vision model returned no two-dimensional global embedding")


def load_images(paths: Sequence[Path]) -> list:
    """Open image files as RGB PIL images."""
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as opened:
            images.append(opened.convert("RGB").copy())
    return images


def embed_paths(encoder: Any, paths: Sequence[Path], *, batch_size: int = 16) -> np.ndarray:
    """Embed image files in order; returns ``(len(paths), d)`` unit-norm float64 rows."""
    chunks = [encoder.embed(load_images(batch)) for batch in _batched(list(paths), batch_size)]
    return np.concatenate(chunks, axis=0)


class HuggingFaceEncoder:
    """DINOv2-base, DINOv3-L/16 (global representation) or CLIP ViT-B/32 (image features)."""

    def __init__(self, *, repo: str, revision: str, device: str, clip: bool = False) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.device, self.clip = device, clip
        if clip:
            self.processor = CLIPProcessor.from_pretrained(repo, revision=revision)
            self.model = CLIPModel.from_pretrained(repo, revision=revision, torch_dtype=torch.bfloat16)
        else:
            self.processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
            self.model = AutoModel.from_pretrained(repo, revision=revision, torch_dtype=torch.bfloat16)
        self.model = self.model.to(device).eval()
        self.provenance = {"model_id": repo, "revision": revision, "type": "clip" if clip else "image_encoder"}

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            inputs = self.processor(images=list(images), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            values = self.model.get_image_features(**inputs) if self.clip else self.model(**inputs)
        return normalize(_select_embedding_output(values))


class OursEncoder:
    """The trained semantic encoder (DINOv2-base + LoRA + MLP head), 256-d."""

    def __init__(self, checkpoint: Path, device: str, *, expected_sha256: str, batch_size: int = 16) -> None:
        from encoder.checkpoint import load_encoder as load_trained

        self.model, self.processor, self.provenance = load_trained(
            checkpoint, device=device, expected_sha256=expected_sha256
        )
        self.batch_size = batch_size

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        chunks = []
        for batch in _batched(list(images), self.batch_size):
            pixels = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.inference_mode():
                chunks.append(normalize(self.model(pixel_values=pixels)))
        return np.concatenate(chunks, axis=0)


class TPIPSEncoder:
    """TPIPS-Embed (Qwen3-VL-8B), overall factor, embedded in micro-batches."""

    def __init__(self, device: str, *, batch_size: int = 4, model_path: Optional[Path] = None) -> None:
        if batch_size <= 0:
            raise ValueError("TPIPS batch size must be positive")
        import tpips

        kwargs = {"model_path": str(model_path)} if model_path is not None else {}
        self.model = tpips.load_model("embedding", device=device, **kwargs)
        self.batch_size = batch_size
        self.provenance = {"model": load_assets()["models"]["tpips_embed"]["repo"], "factor": "overall",
                           "batch_size": batch_size}

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        method = self.model.embed
        kwargs = {"factor": "overall"} if "factor" in inspect.signature(method).parameters else {}
        parts: List[np.ndarray] = []
        for batch in _batched(list(images), self.batch_size):
            try:
                values = method(batch, **kwargs)
            except (TypeError, ValueError):
                values = [method(image, **kwargs) for image in batch]
            if isinstance(values, (list, tuple)):
                parts.append(np.concatenate([normalize(value) for value in values], axis=0))
            else:
                parts.append(normalize(values))
        return np.concatenate(parts, axis=0)


def depth_descriptor(depth: Any, grid: int = DEPTH_GRID) -> np.ndarray:
    """Area-pool a ``(B, H, W)`` relative depth map to ``grid x grid``, then zero-mean per image.

    Relative depth has an arbitrary per-image scale and shift, so only the spatial pattern
    is kept (the unit norm is applied by :func:`normalize`).
    """
    import torch

    pooled = torch.nn.functional.adaptive_avg_pool2d(depth[:, None].float(), grid).flatten(1).double()
    pooled = pooled - pooled.mean(dim=1, keepdim=True)
    return pooled.cpu().numpy()


def load_csd_visual(weights: Path) -> "tuple[Any, Any]":
    """CSD ViT-L/14 backbone (open_clip, OpenAI layout) and its style head, from a checkpoint file."""
    import open_clip
    import torch

    # The checkpoint carries training metadata (numpy scalars, an argparse namespace);
    # allowlist exactly those and load tensors only.
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(str(weights))
    if any(not name.startswith("numpy.") and name != "argparse.Namespace" for name in unsafe):
        raise ValueError("CSD checkpoint references unexpected globals: %s" % (unsafe,))
    allowed = [(getattr(importlib.import_module(name.rpartition(".")[0]), name.rpartition(".")[2]), name)
               for name in unsafe]
    allowed += [getattr(np.dtypes, name) for name in np.dtypes.__all__]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    visual = open_clip.create_model("ViT-L-14-quickgelu", pretrained=None).visual
    backbone = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
    result = visual.load_state_dict(backbone, strict=False)
    if set(result.missing_keys) != {"proj"} or result.unexpected_keys:
        raise ValueError("CSD backbone does not match open_clip ViT-L-14: %s" % (result,))
    visual.proj = None
    return visual.float().eval(), state["last_layer_style"].float()


class CSDEncoder:
    """Style encoder: CSD ViT-L style head, CLIP preprocessing at 224 px."""

    def __init__(self, device: str, *, weights: Optional[Path] = None) -> None:
        from torchvision import transforms

        if weights is None:
            from huggingface_hub import hf_hub_download

            spec = load_assets()["models"]["csd_vit_l"]
            weights = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"]))
        self.device = device
        visual, style = load_csd_visual(Path(weights))
        self.visual, self.style = visual.to(device), style.to(device)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        self.provenance = {"weights": Path(weights).name, "open_clip": importlib.metadata.version("open_clip_torch"),
                           "preprocess": "resize 224 bicubic, center crop 224, CLIP normalize"}

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        import torch

        with torch.no_grad():
            batch = torch.stack([self.transform(image) for image in images]).to(self.device)
            return normalize((self.visual(batch) @ self.style).double().cpu().numpy())


class DepthEncoder:
    """Layout encoder: Depth Anything V2 Small relative depth, pooled to 16x16 and standardized."""

    def __init__(self, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        spec = load_assets()["models"]["depth_anything_v2_small"]
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
        self.model = AutoModelForDepthEstimation.from_pretrained(spec["repo"], revision=spec["revision"])
        self.model = self.model.eval().to(device)
        self.provenance = {"model": spec["repo"], "revision": spec["revision"],
                           "descriptor": "predicted_depth area-pooled %dx%d, zero-mean, unit-norm" % (DEPTH_GRID, DEPTH_GRID)}

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        import torch

        with torch.no_grad():
            inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
            return normalize(depth_descriptor(self.model(**inputs).predicted_depth))


def load_encoder(name: str, *, device: str = "cuda", checkpoint: Optional[Path] = None,
                 checkpoint_sha256: Optional[str] = None, tpips_batch_size: int = 4) -> Any:
    """Construct one of :data:`ENCODER_NAMES` at its pinned revision.

    The trained encoder's checkpoint is verified before it is unpickled, against
    ``checkpoint_sha256`` or, by default, the released checkpoint's hash.
    """
    if name == "ours":
        if checkpoint is None:
            raise ValueError("the trained encoder needs --checkpoint")
        expected = checkpoint_sha256 or load_assets()["checkpoint"]["sha256"]
        return OursEncoder(Path(checkpoint), device, expected_sha256=expected)
    models = load_assets()["models"]
    if name in ("dinov2_base", "dinov3_l16"):
        spec = models[name]
        return HuggingFaceEncoder(repo=spec["repo"], revision=spec["revision"], device=device)
    if name == "clip_vit_b32":
        spec = models[name]
        return HuggingFaceEncoder(repo=spec["repo"], revision=spec["revision"], device=device, clip=True)
    if name == "tpips_embed":
        return TPIPSEncoder(device, batch_size=tpips_batch_size)
    if name == "csd_style":
        return CSDEncoder(device)
    if name == "depth_layout":
        return DepthEncoder(device)
    raise ValueError("unknown encoder: %s (choose from %s)" % (name, ", ".join(ENCODER_NAMES)))
