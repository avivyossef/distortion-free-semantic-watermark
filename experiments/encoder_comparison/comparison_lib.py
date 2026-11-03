"""Shared pieces of the encoder comparison: panels, statistics, perturbation controls.

* Same-prompt panels are prompt directories holding a ``panel.json`` (see
  :func:`panel_image_paths`); the first 200 prompt identities in sorted order are used.
* Semantic entropy (paper Definition 1): pair distances, pooled estimate.
* Stability controls: ImageNette / ImageWoof source images and the four perturbation families.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.config import load_assets  # noqa: E402

PROMPT_COUNT = 200
GROUP_SIZE = 16
ENCODERS = ("dinov2_base", "clip_vit_b32", "dinov3_l16", "tpips_embed", "ours")
ENCODER_LABELS = {
    "dinov2_base": "DINOv2-base", "clip_vit_b32": "CLIP ViT-B/32", "dinov3_l16": "DINOv3 L/16",
    "tpips_embed": "TPIPS-Embed", "ours": "Ours",
}
ENCODER_COLORS = {
    "ours": "#b63836", "dinov2_base": "#2376a5", "clip_vit_b32": "#9a6a16", "dinov3_l16": "#528248",
    "tpips_embed": "#8059a0",
}
TABLE_EPSILON = 0.5  # radius of the encoder-comparison table (cosine distance)
EPSILON_GRID = np.linspace(0, 2, 401)
SOURCE_SEED = 20260905  # shuffling of the ImageNette / ImageWoof class images

# FLUX.2 [klein] variants of the paper's encoder comparison. Each of a prompt's 16 images has
# its own seed: ``seed_base + stride * prompt_index + position`` with position 0..15.
_ASSETS = load_assets()["models"]
FLUX_VARIANTS: Dict[str, Dict[str, Any]] = {
    "distilled": {
        "repo": _ASSETS["flux2_klein_distilled"]["repo"], "revision": _ASSETS["flux2_klein_distilled"]["revision"],
        "steps": 4, "guidance": 1.0, "seed_base": 202608300000, "seed_stride": 16,
    },
    "base": {
        "repo": _ASSETS["flux2_klein_base"]["repo"], "revision": _ASSETS["flux2_klein_base"]["revision"],
        "steps": 50, "guidance": 4.0, "seed_base": 202610280000, "seed_stride": 17,
    },
}
FLUX_IMAGE_SIZE = 1024


def flux_seeds(variant: str, prompt_index: int) -> List[int]:
    """The 16 per-image seeds of prompt ``prompt_index`` (rank in sorted prompt order)."""
    spec = FLUX_VARIANTS[variant]
    if not 0 <= prompt_index < PROMPT_COUNT:
        raise ValueError("prompt index must lie in 0..%d" % (PROMPT_COUNT - 1))
    first = spec["seed_base"] + spec["seed_stride"] * prompt_index
    return list(range(first, first + GROUP_SIZE))


# --------------------------------------------------------------------------- panels


def select_prompts(rows: Sequence[Dict[str, Any]], count: int = PROMPT_COUNT) -> List[Dict[str, Any]]:
    """The first ``count`` prompt rows in sorted ``prompt_uid`` order."""
    uids = [str(row["prompt_uid"]) for row in rows]
    if len(set(uids)) != len(uids) or len(rows) < count:
        raise ValueError("prompt registry needs at least %d unique prompt identities" % count)
    return sorted(rows, key=lambda row: str(row["prompt_uid"]))[:count]


def panel_image_paths(panel_root: Path, prompt_uid: str) -> List[Path]:
    """The 16 same-prompt image files of one prompt, in seed order.

    ``<panel_root>/<prompt_uid>/panel.json`` is either the SD2.1 corpus record (slot 1,
    candidate positions 1..16) or the FLUX record written by ``generate_flux.py``
    (``images``, positions 1..16).
    """
    root = Path(panel_root) / prompt_uid
    record = json.loads((root / "panel.json").read_text(encoding="utf-8"))
    if "batch_slots" in record:
        slots = [slot for slot in record["batch_slots"] if slot.get("slot") == 1]
        if len(slots) != 1:
            raise ValueError("prompt %s does not contain exactly one slot 1" % prompt_uid)
        images = slots[0]["candidate_images"]
    else:
        images = record["images"]
    images = sorted(images, key=lambda value: value["candidate_position"])
    if [value["candidate_position"] for value in images] != list(range(1, GROUP_SIZE + 1)):
        raise ValueError("prompt %s does not contain positions 1..%d" % (prompt_uid, GROUP_SIZE))
    paths = [(root / value["relative_path"]).resolve() for value in images]
    if any(root.resolve() not in path.parents or not path.is_file() for path in paths):
        raise ValueError("prompt %s contains an invalid image path" % prompt_uid)
    return paths


def panel_prompt_uids(panel_root: Path) -> List[str]:
    """The 200 prompt identities of a panel directory (sorted, first 200)."""
    uids = sorted(path.parent.name for path in Path(panel_root).glob("*/panel.json"))
    if len(uids) < PROMPT_COUNT:
        raise ValueError("panel has %d prompts, expected at least %d" % (len(uids), PROMPT_COUNT))
    return uids[:PROMPT_COUNT]


# --------------------------------------------------------------------------- semantic entropy


def pair_distances(embeddings):
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("expected finite prompt x generation x feature embeddings")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding")
    x = x / norms
    gram = x @ x.transpose(0, 2, 1)
    i, j = np.triu_indices(x.shape[1], 1)
    return np.clip(1 - gram[:, i, j], 0, 2)


def estimate(distances, grid):
    p = (distances[..., None] <= grid).mean(axis=(0, 1))
    with np.errstate(divide="ignore"):
        return p, -np.log(p)


def mean_pairwise_cosine(embeddings) -> float:
    """Mean cosine similarity over all within-prompt pairs, pooled over prompts.

    Every prompt contributes the same number of pairs, so this equals the mean of the
    per-prompt (per-group) means.
    """
    return float(1.0 - pair_distances(embeddings).mean())


def block_mean_cosine(values: np.ndarray, *, block_size: int) -> float:
    """Mean pairwise cosine similarity of one block of ``block_size`` embeddings."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != block_size or matrix.shape[1] <= 0:
        raise ValueError("each block must have shape (%d, dimension)" % block_size)
    if not np.isfinite(matrix).all():
        raise ValueError("embedding block contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0):
        raise ValueError("embedding block contains zero-norm values")
    embeddings = matrix / norms[:, None]
    kernel = embeddings @ embeddings.T
    return float((kernel.sum() - block_size) / (block_size * (block_size - 1)))


# --------------------------------------------------------------------------- stability controls

FAMILIES = ("jpeg", "gaussian_blur", "resize_resample", "mild_crop_color")
FAMILY_LABELS = {
    "jpeg": "JPEG compression", "gaussian_blur": "Gaussian blur", "resize_resample": "Resize-resample",
    "mild_crop_color": "Mild crop + color",
}


@dataclass(frozen=True)
class ImageBlock:
    condition: str
    block_id: str
    paths: Tuple[Path, ...]


def _class_images(*roots: Path, seed: int) -> Dict[str, List[Path]]:
    generator = np.random.default_rng(seed)
    groups: Dict[str, List[Path]] = {}
    for root in roots:
        for split in ("train", "val"):
            split_root = root / split
            if not split_root.is_dir():
                continue
            for directory in sorted(path for path in split_root.iterdir() if path.is_dir()):
                key = f"{root.name}:{directory.name}"
                groups.setdefault(key, []).extend(
                    sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
                )
    for key, paths in groups.items():
        order = generator.permutation(len(paths))
        groups[key] = [paths[index] for index in order]
    return groups


def build_perturbation_blocks(
    imagenette_root: Path, imagewoof_root: Path, *, count: int = PROMPT_COUNT,
    block_size: int = GROUP_SIZE, seed: int = SOURCE_SEED,
) -> Dict[str, List[ImageBlock]]:
    """One source image per block for each perturbation family; the block holds it ``block_size`` times.

    Sources cycle through the ImageNette and ImageWoof classes. Each class's images are
    shuffled once; the first ``block_size`` images of every class visit are set aside
    (they form the same-class control group of the comparison, which is not part of the
    paper's tables) so the sources are the same images the paper's run used.
    """
    groups = _class_images(imagenette_root, imagewoof_root, seed=seed)
    labels = sorted(groups)
    if len(labels) < block_size:
        raise ValueError(f"at least {block_size} ImageNet classes are required")
    cursors = {label: 0 for label in labels}

    def take(label: str, amount: int) -> Tuple[Path, ...]:
        start, end = cursors[label], cursors[label] + amount
        if end > len(groups[label]):
            raise ValueError(f"class {label} does not contain enough unique images")
        cursors[label] = end
        return tuple(groups[label][start:end])

    result: Dict[str, List[ImageBlock]] = {condition: [] for condition in FAMILIES}
    for index in range(count):
        take(labels[index % len(labels)], block_size)
    for index in range(count):
        label = labels[index % len(labels)]
        source = take(label, 1)[0]
        for condition in FAMILIES:
            result[condition].append(ImageBlock(condition, f"{condition}-{index:04d}-{label}", (source,) * block_size))
    return result


def _augment(image: Image.Image, condition: str, index: int) -> Image.Image:
    image = image.convert("RGB")
    if condition == "jpeg":
        payload = io.BytesIO()
        image.save(payload, format="JPEG", quality=35 + 4 * index, optimize=True)
        payload.seek(0)
        with Image.open(payload) as opened:
            return opened.convert("RGB").copy()
    if condition == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=0.25 + 0.1 * index))
    if condition == "resize_resample":
        scale = 0.65 + 0.02 * index
        target = (max(16, round(image.width * scale)), max(16, round(image.height * scale)))
        return image.resize(target, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BICUBIC)
    if condition == "mild_crop_color":
        factor = 0.86 + 0.008 * index
        width, height = max(16, round(image.width * factor)), max(16, round(image.height * factor))
        offset_x = (image.width - width) * (index % 4) // 3
        offset_y = (image.height - height) * (index // 4) // 3
        cropped = image.crop((offset_x, offset_y, offset_x + width, offset_y + height)).resize(image.size, Image.Resampling.BICUBIC)
        return ImageEnhance.Color(cropped).enhance(0.90 + 0.012 * index)
    return image


def load_variants(block: ImageBlock) -> List[Image.Image]:
    """The 16 perturbed versions of a block's source image, severity increasing with the index."""
    images = []
    for index, path in enumerate(block.paths):
        with Image.open(path) as opened:
            images.append(_augment(opened, block.condition, index))
    return images
