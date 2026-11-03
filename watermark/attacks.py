"""The 20 ordinary image edits of the robustness evaluation.

Each edit is deterministic: random edits (Gaussian and salt-and-pepper noise) draw from a
generator seeded by the source image's pixels and the edit identifier, so replaying an edit
on the same image reproduces the same pixels. Geometric edits keep the source geometry
(crops are resized back to the source size).
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class EditSpec:
    edit_id: str
    family: str
    operation_order: Tuple[str, ...]
    parameters: Mapping[str, Any]


def _jpeg_options(quality: int) -> Dict[str, Any]:
    return {"quality": quality, "optimize": False, "progressive": False, "subsampling": 2}


def _spec(edit_id: str, family: str, order: Tuple[str, ...], **parameters: Any) -> EditSpec:
    return EditSpec(edit_id, family, order, parameters)


# Geometric: centre and off-centre crops (resized back) and black-padded translations.
GEOMETRIC = (
    _spec("crop_50", "crop", ("center_crop", "bicubic_restore"), crop_fraction=0.50),
    _spec("crop_75", "crop", ("center_crop", "bicubic_restore"), crop_fraction=0.75),
    _spec("offcenter_crop_75", "offcenter_crop", ("offcenter_crop", "bicubic_restore"),
          crop_fraction=0.75, anchor_relative=[0.65, 0.35]),
    _spec("offcenter_crop_90", "offcenter_crop", ("offcenter_crop", "bicubic_restore"),
          crop_fraction=0.90, anchor_relative=[0.65, 0.35]),
    _spec("translate_32", "translation", ("translate_black_padding",), offset=[32, 32], fillcolor=[0, 0, 0]),
    _spec("translate_8", "translation", ("translate_black_padding",), offset=[8, 8], fillcolor=[0, 0, 0]),
)

# Appearance and compression: JPEG, blur, noise, photometric changes, median filter.
APPEARANCE_COMPRESSION = (
    _spec("jpeg_q90", "jpeg", ("jpeg",), jpeg=_jpeg_options(90)),
    _spec("jpeg_q75", "jpeg", ("jpeg",), jpeg=_jpeg_options(75)),
    _spec("gaussian_blur_0p5", "gaussian_blur", ("gaussian_blur",), radius=0.5),
    _spec("gaussian_blur_1p5", "gaussian_blur", ("gaussian_blur",), radius=1.5),
    _spec("gaussian_noise_2", "gaussian_noise", ("gaussian_noise",), stddev_255=2.0),
    _spec("gaussian_noise_6", "gaussian_noise", ("gaussian_noise",), stddev_255=6.0),
    _spec("brightness_0p8", "brightness", ("brightness",), brightness_factor=0.8),
    _spec("brightness_1p2", "brightness", ("brightness",), brightness_factor=1.2),
    _spec("contrast_0p8", "contrast", ("contrast",), contrast_factor=0.8),
    _spec("contrast_1p2", "contrast", ("contrast",), contrast_factor=1.2),
    _spec("median_3", "median", ("median_filter",), size=3),
    _spec("median_5", "median", ("median_filter",), size=5),
    _spec("salt_pepper_0p01", "salt_pepper", ("salt_pepper_noise",), fraction=0.01),
    _spec("salt_pepper_0p03", "salt_pepper", ("salt_pepper_noise",), fraction=0.03),
)

EDITS: Dict[str, EditSpec] = {spec.edit_id: spec for spec in GEOMETRIC + APPEARANCE_COMPRESSION}
EDIT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "geometric": tuple(spec.edit_id for spec in GEOMETRIC),
    "appearance_compression": tuple(spec.edit_id for spec in APPEARANCE_COMPRESSION),
}
EDIT_IDS: Tuple[str, ...] = tuple(EDITS)

if len(EDIT_IDS) != 20:  # pragma: no cover - guards the registry against accidental edits
    raise AssertionError("the robustness evaluation uses exactly 20 ordinary edits")


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256(("RGB:%dx%d\0" % (rgb.width, rgb.height)).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _seed(source_hash: str, edit_id: str) -> int:
    return int.from_bytes(hashlib.sha256(("%s\0%s" % (source_hash, edit_id)).encode()).digest()[:8], "big")


def _crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    retained = (math.floor(width * fraction + .5), math.floor(height * fraction + .5))
    left, top = (width - retained[0]) // 2, (height - retained[1]) // 2
    return image.crop((left, top, left + retained[0], top + retained[1]))


def _offcenter_crop(image: Image.Image, fraction: float, anchor_relative: Sequence[float]) -> Image.Image:
    anchor_x, anchor_y = (float(value) for value in anchor_relative)
    width, height = image.size
    retained = (math.floor(width * fraction + .5), math.floor(height * fraction + .5))
    left = min(max(math.floor(width * anchor_x - retained[0] / 2 + .5), 0), width - retained[0])
    top = min(max(math.floor(height * anchor_y - retained[1] / 2 + .5), 0), height - retained[1])
    return image.crop((left, top, left + retained[0], top + retained[1]))


def _jpeg(image: Image.Image, options: Mapping[str, Any]) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", **dict(options))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def render_edit(image: Image.Image, edit_id: str) -> Image.Image:
    """Apply one of the 20 edits to an image and return the edited RGB image."""
    if edit_id not in EDITS:
        raise ValueError("unknown edit: %s" % edit_id)
    parameters = dict(EDITS[edit_id].parameters)
    source = image.convert("RGB")
    if not source.width or not source.height:
        raise ValueError("source image has empty geometry")
    current, source_hash = source, pixel_sha256(source)
    if "crop_fraction" in parameters:
        if "anchor_relative" in parameters:
            current = _offcenter_crop(current, float(parameters["crop_fraction"]), parameters["anchor_relative"])
        else:
            current = _crop(current, float(parameters["crop_fraction"]))
        current = current.resize(source.size, Image.Resampling.BICUBIC)
    if "offset" in parameters:
        translation = tuple(int(value) for value in parameters["offset"])
        current = Image.new("RGB", source.size, tuple(int(value) for value in parameters["fillcolor"]))
        current.paste(source, translation)
    if "radius" in parameters:
        current = current.filter(ImageFilter.GaussianBlur(float(parameters["radius"])))
    if "size" in parameters:
        current = current.filter(ImageFilter.MedianFilter(int(parameters["size"])))
    if "brightness_factor" in parameters:
        current = ImageEnhance.Brightness(current).enhance(float(parameters["brightness_factor"]))
    if "contrast_factor" in parameters:
        current = ImageEnhance.Contrast(current).enhance(float(parameters["contrast_factor"]))
    if "stddev_255" in parameters:
        rng = np.random.default_rng(_seed(source_hash, edit_id))
        noise = rng.normal(0, float(parameters["stddev_255"]), (source.height, source.width, 1))
        current = Image.fromarray(
            np.clip(np.asarray(current, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
        )
    if "fraction" in parameters:
        rng = np.random.default_rng(_seed(source_hash, edit_id))
        array = np.asarray(current).copy()
        mask = rng.random(array.shape[:2]) < float(parameters["fraction"])
        array[mask] = np.where(rng.integers(0, 2, size=(mask.sum(), 1), dtype=np.uint8) == 0, 0, 255)
        current = Image.fromarray(array)
    if "jpeg" in parameters:
        current = _jpeg(current, parameters["jpeg"])
    if current.size != source.size:
        raise ValueError("edit changed source geometry")
    return current.convert("RGB")
