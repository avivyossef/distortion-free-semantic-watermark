"""Positive-pair augmentations: the frozen training and validation attack registry.

Positive pairs join an image with a content-preserving transformation of itself. Six
training families (Gaussian noise, Gaussian blur, JPEG compression, down/up-sampling,
photometric changes, small affine maps) and two validation families (noise followed by a
median filter, perspective warp) each have three severities. Every transformation is
deterministic: its random draws come from a generator seeded by the source identity,
family and severity.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ATTACK_REGISTRY_VERSION = "attack_registry_v1_2026_07_23"


@dataclass(frozen=True)
class AttackFamily:
    key: str
    partition: str  # "train" or "validation"
    severities: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class AttackChoice:
    family: str
    severity: str
    parameters: dict
    deterministic_seed: int


def _severity_map(**values: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType({severity: MappingProxyType(dict(params)) for severity, params in values.items()})


ATTACK_REGISTRY = MappingProxyType({
    "gaussian_noise": AttackFamily("gaussian_noise", "train", _severity_map(
        mild={"sigma": 2.0}, medium={"sigma": 6.0}, severe={"sigma": 12.0})),
    "gaussian_blur": AttackFamily("gaussian_blur", "train", _severity_map(
        mild={"radius": 0.5}, medium={"radius": 1.5}, severe={"radius": 3.0})),
    "jpeg_compression": AttackFamily("jpeg_compression", "train", _severity_map(
        mild={"quality": 85}, medium={"quality": 60}, severe={"quality": 35})),
    "downsample_upsample": AttackFamily("downsample_upsample", "train", _severity_map(
        mild={"scale": 0.85, "resample": "lanczos"}, medium={"scale": 0.60, "resample": "lanczos"},
        severe={"scale": 0.35, "resample": "lanczos"})),
    "photometric": AttackFamily("photometric", "train", _severity_map(
        mild={"brightness": 1.04, "contrast": 0.96, "color": 1.03},
        medium={"brightness": 0.90, "contrast": 1.12, "color": 0.82},
        severe={"brightness": 1.20, "contrast": 0.78, "color": 1.28})),
    "affine": AttackFamily("affine", "train", _severity_map(
        mild={"degrees": 1.5, "translate_fraction": 0.01, "scale": 0.98},
        medium={"degrees": 4.0, "translate_fraction": 0.03, "scale": 0.94},
        severe={"degrees": 8.0, "translate_fraction": 0.06, "scale": 0.88})),
    "noise_restoration": AttackFamily("noise_restoration", "validation", _severity_map(
        mild={"sigma": 3.0, "median_size": 3}, medium={"sigma": 8.0, "median_size": 3},
        severe={"sigma": 14.0, "median_size": 5})),
    "perspective_warp": AttackFamily("perspective_warp", "validation", _severity_map(
        mild={"displacement_fraction": 0.015}, medium={"displacement_fraction": 0.04},
        severe={"displacement_fraction": 0.08})),
})


def attack_seed(source_uid: str, family: str, severity: str) -> int:
    """Deterministic seed of one (source, family, severity) cell."""
    digest = hashlib.sha256(
        (source_uid + "\0" + family + "\0" + severity + "\0" + ATTACK_REGISTRY_VERSION).encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big")


def attack_choice(source_uid: str, family: str, severity: str) -> AttackChoice:
    """The registry cell ``(family, severity)`` bound to one source image."""
    return AttackChoice(family, severity, dict(ATTACK_REGISTRY[family].severities[severity]),
                        attack_seed(source_uid, family, severity))


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=False, progressive=False, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    array += rng.normal(0.0, float(sigma), size=array.shape)
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def _resize_round_trip(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    reduced = (max(1, int(round(width * float(scale)))), max(1, int(round(height * float(scale)))))
    return image.resize(reduced, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)


def _photometric(image: Image.Image, parameters: Mapping[str, Any]) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(float(parameters["brightness"]))
    result = ImageEnhance.Contrast(result).enhance(float(parameters["contrast"]))
    return ImageEnhance.Color(result).enhance(float(parameters["color"])).convert("RGB")


def _affine(image: Image.Image, parameters: Mapping[str, Any], rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    angle = math.radians(sign * float(parameters["degrees"]))
    scale = float(parameters["scale"])
    translate_fraction = float(parameters["translate_fraction"])
    tx = sign * translate_fraction * width
    ty = -sign * translate_fraction * height
    cosine = math.cos(angle) / scale
    sine = math.sin(angle) / scale
    center_x, center_y = width / 2.0, height / 2.0
    coefficients = (
        cosine, sine, center_x - cosine * center_x - sine * center_y - tx,
        -sine, cosine, center_y + sine * center_x - cosine * center_y - ty,
    )
    return image.transform(
        image.size, Image.Transform.AFFINE, coefficients, resample=Image.Resampling.BICUBIC,
        fillcolor=(127, 127, 127),
    ).convert("RGB")


def _perspective(image: Image.Image, fraction: float, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    dx, dy = fraction * width, fraction * height
    offsets = rng.uniform(-1.0, 1.0, size=(4, 2))
    corners = np.asarray([[0.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0], [width - 1.0, 0.0]])
    corners += offsets * np.asarray([dx, dy])
    return image.transform(
        image.size, Image.Transform.QUAD, tuple(float(value) for value in corners.reshape(-1)),
        resample=Image.Resampling.BICUBIC, fillcolor=(127, 127, 127),
    ).convert("RGB")


def render_attack(image: Image.Image, choice: AttackChoice) -> Image.Image:
    """Apply one registry transformation to an image."""
    if choice.family not in ATTACK_REGISTRY:
        raise ValueError("unknown attack family %r" % choice.family)
    spec = ATTACK_REGISTRY[choice.family]
    if choice.severity not in spec.severities:
        raise ValueError("unknown severity %r" % choice.severity)
    if choice.parameters != dict(spec.severities[choice.severity]):
        raise ValueError("attack parameters do not match the frozen registry")
    source = image.convert("RGB")
    rng = np.random.default_rng(choice.deterministic_seed)
    parameters = choice.parameters
    if choice.family == "gaussian_noise":
        result = _noise(source, float(parameters["sigma"]), rng)
    elif choice.family == "gaussian_blur":
        result = source.filter(ImageFilter.GaussianBlur(float(parameters["radius"])))
    elif choice.family == "jpeg_compression":
        result = _jpeg(source, int(parameters["quality"]))
    elif choice.family == "downsample_upsample":
        result = _resize_round_trip(source, float(parameters["scale"]))
    elif choice.family == "photometric":
        result = _photometric(source, parameters)
    elif choice.family == "affine":
        result = _affine(source, parameters, rng)
    elif choice.family == "noise_restoration":
        result = _noise(source, float(parameters["sigma"]), rng).filter(
            ImageFilter.MedianFilter(int(parameters["median_size"])))
    elif choice.family == "perspective_warp":
        result = _perspective(source, float(parameters["displacement_fraction"]), rng)
    else:  # pragma: no cover - registry and dispatch are tested in lockstep
        raise AssertionError("unhandled attack family %s" % choice.family)
    return result.convert("RGB")
