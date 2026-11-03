"""GPU adapters around the official Tree-Ring, Gaussian Shading and SEAL code.

Each adapter drives the pinned upstream checkout of ``third_party/clone.sh`` on the pinned
SD2.1 weights (512 x 512, 50 steps, guidance 7.5). ``load_generator`` returns an object whose
``generate(prompt, seed)`` yields ``(watermarked, clean)`` images; ``load_scorer`` returns a
function mapping an image path to the native detection score, oriented so that a higher score
means more likely watermarked (Tree-Ring: negative L1 distance in Fourier space; Gaussian
Shading: recovered-bit accuracy; SEAL: number of matching patches).

Everything heavy (torch, diffusers, transformers, the upstream modules) is imported inside the
functions; run this module only through ``generate.py`` and ``score.py`` in the environment of
the method (see the README).
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Tuple

import numpy as np

import _baselines_lib as lib
from watermark.config import load_assets
from watermark.generation import verified_sd21_snapshot
from watermark.io import sha256_file

SEAL_BLIP_REVISION = "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"
SEAL_SENTENCE_REVISION = "1f155d6a17f7a176d7d269f7ec04bcfa376ce297"
SEAL_PATCHES, SEAL_BITS, SEAL_NOISE_SEED, SEAL_TAU = 1024, 7, 42, 2.3
GAUSSIAN_CONFIG = {"channel_copy": 1, "hw_copy": 8, "fpr": 1e-6, "user_number": 1_000_000}
INVERSION_STEPS = 50
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
IMAGE_LENGTH = 512


def sd21_model() -> dict:
    return load_assets()["models"]["sd21"]


def seed_everything(seed: int, torch: Any) -> None:
    bounded = int(seed) % (2 ** 32)
    random.seed(bounded)
    np.random.seed(bounded)
    torch.manual_seed(bounded)
    torch.cuda.manual_seed_all(bounded)


def official_module(repo: Path, filename: str, alias: str) -> Any:
    """Import one upstream file under ``alias``.

    Gaussian Shading's ``watermark.py`` would otherwise collide with this repository's
    ``watermark`` package.
    """
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(alias, str(repo / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pipe(repo: Path, utility_module: str = "optim_utils", *, scheduler: str = "dpm") -> Tuple[Any, Any, Any]:
    """The official invertible SD2.1 pipeline: ``scheduler`` is ``dpm``, ``ddim`` or ``default``."""
    sys.path.insert(0, str(repo))
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    inverse = importlib.import_module("inverse_stable_diffusion")
    model = sd21_model()
    snapshot = verified_sd21_snapshot(model)
    kwargs = {"torch_dtype": torch.float16, "use_safetensors": False, "local_files_only": True}
    if scheduler == "dpm":
        kwargs["scheduler"] = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            snapshot, subfolder="scheduler", local_files_only=True)
    elif scheduler == "ddim":
        kwargs["scheduler"] = diffusers.DDIMScheduler.from_pretrained(
            snapshot, subfolder="scheduler", local_files_only=True)
    elif scheduler != "default":
        raise ValueError("unknown scheduler")
    pipe = inverse.InversableStableDiffusionPipeline.from_pretrained(snapshot, **kwargs).to("cuda")
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    return torch, pipe, importlib.import_module(utility_module)


def invert(pipe: Any, utils: Any, path: str) -> Any:
    """Latents of an image under the official DDIM-style forward diffusion with an empty prompt."""
    from PIL import Image

    with Image.open(path) as image:
        tensor = utils.transform_img(image.convert("RGB")).unsqueeze(0).to("cuda").to(pipe.vae.dtype)
    latents = pipe.get_image_latents(tensor, sample=False)
    return pipe.forward_diffusion(
        latents=latents, text_embeddings=pipe.get_text_embedding(""),
        guidance_scale=1, num_inference_steps=INVERSION_STEPS)


def _pipe_call(pipe: Any, prompt: str, **kwargs: Any) -> Any:
    return pipe(prompt, guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS, **kwargs).images[0]


# --------------------------------------------------------------------------- Tree-Ring


def _tree_fft_fp32(latents: Any, torch: Any) -> Any:
    """Tree-Ring's small FFT on CPU: the legacy torch 1.13 CUDA build fails cuFFT on current GPUs."""
    cpu_latents = latents.detach().to(device="cpu", dtype=torch.float32)
    return torch.fft.fftshift(torch.fft.fft2(cpu_latents), dim=(-1, -2))


def _tree_pattern_fp32(pipe: Any, utils: Any, config: Any, torch: Any) -> Any:
    """The official ``rand`` pattern with a stable FFT dtype."""
    utils.set_random_seed(config.w_seed)
    patch = _tree_fft_fp32(pipe.get_random_latents(), torch)
    patch[:] = patch[0]
    return patch


def _tree_inject_fp32(latents: Any, mask: Any, patch: Any, torch: Any) -> Any:
    """Official complex-frequency injection, returned in the original dtype."""
    original_dtype, original_device = latents.dtype, latents.device
    spectrum = _tree_fft_fp32(latents, torch)
    cpu_mask, cpu_patch = mask.detach().to("cpu"), patch.detach().to("cpu")
    spectrum[cpu_mask] = cpu_patch[cpu_mask].clone()
    injected = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-1, -2))).real
    return injected.to(device=original_device, dtype=original_dtype)


def _tree_config(master_key: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        w_seed=lib.tree_ring_seed(master_key), w_channel=0, w_pattern="rand", w_mask_shape="circle",
        w_radius=10, w_measurement="l1_complex", w_injection="complex", w_pattern_const=0.0)


class TreeRingGenerator:
    def __init__(self, repo: Path, master_key: bytes) -> None:
        self.torch, self.pipe, self.utils = load_pipe(repo)
        self.config = _tree_config(master_key)
        self.patch = _tree_pattern_fp32(self.pipe, self.utils, self.config, self.torch)

    def generate(self, prompt: str, seed: int) -> Tuple[Any, Any]:
        seed_everything(seed, self.torch)
        initial = self.pipe.get_random_latents()
        clean = _pipe_call(self.pipe, prompt, height=IMAGE_LENGTH, width=IMAGE_LENGTH, latents=copy.deepcopy(initial))
        mask = self.utils.get_watermarking_mask(initial, self.config, "cuda")
        marked = _tree_inject_fp32(copy.deepcopy(initial), mask, self.patch, self.torch)
        watermarked = _pipe_call(self.pipe, prompt, height=IMAGE_LENGTH, width=IMAGE_LENGTH, latents=marked)
        return watermarked, clean


def _tree_scorer(repo: Path, master_key: bytes) -> Callable[[str], float]:
    torch, pipe, utils = load_pipe(repo)
    config = _tree_config(master_key)
    patch = _tree_pattern_fp32(pipe, utils, config, torch).cpu()

    def score(path: str) -> float:
        reversed_latents = invert(pipe, utils, path)
        mask = utils.get_watermarking_mask(reversed_latents, config, "cuda")
        fft = _tree_fft_fp32(reversed_latents, torch)
        cpu_mask = mask.detach().cpu()
        return -float(torch.abs(fft[cpu_mask] - patch[cpu_mask]).mean().item())

    return score


# --------------------------------------------------------------------------- Gaussian Shading


def _gaussian_official(repo: Path, master_key: bytes) -> Tuple[Any, Any]:
    """The official ChaCha sampler with its random key, nonce and message replaced by the fixed state."""
    import torch

    module = official_module(repo, "watermark.py", "gaussian_shading_official")
    official = module.Gaussian_Shading_chacha(
        GAUSSIAN_CONFIG["channel_copy"], GAUSSIAN_CONFIG["hw_copy"], GAUSSIAN_CONFIG["fpr"],
        GAUSSIAN_CONFIG["user_number"])
    official.key, official.nonce = lib.gaussian_key_nonce(master_key)
    bits = lib.gaussian_watermark_bits(master_key, 4 // official.ch, 64 // official.hw)
    official.watermark = torch.from_numpy(bits)
    return official, torch


class GaussianShadingGenerator:
    def __init__(self, repo: Path, master_key: bytes) -> None:
        from Crypto.Cipher import ChaCha20

        self.torch, self.pipe, _ = load_pipe(repo)
        self.official, _ = _gaussian_official(repo, master_key)
        official = self.official
        plaintext = np.packbits(
            official.watermark.repeat(1, official.ch, official.hw, official.hw).flatten().cpu().numpy()).tobytes()
        ciphertext = ChaCha20.new(key=official.key, nonce=official.nonce).encrypt(plaintext)
        self.message_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))

    def generate(self, prompt: str, seed: int) -> Tuple[Any, Any]:
        seed_everything(seed, self.torch)
        marked = self.official.truncSampling(self.message_bits.copy())
        watermarked = _pipe_call(self.pipe, prompt, height=IMAGE_LENGTH, width=IMAGE_LENGTH, latents=marked)
        seed_everything(seed, self.torch)
        clean = _pipe_call(self.pipe, prompt, height=IMAGE_LENGTH, width=IMAGE_LENGTH,
                           latents=self.pipe.get_random_latents())
        return watermarked, clean


def _gaussian_scorer(repo: Path, master_key: bytes) -> Callable[[str], float]:
    # The official inverse update reads DDIM alpha products, so the detector inverts with the DDIM
    # scheduler (generation keeps the DPM-Solver scheduler of the official run script).
    torch, pipe, utils = load_pipe(repo, "image_utils", scheduler="ddim")
    detector, _ = _gaussian_official(repo, master_key)
    detector.watermark = detector.watermark.to("cuda")

    def score(path: str) -> float:
        return float(detector.eval_watermark(invert(pipe, utils, path)))

    return score


# --------------------------------------------------------------------------- SEAL


def _seal_models(torch: Any) -> Tuple[Any, Any, Any]:
    transformers = importlib.import_module("transformers")
    sentence_transformers = importlib.import_module("sentence_transformers")
    processor = transformers.Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", revision=SEAL_BLIP_REVISION)
    captioner = transformers.Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", revision=SEAL_BLIP_REVISION, torch_dtype=torch.float16).to("cuda")
    encoder = sentence_transformers.SentenceTransformer(
        "kasraarabi/finetuned-caption-embedding", revision=SEAL_SENTENCE_REVISION).to("cuda")
    return processor, captioner, encoder


def _seal_noise(utils: Any, torch: Any, encoder: Any, caption: str) -> Any:
    embedding = encoder.encode(caption, convert_to_tensor=True).to("cuda")
    embedding = embedding / torch.norm(embedding)
    return utils.generate_initial_noise(embedding, SEAL_PATCHES, SEAL_BITS, SEAL_NOISE_SEED, "cuda")


class SealGenerator:
    def __init__(self, repo: Path, master_key: bytes) -> None:
        # SEAL has no secret key beyond its fixed configuration; the official default scheduler is used.
        self.torch, self.pipe, self.utils = load_pipe(repo, "utils", scheduler="default")
        self.processor, self.captioner, self.encoder = _seal_models(self.torch)

    def generate(self, prompt: str, seed: int) -> Tuple[Any, Any]:
        seed_everything(seed, self.torch)
        clean = _pipe_call(self.pipe, prompt)
        caption = self.utils.generate_caption(clean, self.processor, self.captioner)
        marked = _seal_noise(self.utils, self.torch, self.encoder, caption).to(dtype=self.pipe.vae.dtype)
        watermarked = _pipe_call(self.pipe, prompt, latents=marked)
        return watermarked, clean


def _seal_scorer(repo: Path, master_key: bytes) -> Callable[[str], float]:
    from PIL import Image

    torch, pipe, utils = load_pipe(repo, "utils", scheduler="default")
    processor, captioner, encoder = _seal_models(torch)

    def score(path: str) -> float:
        with Image.open(path) as image:
            caption = utils.generate_caption(image.convert("RGB"), processor, captioner)
        expected = _seal_noise(utils, torch, encoder, caption).to(pipe.vae.dtype)
        patch_l2 = utils.calculate_patch_l2(invert(pipe, utils, path), expected, SEAL_PATCHES)
        return seal_match_score(patch_l2)

    return score


def seal_match_score(patch_l2: Any, *, tau: float = SEAL_TAU) -> float:
    """SEAL's scalar: the number of patches whose distance is below ``tau``."""
    return float(sum(distance < tau for distance in patch_l2))


# --------------------------------------------------------------------------- entry points

_GENERATORS = {"tree_ring": TreeRingGenerator, "gaussian_shading": GaussianShadingGenerator, "seal": SealGenerator}
_SCORERS = {"tree_ring": _tree_scorer, "gaussian_shading": _gaussian_scorer, "seal": _seal_scorer}


def load_generator(method: str, official_root: Path, master_key: bytes) -> Any:
    repo = (Path(official_root) / lib.REPOSITORY_DIRS[method]).resolve()
    lib.verify_checkout(repo, method)
    return _GENERATORS[method](repo, master_key)


def load_scorer(method: str, official_root: Path, master_key: bytes) -> Callable[[str], float]:
    repo = (Path(official_root) / lib.REPOSITORY_DIRS[method]).resolve()
    lib.verify_checkout(repo, method)
    return _SCORERS[method](repo, master_key)
