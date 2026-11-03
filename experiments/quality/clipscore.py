#!/usr/bin/env python3
"""Image-prompt alignment (appendix quality table): OpenCLIP ViT-g/14 cosine similarity.

The score of an image is the cosine similarity between its OpenCLIP image feature and the
feature of its prompt; the score of a population is the mean over its prompts. One image per
prompt is scored (slot 1). Our watermarked populations are those of ``select_populations.py``
(every hyperplane frame and key), compared with the mean of the matched-random controls; a
native baseline contributes its watermarked and its own clean images. The report gives the mean
score of each side and the relative change ``100 (wm / clean - 1)``.

    python experiments/quality/clipscore.py --selections selections.npz --corpus-dir corpus \
        --prompts prompts.jsonl --native-dirs tree_ring=out/tree_ring gaussian_shading=out/gaussian_shading \
        seal=out/seal --output-dir results/clipscore

Needs a GPU and ``open_clip_torch``; the ``laion2b_s34b_b88k`` weights are fetched from the Hub at the
revision pinned in ``configs/assets.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402

from watermark.config import load_assets  # noqa: E402
from watermark.io import sha256_file  # noqa: E402

CLIP_ASSET = load_assets()["models"]["openclip_vit_g14"]
CLIP_MODEL = CLIP_ASSET["model"]


def verified_clip_weights() -> str:
    """Fetch the pinned OpenCLIP weights and check their SHA-256 before use."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CLIP_ASSET["repo"], CLIP_ASSET["file"], revision=CLIP_ASSET["revision"])
    if sha256_file(Path(path)) != CLIP_ASSET["sha256"]:
        raise ValueError("OpenCLIP weights SHA-256 does not match configs/assets.yaml")
    return path


def cosine_similarity_rows(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two matching matrices."""
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("image and text features must be matching two-dimensional arrays")
    image_norms = np.linalg.norm(image_features, axis=1)
    text_norms = np.linalg.norm(text_features, axis=1)
    if np.any(image_norms == 0) or np.any(text_norms == 0):
        raise ValueError("cosine similarity is undefined for zero-norm features")
    return np.sum(image_features * text_features, axis=1) / (image_norms * text_norms)


def candidate_population_score(features: np.ndarray, text: np.ndarray, positions: np.ndarray) -> float:
    """Mean alignment of the images selected by ``positions`` (1-based, one per prompt).

    ``features`` is ``(P, N, D)`` (every candidate of every prompt), ``text`` is ``(P, D)``.
    """
    rows = np.arange(len(positions))
    return float(np.mean(cosine_similarity_rows(features[rows, positions - 1], text)))


def relative_change(watermarked: float, clean: float) -> float:
    return 100.0 * (watermarked / clean - 1.0)


def summarize_ours(features: np.ndarray, text: np.ndarray, selections: dict, n: int, m: int) -> dict:
    """Mean alignment of the watermarked populations of (N, M) against the matched controls."""
    marked = selections[lib.selection_key(n, m, "watermarked")][..., 0]  # (frames, keys, P)
    controls = selections[lib.selection_key(n, m, "random")][..., 0]  # (keys, P)
    watermarked = [candidate_population_score(features, text, positions)
                   for frame in marked for positions in frame]
    clean = [candidate_population_score(features, text, positions) for positions in controls]
    return {"method": "ours", "N": n, "M": m, "watermarked_mean": float(np.mean(watermarked)),
            "watermarked_populations": len(watermarked), "clean_mean": float(np.mean(clean)),
            "clean_populations": len(clean),
            "relative_change_percent": relative_change(float(np.mean(watermarked)), float(np.mean(clean)))}


def image_features(paths: Sequence[Path], model, preprocess, *, device: str, dtype, batch_size: int) -> np.ndarray:
    import torch
    from PIL import Image

    chunks: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        with torch.inference_mode():
            batch = torch.stack(tensors).to(device=device, dtype=dtype)
            chunks.append(torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True, help="candidate corpus of our method")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--native-dirs", nargs="*", default=[], metavar="NAME=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import open_clip
    import torch

    uids, selections, metadata = lib.load_selections(args.selections)
    texts = {str(row["prompt_uid"]): str(row["text"]) for row in lib.read_prompts(args.prompts)}
    precision = "fp16" if args.device.startswith("cuda") else "fp32"
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, device=args.device, precision=precision,
        pretrained=verified_clip_weights())
    model.eval()
    dtype = open_clip.get_input_dtype(precision) or torch.float32
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    with torch.inference_mode():
        tokens = tokenizer([texts[uid] for uid in uids]).to(args.device)
        text = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1).cpu().numpy()
    rows = []
    paths = [path for uid in uids
             for path in lib.panel_images(args.corpus_dir, lib.read_panel(args.corpus_dir, uid), 1)[0][0]]
    features = image_features(paths, model, preprocess, device=args.device, dtype=dtype,
                              batch_size=args.batch_size).reshape(len(uids), -1, text.shape[1])
    for n, m in metadata["configurations"]:
        rows.append(summarize_ours(features, text, selections, n, m))
    for name, directory in sorted(lib.parse_named_paths(args.native_dirs).items()):
        pairs = [lib.native_images(directory, uid, 1) for uid in uids]
        marked = image_features([item[0][0] for item in pairs], model, preprocess, device=args.device,
                                dtype=dtype, batch_size=args.batch_size)
        clean = image_features([item[1][0] for item in pairs], model, preprocess, device=args.device,
                               dtype=dtype, batch_size=args.batch_size)
        wm_score = float(np.mean(cosine_similarity_rows(marked, text)))
        clean_score = float(np.mean(cosine_similarity_rows(clean, text)))
        rows.append({"method": name, "watermarked_mean": wm_score, "watermarked_populations": 1,
                     "clean_mean": clean_score, "clean_populations": 1,
                     "relative_change_percent": relative_change(wm_score, clean_score)})
    lib.write_csv(args.output_dir / "summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
