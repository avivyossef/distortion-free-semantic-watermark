#!/usr/bin/env python3
"""Inception-v3 pool-3 activations (pytorch-fid) of every image of a corpus, one shard of the prompts.

``--source`` selects what is embedded:

- ``corpus``: the candidates and clean control of every slot of an SD2.1 candidate corpus
  (``experiments/generation``); arrays ``candidates`` (p, S, N, 2048) and ``cleans`` (p, S, 2048).
- ``native``: the watermarked and clean images of a native-baseline generation
  (``experiments/baselines/generate.py``); arrays ``watermarked`` and ``cleans`` (p, S, 2048).
- ``real``: the real COCO validation images named by the ``file_name`` of a prompt panel
  (``fid5k_panel.py``); array ``images`` (p, 2048).

Images are converted to RGB and bilinearly resized to 299 x 299 before the network, one
activation pass per image. Writes ``<output-dir>/activations-shard-III-of-KKK.npz``.

    python experiments/quality/inception_activations.py --source corpus --corpus-dir corpus \
        --prompts prompts.jsonl --output-dir activations/ours --shard-index 0 --num-shards 1

Needs ``pytorch-fid`` (the same Inception weights as the standard FID).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _quality_lib as lib  # noqa: E402


def image_transform():
    """Resize to 299 x 299 (bilinear) and convert to a tensor in [0, 1]."""
    from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

    return Compose([Resize((299, 299), interpolation=InterpolationMode.BILINEAR), ToTensor()])


def load_inception(device: str):
    from pytorch_fid.inception import InceptionV3

    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[lib.FID_DIMENSION]]).to(device).eval()


def activations(paths: Sequence[Path], model, *, batch_size: int, device: str, num_workers: int) -> np.ndarray:
    """Pool-3 activations ``(len(paths), 2048)`` of the images at ``paths``."""
    import torch
    from torch.nn.functional import adaptive_avg_pool2d
    from torch.utils.data import DataLoader
    from pytorch_fid.fid_score import ImagePathDataset

    dataset = ImagePathDataset([str(path) for path in paths], transforms=image_transform())
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, drop_last=False,
                        num_workers=num_workers)
    result = np.empty((len(dataset), lib.FID_DIMENSION), dtype=np.float32)
    start = 0
    for batch in loader:
        with torch.no_grad():
            prediction = model(batch.to(device))[0]
        if prediction.size(2) != 1 or prediction.size(3) != 1:
            prediction = adaptive_avg_pool2d(prediction, output_size=(1, 1))
        values = prediction.squeeze(3).squeeze(2).cpu().numpy()
        result[start:start + len(values)] = values
        start += len(values)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("corpus", "native", "real"), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, help="candidate corpus or native generation directory")
    parser.add_argument("--images-dir", type=Path, help="real COCO images (--source real)")
    parser.add_argument("--slots", type=int, help="use only the first slots (default: all)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = lib.deal(lib.read_prompts(args.prompts), args.shard_index, args.num_shards)
    uids = [str(row["prompt_uid"]) for row in rows]
    model = load_inception(args.device)
    paths: List[Path] = []
    arrays = {}
    if args.source == "real":
        if args.images_dir is None:
            raise SystemExit("--images-dir is required for --source real")
        paths = [args.images_dir / str(row["file_name"]) for row in rows]
        arrays["images"] = activations(paths, model, batch_size=args.batch_size, device=args.device,
                                       num_workers=args.num_workers)
    else:
        if args.corpus_dir is None:
            raise SystemExit("--corpus-dir is required for --source corpus and native")
        layout: List[tuple] = []  # (candidate paths per slot or None, watermarked, clean) per prompt
        for uid in uids:
            if args.source == "corpus":
                candidates, cleans = lib.panel_images(args.corpus_dir, lib.read_panel(args.corpus_dir, uid), args.slots)
                layout.append((candidates, None, cleans))
            else:
                marked, cleans = lib.native_images(args.corpus_dir, uid, args.slots)
                layout.append((None, marked, cleans))
        for candidates, marked, cleans in layout:
            if candidates is not None:
                paths.extend(path for slot in candidates for path in slot)
            else:
                paths.extend(marked)
            paths.extend(cleans)
        values = activations(paths, model, batch_size=args.batch_size, device=args.device,
                             num_workers=args.num_workers)
        cursor, candidate_rows, marked_rows, clean_rows = 0, [], [], []
        for candidates, marked, cleans in layout:
            if candidates is not None:
                count = sum(len(slot) for slot in candidates)
                candidate_rows.append(values[cursor:cursor + count].reshape(len(candidates), len(candidates[0]), -1))
                cursor += count
            else:
                marked_rows.append(values[cursor:cursor + len(marked)])
                cursor += len(marked)
            clean_rows.append(values[cursor:cursor + len(cleans)])
            cursor += len(cleans)
        if args.source == "corpus":
            arrays["candidates"] = np.stack(candidate_rows)
        else:
            arrays["watermarked"] = np.stack(marked_rows)
        arrays["cleans"] = np.stack(clean_rows)
    lib.write_shard(args.output_dir / ("activations-shard-%03d-of-%03d.npz" % (args.shard_index, args.num_shards)),
                    uids, **arrays)
    print("wrote %d prompts" % len(uids), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
