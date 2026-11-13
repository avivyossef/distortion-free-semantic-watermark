#!/usr/bin/env python3
"""Train the semantic encoder (DINOv2-base + rank-8 LoRA + MLP head) on SDXL typed pairs.

``--data-dir`` is the output of ``generate_pairs.py``. Training prompts use their first
``--seeds-per-prompt`` allocated seeds (re-run with ``--resume`` and a larger value for the
8 -> 16 -> 32 -> 64 seed curriculum); validation prompts use their first ``--validation-seeds``.
One epoch is four optimizer updates per training prompt, two of each pair type, with 16 pairs per update.
A checkpoint ``checkpoint-step-<updates>.pt`` is written after every ``--save-every-epochs``
epochs together with validation metrics.

    python experiments/encoder_training/train.py --data-dir data/sources \
        --output-dir runs/encoder --target-updates 19840
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark.io import read_jsonl  # noqa: E402


def load_manifest(data_dir: Path, seeds_per_prompt: int, validation_seeds: int):
    """Merge shard manifests; keep each prompt's first K rows in allocation order."""
    by_prompt = OrderedDict()
    for shard in sorted((data_dir / "manifests").glob("shard-*.jsonl")):
        for row in read_jsonl(shard):
            by_prompt.setdefault(row["prompt_uid"], []).append(row)
    if not by_prompt:
        raise SystemExit("no manifest shards under %s" % (data_dir / "manifests"))
    rows = []
    for prompt_uid in sorted(by_prompt):
        prompt_rows = by_prompt[prompt_uid]
        limit = seeds_per_prompt if prompt_rows[0]["prompt_split"] == "train" else validation_seeds
        if len(prompt_rows) < limit:
            raise SystemExit("prompt %s has %d sources, fewer than %d" % (prompt_uid, len(prompt_rows), limit))
        rows.extend(prompt_rows[:limit])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument("--seeds-per-prompt", type=int, default=64)
    parser.add_argument("--validation-seeds", type=int, default=64)
    parser.add_argument("--negative-cap", type=float, default=0.0,
                        help="negative-pair loss is max(0, cos - cap); the paper uses 0")
    parser.add_argument("--training-seed", type=int, default=20260814)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor

    from encoder.data import SourceCorpus
    from encoder.model import DinoLoRAMetric
    from encoder.training import CandidateConfiguration, manifest_sha256, run_training
    from watermark.config import load_assets

    revision = load_assets()["models"]["dinov2_base"]["revision"]
    rows = load_manifest(args.data_dir, args.seeds_per_prompt, args.validation_seeds)
    corpus = SourceCorpus.from_rows(rows, data_root=args.data_dir)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", revision=revision)
    model = DinoLoRAMetric.from_pretrained(revision=revision, variant="lora_mlp", lora_rank=8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_training(
        corpus=corpus, processor=processor, model=model, device=torch.device(args.device),
        output_dir=args.output_dir, candidate=CandidateConfiguration("lora_mlp", 8),
        target_updates=args.target_updates, training_seed=args.training_seed, negative_cap=args.negative_cap,
        resume=args.resume, save_every_epochs=args.save_every_epochs, data_manifest_sha256=manifest_sha256(rows),
    )
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
