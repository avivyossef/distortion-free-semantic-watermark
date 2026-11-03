# Semantic encoder training (paper Appendix: Semantic Encoder Training)

Trains the semantic encoder: frozen DINOv2-base, rank-8 LoRA on the query and value
projections of the last four blocks, a 768-512-256 MLP head (GELU, dropout 0.1) and L2
normalization. Positive pairs join an image with a transformation of itself (noise, blur,
JPEG, resampling, photometric, affine) or with its fifth-step preview; negative pairs join
generations of the same prompt from different seeds. Each optimizer update holds 16 pairs of
one type; AdamW, learning rate 1e-4, mixed precision, gradient clipping at 1.0 and a learning
rate decay of 0.95 per epoch. One epoch is four updates per training prompt.

Produces: a checkpoint `checkpoint-step-<updates>.pt` (the paper's encoder is the checkpoint
after 19,840 updates; checkpoint SHA-256 and download: `scripts/download_checkpoint.py`).
Needs a GPU, `diffusers`, `transformers` and `peft`.

## Inputs

A prompt list, JSONL, one row per prompt (100 prompts: 80 train, 20 validation; half from
DiffusionDB's 2M text-only split, half from COCO-2017 training captions):

```json
{"prompt_uid": "p0001", "text": "a photo of ...", "split": "train"}
```

## 1. Generate the SDXL sources (final image and fifth-step preview per seed)

Shard over GPUs with `--shard-index I --num-shards K`; every shard writes its own manifest.

```bash
python experiments/encoder_training/generate_pairs.py \
    --prompts prompts.jsonl --output-dir data/sources \
    --registry-seed <seed> --seeds-per-prompt 64 \
    --shard-index 0 --num-shards 1
```

Seeds are allocated once from `--registry-seed`; keep it with the prompt list.

## 2. Train

```bash
python experiments/encoder_training/train.py \
    --data-dir data/sources --output-dir runs/encoder \
    --target-updates 19840
```

Outputs in `--output-dir`: `checkpoint-step-<updates>.pt` per epoch (with validation metrics in
its `history`) and `run.json`. The target must be a whole number of epochs
(`4 x training prompts` updates each).

The model is always the paper's: DINOv2-base with rank-8 LoRA on the query and value projections
of the last four blocks and an MLP head. Variations:

- Seed curriculum: train with `--seeds-per-prompt 8`, then re-run with `--resume <checkpoint>
  --seeds-per-prompt 16`, and so on up to 64.
- `--negative-cap` sets the negative-pair loss `max(0, cos - cap)`; the default 0 is the loss
  stated in the paper.

## 3. Use the checkpoint

```python
from watermark.embedding import load_encoder
encoder = load_encoder("ours", device="cuda", checkpoint="checkpoints/checkpoint-step-00019840.pt")
vectors = encoder.embed(pil_images)        # (n, 256), unit norm
```
