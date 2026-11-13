# Semantic encoder training

These scripts implement the pair construction and objective in the paper's
*Semantic Encoder Training* appendix.

| Pair | Images | Loss |
|---|---|---|
| Positive | An SDXL image and a transformed copy of that image | `1 - cos` |
| Negative | Two final SDXL images from the same prompt and different seeds | `max(0, cos)` |

Here `cos` is the cosine similarity between the two normalized embeddings.
The positive loss pulls an image and its transformed copy together. The
negative loss penalizes positive similarity between independent generations.
Training transformations include Gaussian noise, Gaussian blur, JPEG compression,
resampling, photometric changes, and affine transformations.

The encoder uses frozen DINOv2-base, rank-8 LoRA on the query and value projections
of the last four blocks, and a 768–512–256 MLP head with GELU, dropout 0.1, and
L2 normalization. Each update uses 16 pairs of one type, with an equal number of
positive and negative updates. Training uses AdamW at a learning rate of
`1e-4`, CUDA mixed precision, gradient clipping at 1.0, and a learning rate
multiplier of 0.95 per epoch. It needs a GPU, `diffusers`, `transformers`, and `peft`.

## Inputs

A prompt list, JSONL, one row per prompt (100 prompts: 80 train, 20 validation; half from
DiffusionDB's 2M text-only split, half from COCO-2017 training captions):

```json
{"prompt_uid": "p0001", "text": "a photo of ...", "split": "train"}
```

## 1. Generate one final SDXL image per prompt and seed

Shard over GPUs with `--shard-index I --num-shards K`; every shard writes its own manifest.

```bash
python experiments/encoder_training/generate_pairs.py \
    --prompts prompts.jsonl --output-dir data/sources \
    --registry-seed 1 --seeds-per-prompt 64 \
    --shard-index 0 --num-shards 1
```

The registry seed above is an example. Keep your seed with the prompt list. This command writes final images
and a source manifest. The trainer constructs pairs and applies transformations
as it loads the images. Existing manifests with a `preview_path` field still work.

## 2. Train

```bash
python experiments/encoder_training/train.py \
    --data-dir data/sources --output-dir runs/encoder \
    --target-updates 19840
```

The script writes a checkpoint per epoch and records validation metrics in its
`history`. It also writes `run.json`. An epoch contains four updates per
training prompt, two of each pair type. Training and validation each use
64 seeds per prompt by default.

## 3. Load the checkpoint used in the reported experiments

The reported experiments use the pinned `checkpoint-step-00019840.pt`.
Its original training run also used fifth-step preview pairs and a negative
loss of `max(0, cos - 0.25)`. The command above trains fresh weights using
the appendix recipe. To evaluate the reported checkpoint, download the pinned
file with `scripts/download_checkpoint.py` and load it as follows.

```python
from watermark.embedding import load_encoder
encoder = load_encoder("ours", device="cuda", checkpoint="checkpoints/checkpoint-step-00019840.pt")
vectors = encoder.embed(pil_images)        # (n, 256), unit norm
```
