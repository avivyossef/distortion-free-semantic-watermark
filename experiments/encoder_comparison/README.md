# Encoder comparison (Section "Semantic Representation Analysis" and Appendix "Encoder Evaluation")

Compares the trained encoder with DINOv2-base, CLIP ViT-B/32, DINOv3 L/16 and TPIPS-Embed.

Produces:

| Paper item | Script |
|---|---|
| Table "Same-prompt diversity in encoder space" (mean pairwise cosine similarity and semantic entropy at the table radius, SD2.1 / FLUX distilled / FLUX base) | `embed_generations.py`, `semantic_entropy.py` |
| Figure "Semantic entropy as a function of epsilon" | `semantic_entropy.py`, `plot_semantic_entropy_epsilon.py` |
| Table "Encoder stability under content-preserving transformations" | `stability.py` |
| Figure "Distinct codes under semantics-preserving perturbations" | `stability.py encode`, `distinct_codes.py perturbations` |
| Figure "Distinct codes under independent generations" | `embed_generations.py`, `distinct_codes.py generations` |
| Figure "Different images from one prompt" (appendix; 16 SDXL images, DINOv2 vs ours cosine matrices) | `entropy_example.py` |

The two distinct-code figures are a reconstruction from the paper's description (see Notes).

## Prerequisites

- Environment: the repository's `pyproject.toml` dependencies; a GPU for generation and
  embedding. TPIPS-Embed needs the `tpips` package.
- The trained encoder checkpoint (`scripts/download_checkpoint.py`).
- A prompt list `prompts.jsonl`, one row per held-out prompt: `{"prompt_uid": "...", "text": "..."}`.
  The first 200 identities in sorted order are used by every script.
- ImageNette and ImageWoof, 160-pixel versions, unpacked (each with `train/` and `val/` class folders).

## 1. Same-prompt panels (200 prompts x 16 images per generator)

SD2.1: the candidate corpus of `experiments/generation`, slot 1, positions 1 to 16 (one slot and
16 candidates are enough; positions generated earlier keep their seeds when more are added):

```bash
python experiments/generation/new_key.py --output keys/seed.key
python experiments/generation/generate_corpus.py --prompts prompts.jsonl --seed-key-file keys/seed.key \
    --output-dir panels/sd21 --candidates 16 --slots 1
```

FLUX.2 [klein], the distilled and the base variant, 16 images with 16 different seeds per
prompt (shard with `--shard-index I --num-shards K`):

```bash
python experiments/encoder_comparison/generate_flux.py --prompts prompts.jsonl \
    --variant distilled --output-dir panels/flux_distilled
python experiments/encoder_comparison/generate_flux.py --prompts prompts.jsonl \
    --variant base --output-dir panels/flux_base
```

## 2. Embed each panel with each encoder

Run for every panel (`sd21`, `flux_distilled`, `flux_base`) and every encoder
(`dinov2_base`, `clip_vit_b32`, `dinov3_l16`, `tpips_embed`, `ours`; `ours` needs `--checkpoint`):

```bash
python experiments/encoder_comparison/embed_generations.py --panel-root panels/flux_base \
    --encoder ours --checkpoint checkpoints/checkpoint-step-00019840.pt \
    --output embeddings/flux_base/ours.npz
```

## 3. Table and entropy curves

```bash
for panel in sd21 flux_distilled flux_base; do
  python experiments/encoder_comparison/semantic_entropy.py \
      --embeddings-dir embeddings/$panel --output-dir results/$panel
done
python experiments/encoder_comparison/plot_semantic_entropy_epsilon.py \
    --sd21 results/sd21/curves.csv --flux-distilled results/flux_distilled/curves.csv \
    --flux-base results/flux_base/curves.csv --output figures/semantic_entropy_epsilon.pdf
```

`results/<panel>/table.csv` holds one row per encoder: the mean pairwise cosine similarity over
the 120 within-prompt pairs of every prompt, and the semantic-entropy estimate at the table
radius (0.5 cosine distance), the negative logarithm of the pooled fraction of pairs within
that radius. `curves.csv` holds the estimate on a 401-point grid over [0, 2].

## 4. Stability under content-preserving transformations

200 ImageNette / ImageWoof images, 16 variants each (increasing severity) for JPEG compression,
Gaussian blur, resize-resample and mild crop with color adjustment. Run `encode` once per encoder,
then `table`:

```bash
python experiments/encoder_comparison/stability.py encode --encoder ours \
    --checkpoint checkpoints/checkpoint-step-00019840.pt \
    --imagenette-root data/imagenette2-160 --imagewoof-root data/imagewoof2-160 \
    --output-dir results/stability
python experiments/encoder_comparison/stability.py table --output-dir results/stability
```

## 5. Distinct hash codes

```bash
python experiments/encoder_comparison/distinct_codes.py perturbations \
    --embeddings-dir results/stability/embeddings --output-dir results/unique_codes_perturbations
python experiments/encoder_comparison/distinct_codes.py generations \
    --panel "SD2.1=embeddings/sd21" --panel "FLUX distilled=embeddings/flux_distilled" \
    --panel "FLUX base=embeddings/flux_base" --output-dir results/unique_codes_generations
```

## 6. Same-prompt example figure

Generates 16 SDXL images (seeds `--seed-base` to `--seed-base + 15`, default 0) of one held-out
prompt, embeds them with DINOv2-base and the trained encoder, and draws the image grid next to
the two pairwise-cosine matrices. GPU.

```bash
python experiments/encoder_comparison/entropy_example.py --prompt "<held-out prompt>" \
    --checkpoint checkpoints/checkpoint-step-00019840.pt --output-dir figures/entropy_example
```

The paper does not name the prompt or the seeds of its figure; choose any held-out prompt.

## Notes

- Hash-code figures: no code producing the published figures exists, so `distinct_codes.py` is
  written from the appendix text: 8-bit random-hyperplane codes, 200 hyperplane draws (seeds
  derived from `frame_seed_base` of `configs/detection.yaml`, disjoint from the detector's
  frames), the same draw for every group and encoder, curves over the number of group members
  included. The paper's caption names SDXL as the first generator panel of the generation figure;
  this script plots the panels given on the command line (here SD2.1 and the two FLUX variants,
  as in the table). An SDXL panel needs SDXL images in the same panel layout.
- The FLUX variants use the pinned revisions and sampler settings of `comparison_lib.FLUX_VARIANTS`
  (distilled: 4 steps, guidance 1; base: 50 steps, guidance 4; 1024 x 1024).
