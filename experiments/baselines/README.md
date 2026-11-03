# Native baselines: Tree-Ring, Gaussian Shading, SEAL

Runs the three native image watermarks on Stable Diffusion 2.1 Base (512 x 512, 50 steps,
guidance 7.5, the pinned weights of `configs/assets.yaml`) with each method's own generation and
detection procedure. It produces

- the generated images behind the baseline rows of the generation quality and diversity table
  (FID against the matched clean images, same-prompt diversity) and of the appendix quality and
  prompt-alignment table; the `quality` experiment reads the layout below;
- the baseline rows of the clean-detection table and the operating-points table (single image);
- the baseline cells of the ordinary-edit robustness evaluation (edited TPR, edited FPR, paired
  retention, category means over the 20 settings of `watermark.attacks`).

The generative-edit baseline cells use `score.py` on edited images (see `generative_edits`).

## Prerequisites

- GPU for `generate.py` and `score.py`; the aggregation scripts are CPU only.
- The upstream code, cloned at the pinned commits (their licenses apply; they are not vendored):

  ```bash
  bash third_party/clone.sh third_party/checkouts      # --official-root is third_party/checkouts
  ```

- Hugging Face weights of SD2.1 Base (pinned in `configs/assets.yaml`; the UNet hash is verified)
  and, for SEAL, `Salesforce/blip2-flan-t5-xl` and `kasraarabi/finetuned-caption-embedding` at the
  revisions set in `_baselines_native.py`.
- A prompt file, JSONL with one row `{"prompt_uid": ..., "text": ...}` per prompt: the same
  evaluation prompts as the corresponding experiment of our method (1,000 prompts for clean
  detection and quality, the 200 robustness prompts, 60 for diversity, 5,000 COCO validation
  captions for FID against real images; see the `quality` README).
- A 32-byte master key file (below). It fixes the single method state of a run.

## Environments

| Command | Environment |
|---|---|
| `generate.py`, `score.py` with `--method tree_ring` or `gaussian_shading` | `environment-tree-gaussian.yml` (python 3.10, torch 1.13, diffusers 0.11.1, transformers 4.25.1) |
| `generate.py`, `score.py` with `--method seal` | `environment-seal.yml` (python 3.10, torch 2.5.1, diffusers 0.20.2, transformers 4.47.1, sentence-transformers 3.3.1) |
| `list_images.py`, `attack_images.py`, `clean_detection.py`, `robustness.py` | any environment with `numpy` and `Pillow` (plain `python3`) |

The scripts add the repository root to the import path themselves; run every command from
the repository root. Both environments also need whatever else the pinned upstream commit imports; the
files list the packages the experiment was run with.

## Fixed key

One fixed method state per run, derived from the master key file with the HMAC derivation of
`watermark.prf`: the Tree-Ring pattern seed; the Gaussian Shading ChaCha20 key, nonce and message.
SEAL has no secret beyond its fixed configuration (1024 patches, 7 bits, noise seed 42) and derives
its watermark from the caption of the clean image of the same slot. Use the same key file for
generation and scoring of a method.

```bash
python experiments/generation/new_key.py --output keys/baselines.key
```

## 1. Generate

Shard over prompts with `--shard-index I --num-shards K`; finished prompts are skipped on rerun.
`--num-slots S` makes `S` independent generations per prompt (10 for the 10,000-image FID, 60 for
the diversity panel, 1 otherwise). The watermarked and the clean image of a slot share their seed
(slot 1: the native per-identity seed; slots 2 and above: the slot seed of the extension).

```bash
python experiments/baselines/generate.py --method tree_ring --prompts prompts.jsonl --num-slots 10 \
    --official-root third_party/checkouts --master-key-file keys/baselines.key --output-dir out/tree_ring
```

Repeat with `--method gaussian_shading` and `--method seal` (SEAL in its own environment).

Output layout (this is the contract read by the `quality` experiment):

```
<output-dir>/<prompt_uid>/slot_001/watermarked.png
<output-dir>/<prompt_uid>/slot_001/clean.png
<output-dir>/<prompt_uid>/slot_002/...
<output-dir>/<prompt_uid>/panel.json
```

`panel.json` is written last, after all images of the prompt:

```json
{"prompt_uid": "...", "method": "tree_ring", "prompt": "...", "official_commit": "...",
 "model_id": "...", "model_revision": "...", "image_length": 512, "num_inference_steps": 50,
 "guidance_scale": 7.5,
 "slots": [{"slot": 1,
            "watermarked": {"relative_path": "slot_001/watermarked.png", "file_sha256": "...", "pixel_sha256": "..."},
            "clean": {"relative_path": "slot_001/clean.png", "file_sha256": "...", "pixel_sha256": "..."}}]}
```

Relative paths are relative to the prompt directory. The clean image is each method's own clean
control (Tree-Ring and Gaussian Shading: the same pipeline and initial latents without the
watermark; SEAL: the official default scheduler).

## 2. Clean detection (baseline rows of the detection and operating-point tables)

```bash
python3 experiments/baselines/list_images.py --generation-dir out/tree_ring --prompts prompts.jsonl \
    --output out/tree_ring_images.jsonl
python experiments/baselines/score.py --method tree_ring --official-root third_party/checkouts \
    --master-key-file keys/baselines.key --images out/tree_ring_images.jsonl --output out/tree_ring_scores.jsonl
python3 experiments/baselines/clean_detection.py --scores out/tree_ring_scores.jsonl \
    --output-dir analysis/tree_ring_clean
```

`score.py` runs in the environment of the method and may be sharded the same way; give each
shard its own `--output` and pass all files to `--scores`. A higher score always means more
likely watermarked. `clean_detection.py` writes `summary.csv` (AUROC), `operating_points.csv`
(TPR at the empirical FPR targets of `configs/detection.yaml`, read from the ROC curve) and
`roc.csv`.

## 3. Ordinary-edit robustness (baseline cells)

The edits are rendered on both the watermarked and the clean images; the score thresholds are
frozen on the unedited clean images only, one per FPR target, and are not recalibrated.

```bash
python3 experiments/baselines/attack_images.py --generation-dir out/tree_ring --prompts prompts.jsonl \
    --output-dir out/tree_ring_edits --images-output out/tree_ring_edited_images.jsonl
python experiments/baselines/score.py --method tree_ring --official-root third_party/checkouts \
    --master-key-file keys/baselines.key --images out/tree_ring_edited_images.jsonl --output out/tree_ring_edited_scores.jsonl
python3 experiments/baselines/robustness.py \
    --scores out/tree_ring_scores.jsonl out/tree_ring_edited_scores.jsonl --output-dir analysis/tree_ring_robustness
```

`robustness.py` writes `per_edit.csv` (edited TPR, edited FPR and paired retention per setting)
and `categories.csv` (unweighted means over the settings of each category). Use the 200 robustness
prompts (the same file) for both `list_images.py` and `attack_images.py` here; the same
edits and thresholds protocol is used for our method in the `robustness` experiment.

Detection details kept from the upstream code: Tree-Ring inverts the image with its own
forward-diffusion and scores the negative L1 distance to the pattern in Fourier space; Gaussian
Shading inverts with the DDIM scheduler (its inversion reads DDIM quantities) and scores the
recovered-bit accuracy; SEAL captions the image, regenerates the expected noise from the caption
and counts the patches within distance 2.3 of the inverted latents.
