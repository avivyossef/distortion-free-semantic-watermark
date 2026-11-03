# CSD style versus depth layout encoders (Appendix "Black-Box Semantic Encoder Access")

Replaces the semantic encoder by CSD (style) and by a Depth Anything V2 Small layout encoder
and measures detection, retention and embedding change under three style edits (InstructPix2Pix:
oil painting, watercolor, pencil sketch) and three spatial transformations (32-pixel translation,
75% off-center crop, horizontal flip).

Produces: Table "Encoder-dependent watermark robustness" (`score_edits.py`) and Figure
"Transformations used to test encoder-dependent robustness" (`edit_figure.py`).

Settings come from the `style_depth` entry of `configs/detection.yaml`, read by
`watermark.config.load_detection_config()`: the first 100 prompts of the SD2.1 panel (sorted
identities), 16 slot-1 candidates and the slot-1 clean image per prompt, hyperplane frames and
keys of the shared replication grid, selection and boundary thresholds from unedited embeddings
only, and a score threshold set on the same unedited clean images that are evaluated.

## Prerequisites

- Environment: the repository's `pyproject.toml` dependencies (`open_clip_torch`, `diffusers`,
  `transformers`); a GPU. CSD and Depth Anything V2 Small are downloaded at the revisions pinned in
  `configs/assets.yaml`; InstructPix2Pix at the revision in `style_depth_lib.py`.
- The SD2.1 candidate panel of `experiments/generation` (slot 1, 16 candidates) for a prompt list
  `prompts.jsonl` of at least 100 prompts, and a 32-byte detector master key:

```bash
python experiments/generation/new_key.py --output keys/seed.key
python experiments/generation/generate_corpus.py --prompts prompts.jsonl --seed-key-file keys/seed.key \
    --output-dir panels/sd21 --candidates 16 --slots 1
python experiments/generation/new_key.py --output keys/detector.key
```

## 1. Render the style edits (17 images per prompt and edit)

Shard over GPUs with `--shard-index I --num-shards K`:

```bash
python experiments/style_depth_encoders/render_edits.py --panel-root panels/sd21 --output-dir edits
```

## 2. Embed originals and edits

```bash
python experiments/style_depth_encoders/embed_edits.py --panel-root panels/sd21 \
    --edit-dir edits --output embeddings.npz
```

## 3. Score

```bash
python experiments/style_depth_encoders/score_edits.py --embeddings embeddings.npz \
    --key-file keys/detector.key --output-dir results/style_depth
```

`table.csv` has one row per encoder and transformation: detection rate, retention, mean cosine
distance `1 - <z, z'>` over all candidates and clean controls, and the clean false-positive rate;
`replicates.csv` the per-replicate values. Retention averages the replicates in which
at least one image was detected.

## 4. Edit illustration

```bash
python experiments/style_depth_encoders/edit_figure.py --panel-root panels/sd21 --edit-dir edits \
    --prompt-uid <prompt_uid> --candidate-position 1 --output figures/encoder_dependent_edits.pdf
```

## Notes

- A fresh master key is not key-matched to any earlier run.
- `edit_figure.py` is a reconstruction from the figure caption; the image shown in the paper
  was chosen by hand.
