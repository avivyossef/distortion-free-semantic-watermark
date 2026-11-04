# Generative edits (paper: robustness under common transformations, generative-edit panel)

Produces the **generative-edit** columns of the robustness comparison (appendix "Detection and
Robustness Protocols", generative-edit paragraph; the generative rows of the per-edit figure and
of the robustness summary table; Figure "Generative-edit evaluation protocol"): 80 watermarked
images and their 80 unwatermarked counterparts, 20 per category (attribute change, background
cleanup, cat insertion, camera-viewpoint change), edited by an image-editing model, for our three
configurations and for Tree-Ring, Gaussian Shading and SEAL.

Protocol. A panel of prompts is drawn by a hash ranking, balanced over the four categories. For
every source image Qwen3-VL-8B-Instruct is shown the image and the fixed constraint of its
category (the paper's constraint table, in `_edits_lib.CATEGORY_CONSTRAINTS`) and writes one
image-specific edit instruction (greedy decoding, one JSON plan, validated). Qwen-Image-Edit
renders the edit from the source image and the instruction. The same pipeline edits the
watermarked and the unwatermarked source. Detectors are fixed before editing: for our method
the boundary threshold delta (activation target of `configs/detection.yaml`, cross-fitted over
the calibration prompts) and the score threshold tau (from the unedited clean scores at the FPR
targets of the config), for each baseline tau from its unedited clean scores; nothing is
recalibrated on edits. Reported per category and overall: TPR after editing, paired retention
(detected before and after / detected before), and FPR on the edited unwatermarked images, averaged
over the five hyperplane frames x five PRF keys of the config for our method. The paper's
robustness figure uses the 20% FPR target (the largest of `calibration.fpr_targets`).

Model identities, revisions and render settings (seed, steps, guidance) are pinned in
`_edits_lib.py` (`PLANNER`, `EDITOR`); they are not part of `configs/assets.yaml`.

## Prerequisites

* The SD2.1 candidate corpus of `experiments/generation/` with its slot-1 embedding cache
  (`--cache`, our trained encoder) for the **calibration prompts**: the prompt list used by the
  ordinary-edit robustness experiment. `--prompts` (JSONL rows `{"prompt_uid", "text"}`) is that
  list; the panel is drawn from it and its unedited images calibrate every detector. The number of
  prompts must let each activation target select a whole number of images in the cross-fitting
  folds (`watermark.calibration.calibrate_delta` raises otherwise).
* The detector master key file used for the corpus selections (`--key-file`); candidates are
  selected by Gumbel selection with the bare prompt identity as selector identity, as in
  `experiments/robustness`.
* For every baseline, a generation run over the same prompts with one slot
  (`experiments/baselines/generate.py --num-slots 1`), and its key file for scoring.
* The encoder checkpoint (`scripts/download_checkpoint.py`).

## Environments

| Command | Environment |
|---|---|
| `select_panel.py`, `build_sources.py`, `score_ours.py`, `prepare_baseline_scoring.py`, `summarize.py` | any environment with the packages of `environment.yml` that these CPU steps use (`numpy`, the `watermark` package); no GPU |
| `plan_instructions.py`, `render_edits.py`, `embed_edits.py` | the main environment (`environment.yml`) on a GPU. Qwen3-VL needs a `transformers` release with `Qwen3VLForConditionalGeneration`; `render_edits.py` needs `diffusers` with `QwenImageEditPipeline` and enough GPU memory for the full pipeline in bfloat16. The versions the paper's edits were rendered with are not recorded |
| `experiments/baselines/score.py` | the baseline's own environment (`baselines/README.md`): Tree-Ring and Gaussian Shading in `environment-tree-gaussian.yml`, SEAL in `environment-seal.yml` |

## Commands, in order

Run from the repository root. Every GPU step accepts `--shard-index I --num-shards K` and can be rerun to
resume.

```bash
# 1. The panel: 20 prompts per category from the calibration prompts.
python experiments/generative_edits/select_panel.py --prompts prompts.jsonl \
    --per-category 20 --output-dir edits

# 2. The distinct source images to edit: clean controls, every watermarked image that some
#    (configuration, frame, key) replicate selects for a panel prompt, and the baseline images.
python experiments/generative_edits/build_sources.py --panel edits/panel.jsonl \
    --prompts prompts.jsonl --cache embeddings.npz --corpus-dir corpus --key-file keys/detector.key \
    --baseline tree_ring=baselines/tree_ring --baseline gaussian_shading=baselines/gaussian_shading \
    --baseline seal=baselines/seal --output-dir edits

# 3. Edit instructions (Qwen3-VL), then 4. rendering (Qwen-Image-Edit); GPU.
python experiments/generative_edits/plan_instructions.py --sources edits/sources.jsonl \
    --output-dir edits/plans --shard-index 0 --num-shards 1
python experiments/generative_edits/render_edits.py --sources edits/sources.jsonl \
    --plans-dir edits/plans --output-dir edits/rendered --shard-index 0 --num-shards 1
```

**Manual review.** The paper reviews every edited image by eye: the requested change must be
visible and the main subject and scene must be preserved. This is not automated; inspect
`edits/rendered/images/` (named by source image hash; the instruction is in the manifest) before
scoring. Edits that fail review are a protocol decision of the authors and are not handled by the
scripts.

```bash
# 5. Our method: embed the edits, then score every (configuration, frame, key) replicate.
python experiments/generative_edits/embed_edits.py --render-dir edits/rendered \
    --checkpoint checkpoints/checkpoint-step-00019840.pt --output edits/edited_embeddings.npz
python experiments/generative_edits/score_ours.py --panel edits/panel.jsonl --prompts prompts.jsonl \
    --cache embeddings.npz --corpus-dir corpus --key-file keys/detector.key \
    --edited-embeddings edits/edited_embeddings.npz --output-dir edits/ours

# 6. Baselines: list the images, run the native detector in the method's environment, repeat for
#    gaussian_shading and seal (SEAL in environment-seal.yml).
python experiments/generative_edits/prepare_baseline_scoring.py --method tree_ring \
    --baseline-dir baselines/tree_ring --prompts prompts.jsonl --panel edits/panel.jsonl \
    --render-dir edits/rendered --output edits/tree_ring_images.jsonl
python experiments/baselines/score.py --method tree_ring --official-root third_party/checkouts \
    --master-key-file keys/baselines.key --images edits/tree_ring_images.jsonl \
    --output edits/tree_ring_scores.jsonl

# 7. The summary table.
python experiments/generative_edits/summarize.py --panel edits/panel.jsonl \
    --ours edits/ours/replicates.csv --baseline tree_ring=edits/tree_ring_scores.jsonl \
    --baseline gaussian_shading=edits/gaussian_shading_scores.jsonl \
    --baseline seal=edits/seal_scores.jsonl --output-dir edits/summary
```

`--master-key-file` of `score.py` is the key file the baseline images were generated with.
`edits/summary/summary.csv` has one row per method, FPR target and category (`all` pools the
panel): `edited_tpr`, `retention`, `edited_fpr`, `original_tpr`, `calibration_fpr`. A baseline
score file written in shards must be concatenated before step 7.

## Files

* `select_panel.py` panel; `build_sources.py` source images; `plan_instructions.py` Qwen3-VL;
  `render_edits.py` Qwen-Image-Edit; `embed_edits.py`, `score_ours.py` our detector;
  `prepare_baseline_scoring.py` baseline image lists; `summarize.py` tables.
* `_edits_lib.py` shared helpers: panel rule, planner prompts, plan validation, replicate loop,
  frozen-threshold metrics.
