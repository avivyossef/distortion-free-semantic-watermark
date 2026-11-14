# Robustness to ordinary edits, and encoder similarity versus retention

Produces, from one evaluation:

- Robustness table: post-edit TPR, paired retention and post-edit FPR at the nominal 20% clean
  FPR, per category (appearance/compression: 14 settings, geometric: 6 settings), and the
  generative-edit columns when those rows are supplied.
- Category figure and per-edit figure (edited TPR and paired retention for the 20 settings,
  with the transformation-family exposure during DINOv2 pretraining and our fine-tuning).
- Detector-ablation table: boundary-aware versus exact-code detection on the same images.
- Appendix "Encoder Similarity and Watermark Retention": the similarity-versus-retention figure
  and the Spearman correlations over the 20 settings.

Protocol (Detection and Robustness Protocols, "Ordinary edits"): 200 prompts, B = 1 (slot 1),
genuine candidate pools; candidates are selected by Gumbel selection with the bare prompt identity as
selector identity. The same edit (`watermark.attacks`: the 20 settings) is applied to the
watermarked and to the unwatermarked image; delta, the Hamming radius and tau are fixed on the
unedited images (tau from clean scores only, at each FPR target of `configs/detection.yaml`) and
never recalibrated. Paired retention is the fraction of images detected before the edit that are
detected afterwards. Results average the five frames x five keys. The exact-code ablation uses
radius 0 on the same images.

## Prerequisites

The evaluation corpus and cache (`experiments/generation`), the trained-encoder checkpoint, the
detector master key, and a prompt list of the evaluation prompts (a subset of the cache's
prompts; 200 in the paper; give the same list to `experiments/baselines`). The baseline columns
come from `experiments/baselines` and the generative-edit rows from `experiments/generative_edits`.

## Commands

```bash
# 1. render and embed the 20 edits of every image the detector will see (GPU; shardable)
python experiments/robustness/embed_ordinary_edits.py --corpus-dir corpus --cache cache.npz \
    --prompts prompts200.jsonl --key-file keys/detector.key \
    --checkpoint checkpoints/checkpoint-step-00019840.pt --output-dir edits \
    --shard-index 0 --num-shards 1

# 2. edited TPR / FPR / retention per replicate and detector (CPU; shardable)
python experiments/robustness/analyze_robustness.py --cache cache.npz --prompts prompts200.jsonl \
    --edits edits --key-file keys/detector.key --output-dir robustness \
    --shard-index 0 --num-shards 1

# 3. merge: per_edit.csv, categories.csv, edit_results.csv
python experiments/robustness/summarize_robustness.py --shards-dir robustness \
    --output-dir robustness/summary

# 4. tables and figures: experiment3_categories.tex, robustness_per_edit.tex,
#    robustness_detector_ablation.tex, experiment3_robustness_categories.pdf
#    (baseline per_edit.csv files: experiments/baselines/robustness.py; generative summary:
#    experiments/generative_edits/summarize.py; both optional)
python experiments/robustness/render_robustness.py --summary-dir robustness/summary \
    --baseline-per-edit "Tree-Ring=analysis/tree_ring_robustness/per_edit.csv" \
    --baseline-per-edit "Gaussian Shading=analysis/gaussian_shading_robustness/per_edit.csv" \
    --baseline-per-edit "SEAL=analysis/seal_robustness/per_edit.csv" \
    --generative edits/summary/summary.csv --output-dir tables

# 5. similarity vs retention: cosines of each baseline's watermarked/edited images in our
#    encoder space (image lists of experiments/baselines/list_images.py and attack_images.py),
#    then the figure and the Spearman correlations
python experiments/robustness/edit_cosines.py --method "Tree-Ring" \
    --original-images out/tree_ring_images.jsonl --edited-images out/tree_ring_edited_images.jsonl \
    --checkpoint checkpoints/checkpoint-step-00019840.pt --output analysis/tree_ring_cosines.csv
python experiments/robustness/plot_retention_cosine.py \
    --results robustness/summary/per_edit.csv \
    --baseline-per-edit "Tree-Ring=analysis/tree_ring_robustness/per_edit.csv" \
    --cosines analysis/tree_ring_cosines.csv --output-dir figures
```

`--fpr` (default: the largest FPR target, the paper's 20%) selects the nominal FPR of the
headline outputs of steps 3-5 (repeat step 5's `edit_cosines.py` and the `--baseline-per-edit` /
`--cosines` arguments for every baseline). The formats of the baseline and generative-edit inputs
are documented at the top of `render_robustness.py` and `edit_cosines.py`. `_robustness_lib.py` is shared
with `detector_sweep` and `sampling_rules`.
