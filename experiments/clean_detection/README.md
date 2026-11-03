# Clean detection

Produces the clean-detection experiment of the paper (Detection and Robustness Protocols,
"Clean-detection experiment"): the ours rows of the clean-detection table (AUROC and
TPR@20% FPR for B = 1, 5, 10), the clean operating-points table (TPR at 1%, 5% and 20% FPR), and
the two ROC figures (linear and logarithmic FPR axes). The native baselines' rows come from
`experiments/baselines`.

Protocol: for each configuration (N, M), five hyperplane frames x five PRF keys (25 replicates),
each slot of the corpus is an independent candidate pool; selection, boundary calibration
(delta on the selected watermarked images, cross-fitted over the folds) and boundary-aware
scoring run per slot; a set of B images scores the mean of its slots' scores; the score
threshold tau is frozen on the clean set scores. All settings (N, M, activation, radius, FPR
targets, folds, frames, keys, set sizes) come from `configs/detection.yaml`.

## Prerequisites

The evaluation corpus and its embedding cache (`experiments/generation`, 1,000 COCO-2017
validation prompts, all slots) and the detector master key. CPU only.

## Commands

```bash
# 1. replicates (shard over processes; --workers parallelises inside a shard)
python experiments/clean_detection/run_clean_detection.py --cache cache.npz \
    --key-file keys/detector.key --output-dir clean_detection \
    --shard-index 0 --num-shards 1 --workers 4

# 2. merge: summary.csv, detection_table_ours.tex, operating_points_ours.tex, mean_roc.json
python experiments/clean_detection/summarize_clean_detection.py \
    --shards-dir clean_detection --output-dir clean_detection/summary

# 3. figures: experiment2_detection_roc.pdf and experiment2_detection_roc_log.pdf
python experiments/clean_detection/plot_clean_roc.py \
    --mean-roc clean_detection/summary/mean_roc.json --output-dir figures
```

`summary.csv` reports, per (N, M, B), the AUROC and, per FPR target, the TPR read from the mean
ROC curve (`tpr_interp`, the value the tables show), the TPR at the best ROC vertex not above the
target (`tpr_roc`) and the TPR with its achieved FPR at the frozen clean threshold (`frozen_tpr`,
`frozen_fpr`). The two TPR readings differ when scores are discrete (short codes).
