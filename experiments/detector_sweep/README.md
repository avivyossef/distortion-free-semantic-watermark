# Detector-settings sweep

Produces the evidence behind the selected detector settings (Appendix "Detection and Robustness
Protocols", "Detection settings", Table of selected settings): for each (N, M), the boundary
threshold delta (as a clean-activation target) and the Hamming radius are swept on a
*development set* drawn from the COCO-2017 training captions, separate from the evaluation
prompts, considering clean detection, detection after editing and the corresponding
false-positive rate. The sweep is the one experiment that varies activation and radius; its grid
is `sweep_config.yaml`. The three selected settings of `configs/detection.yaml` are read off the
resulting table; HMAC keyed scoring only.

For every replicate (configuration x frame x key, from `configs/detection.yaml`) and grid
point: AUROC and clean TPR; and, with delta, radius and tau fixed on the unedited images, the
edited TPR, edited FPR and paired retention averaged over the 20 ordinary edits. Differences
from exact-code detection (activation 0, radius 0) are reported alongside.

## Prerequisites

A development prompt list (JSONL, `{"prompt_uid", "text"}`; COCO-2017 train captions, disjoint
from the evaluation prompts; its size must make every activation target an integer number of
images in the cross-fitting folds, e.g. 200 prompts), and the detector master key (`keys/detector.key`, see the generation README).

## Commands

```bash
# 1. development corpus: first slot only (GPU), embeddings, and edits
python experiments/generation/new_key.py --output keys/dev_seed.key   # fresh generation-seed key, as in the generation README
python experiments/generation/generate_corpus.py --prompts dev_prompts.jsonl \
    --seed-key-file keys/dev_seed.key --output-dir dev_corpus --slots 1
python experiments/generation/embed_corpus.py --corpus-dir dev_corpus --prompts dev_prompts.jsonl \
    --encoder ours --checkpoint checkpoints/checkpoint-step-00019840.pt --output-dir dev_embeddings
python experiments/generation/merge_embeddings.py --shards-dir dev_embeddings --output dev_cache.npz
python experiments/robustness/embed_ordinary_edits.py --corpus-dir dev_corpus --cache dev_cache.npz \
    --prompts dev_prompts.jsonl --key-file keys/detector.key \
    --checkpoint checkpoints/checkpoint-step-00019840.pt --output-dir dev_edits

# 2. sweep (CPU; shard over processes) and merge
python experiments/detector_sweep/run_sweep.py --cache dev_cache.npz --prompts dev_prompts.jsonl \
    --edits dev_edits --key-file keys/detector.key --output-dir sweep \
    --shard-index 0 --num-shards 1
python experiments/detector_sweep/summarize_sweep.py --shards-dir sweep --output sweep/summary.csv
```

`sweep/summary.csv` has one row per (N, M, activation, radius). The detection-settings table of
the paper lists the chosen target activation and radius per (N, M).
