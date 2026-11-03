# Alternative distortion-free sampling rules

Produces the appendix table "Alternative distortion-free sampling rules" (Appendix "Additional
Results"): Algorithm 1 versus code-level adaptations of Gumbel sampling and non-distortionary
Tournament sampling (`watermark.selection_rules`), replayed on the same candidate pools and
cached embeddings, every configuration of `configs/detection.yaml`, single-image detection, mean
over five frames x five keys, each rule with its own keyed score and the same detector
configuration and clean-threshold calibration. The appendix also compares the rules after
editing (attacked TPR and conditional retention): that comparison edits the 20 ordinary edits
of `watermark.attacks`.

## Prerequisites

The evaluation corpus and cache (`experiments/generation`) and the detector master key. For the
edit comparison, additionally the trained-encoder checkpoint and a prompt list of the evaluation
prompts (as in `experiments/robustness`). CPU for scoring.

## Commands

```bash
# 1. clean detection under the three rules (shardable)
python experiments/sampling_rules/run_rules.py --cache cache.npz --key-file keys/detector.key \
    --output-dir rules --shard-index 0 --num-shards 1

# 2. detection after edits: embed the edits of every rule's selections, then score them
python experiments/robustness/embed_ordinary_edits.py --corpus-dir corpus --cache cache.npz \
    --prompts prompts200.jsonl --key-file keys/detector.key \
    --checkpoint checkpoints/checkpoint-step-00019840.pt --rules current gumbel tournament \
    --output-dir rule_edits
python experiments/sampling_rules/run_rules.py --cache cache.npz --prompts prompts200.jsonl \
    --edits rule_edits --key-file keys/detector.key --output-dir rules

# 3. merge: sampling_rules_clean.csv, alternative_sampling_rules.tex, sampling_rules_edits.csv
python experiments/sampling_rules/summarize_rules.py --shards-dir rules --output-dir tables
```

The table shows AUROC and the TPR at the 20% FPR target, read from the mean ROC curve as in
the clean-detection experiment; `sampling_rules_clean.csv` also has the TPR at the frozen
clean threshold. Tournament selection needs N to be a power of two.
