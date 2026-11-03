# Semantic-encoder ablation for clean detection

Produces the appendix figure "Semantic-encoder ablation for clean watermark detection"
(Appendix "Additional Results"): the watermarking scheme and clean-image detection rerun with
our encoder, DINOv2-base, CLIP ViT-B/32, DINOv3-L/16 and TPIPS-Embed on the same candidate
images, B = 1, every configuration of `configs/detection.yaml`, mean ROC curves over five
hyperplane frames x five PRF keys (25 replicates per encoder and configuration). Each encoder
uses normalized embeddings in its native dimension; the detector settings are the
configuration-specific ones of the main clean-detection experiment.

## Prerequisites

The evaluation corpus (`experiments/generation`) and the detector master key; the encoder
checkpoint for ours; for TPIPS-Embed, the `tpips` package (see `watermark.embedding`). GPU for
embedding, CPU for scoring.

## Commands

For each encoder `E` in `ours dinov2_base clip_vit_b32 dinov3_l16 tpips_embed`:

```bash
python experiments/generation/embed_corpus.py --corpus-dir corpus --prompts prompts.jsonl \
    --encoder E --slots 1 [--checkpoint checkpoints/checkpoint-step-00019840.pt] \
    --output-dir embeddings_E --shard-index 0 --num-shards 1
python experiments/generation/merge_embeddings.py --shards-dir embeddings_E --output cache_E.npz
python experiments/encoder_ablation/run_ablation.py --cache cache_E.npz --encoder E \
    --key-file keys/detector.key --output-dir ablation --shard-index 0 --num-shards 1
```

Then merge and plot:

```bash
python experiments/encoder_ablation/plot_ablation.py --results-dir ablation --output-dir figures
```

Writes `encoder_ablation_summary.csv` (AUROC and TPR per encoder and configuration, mean and
standard deviation over replicates) and `semantic_encoder_detection_ablation.pdf`. Ours with
`--slots 1` reproduces the single-image row of `clean_detection` exactly (same selections and
calibration).
