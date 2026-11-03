# Generation quality, diversity and prompt alignment

Produces the quality results of the paper for our three configurations and the three native
baselines (Tree-Ring, Gaussian Shading, SEAL):

- the table "Fixed-key generation quality and diversity" (Experiments, "Generation Quality and
  Diversity"): the matched-population FID between watermarked and clean generations, and the
  relative same-prompt Vendi score in a frozen DINOv2-base feature space;
- the table "Generation Quality and Prompt Alignment" (appendix "Generation Quality and Prompt
  Alignment"): FID against real COCO-2017 validation images and OpenCLIP image-prompt similarity,
  each watermarked population against its matched clean population.

Quality is measured at fixed keys, as the paper states: one PRF key per population, since the
distortion-free guarantee holds over the key and each fixed key may deviate from it. The FID of
our method is reported for `--num-keys` fixed keys (the paper uses ten), each with every
hyperplane frame of `configs/detection.yaml`.

## Prerequisites

- GPU for generation, embedding, Inception and OpenCLIP; the selection and aggregation scripts are
  CPU only. Python environment of the repository plus `pytorch-fid` (Inception weights),
  `open_clip_torch` and `scipy`.
- The trained encoder checkpoint (`python scripts/download_checkpoint.py`).
- The SD2.1 candidate corpus of `experiments/generation` and its embedding cache. The three
  corpora below differ only in prompts and slots; they are generated and embedded with the
  scripts of that experiment (see its README for keys and sharding).
- Native-baseline generations from `experiments/baselines` (its README gives the environments
  of the three methods and the output layout read here).
- Prompt lists, JSONL rows `{"prompt_uid": ..., "text": ...}`, not shipped:
  `prompts.jsonl` is the evaluation list of 1,000 held-out COCO-2017 validation captions (the
  list of the clean-detection experiment). The diversity panel and the real-image FID panel
  are derived below.
- A 32-byte detector master key (`keys/detector.key`, from `experiments/generation/new_key.py`):
  the keys used here are `derive_trial_key(master, t)`; the first `replication.keys` of them
  are the keys of the detection experiments.

All commands run from the repository root. Every GPU step shards over prompts with
`--shard-index I --num-shards K` (shown once with `0`/`1`).

## 1. Matched-population FID (10,000 images per population)

`prompts.jsonl` has 1,000 prompts, each with 10 slots of 32 candidates, so a population holds
10,000 images. The corpus and the embedding cache are those of the clean-detection experiment.

```bash
python experiments/generation/generate_corpus.py --prompts prompts.jsonl \
    --seed-key-file keys/seed.key --output-dir corpus --shard-index 0 --num-shards 1
python experiments/generation/embed_corpus.py --corpus-dir corpus --prompts prompts.jsonl \
    --encoder ours --checkpoint checkpoints/checkpoint-step-00019840.pt \
    --output-dir embeddings --shard-index 0 --num-shards 1
python experiments/generation/merge_embeddings.py --shards-dir embeddings --output cache.npz
```

Select the populations: for every configuration, frame and key, the images chosen by the
paper's selector, and for every key a matched control that draws one of the same N candidates
uniformly.

```bash
python experiments/quality/select_populations.py --cache cache.npz --key-file keys/detector.key \
    --num-keys 10 --output selections.npz
```

Generate the native baselines with 10 slots (their own clean image per slot is the matched
control; `--official-root`, key file and environments as in the baselines README), then compute
the Inception activations of every candidate and every native image once:

```bash
python experiments/baselines/generate.py --method tree_ring --prompts prompts.jsonl --num-slots 10 \
    --official-root third_party/checkouts --master-key-file keys/baselines.key --output-dir native/tree_ring
#   ... the same for --method gaussian_shading and --method seal (SEAL in its own environment)

python experiments/quality/inception_activations.py --source corpus --corpus-dir corpus \
    --prompts prompts.jsonl --output-dir activations/ours --shard-index 0 --num-shards 1
python experiments/quality/inception_activations.py --source native --corpus-dir native/tree_ring \
    --prompts prompts.jsonl --output-dir activations/tree_ring --shard-index 0 --num-shards 1
#   ... the same for gaussian_shading and seal
```

Compute the FIDs (CPU). Per key, the five frames are averaged, then the keys; the report has one
row per configuration and one per baseline:

```bash
python experiments/quality/fid10k.py --selections selections.npz --ours-activations activations/ours \
    --native-activations tree_ring=activations/tree_ring gaussian_shading=activations/gaussian_shading \
    seal=activations/seal --output-dir results/fid10k
```

Outputs: `results/fid10k/pairs.csv` (one FID per key, frame and configuration, and per baseline)
and `results/fid10k/summary.csv`.

## 2. Same-prompt diversity (relative Vendi score)

60 prompts with 60 slots of 32 candidates each. The images chosen by the selector (our
encoder's embeddings choose) are compared by their frozen DINOv2-base embeddings, which do not
depend on our encoder. The clean reference of our method is a uniform draw among the same N
candidates of every slot; a baseline uses its own clean images.

```bash
python experiments/quality/subset_prompts.py --prompts prompts.jsonl --count 60 \
    --output prompts_diversity.jsonl
python experiments/generation/generate_corpus.py --prompts prompts_diversity.jsonl \
    --seed-key-file keys/seed.key --slots 60 --output-dir corpus_diversity --shard-index 0 --num-shards 1
for encoder in ours dinov2_base; do
  python experiments/generation/embed_corpus.py --corpus-dir corpus_diversity \
      --prompts prompts_diversity.jsonl --encoder $encoder \
      --checkpoint checkpoints/checkpoint-step-00019840.pt \
      --output-dir embeddings_diversity/$encoder --shard-index 0 --num-shards 1
  python experiments/generation/merge_embeddings.py --shards-dir embeddings_diversity/$encoder \
      --output cache_diversity_$encoder.npz
done

python experiments/baselines/generate.py --method tree_ring --prompts prompts_diversity.jsonl --num-slots 60 \
    --official-root third_party/checkouts --master-key-file keys/baselines.key --output-dir native_diversity/tree_ring
#   ... gaussian_shading and seal
python experiments/quality/embed_native.py --native-dir native_diversity/tree_ring \
    --prompts prompts_diversity.jsonl --output-dir embeddings_diversity/tree_ring \
    --shard-index 0 --num-shards 1
#   ... gaussian_shading and seal

python experiments/quality/vendi.py --selection-cache cache_diversity_ours.npz \
    --evaluation-cache cache_diversity_dinov2_base.npz --key-file keys/detector.key \
    --native-embeddings tree_ring=embeddings_diversity/tree_ring \
    gaussian_shading=embeddings_diversity/gaussian_shading seal=embeddings_diversity/seal \
    --output-dir results/vendi
```

For each prompt, key and frame (the keys and frames of `configs/detection.yaml`) the relative
Vendi score is Vendi(watermarked) / Vendi(clean) over the 60 chosen images; the table entry is
the percentage change `100 (mean - 1)`. Outputs: `results/vendi/summary.csv` (one row per
configuration and baseline) and `results/vendi/cells.csv`.

## 3. FID against real images and OpenCLIP alignment (appendix table)

FID against real images uses one image per prompt for 5,000 prompts, the captions of the 5,000
COCO-2017 validation images, matched one-to-one with the real images (clean controls are
generated with the corpus, so the clean FID is shared by our three configurations). The key of
a prompt rotates over `--num-keys` keys along the prompts (default 32, `FID5K_NUM_KEYS` in
`_quality_lib.py`), with hyperplane frame 0.

```bash
python experiments/quality/fid5k_panel.py --captions annotations/captions_val2017.json \
    --images-dir val2017 --output prompts_fid5k.jsonl
python experiments/generation/generate_corpus.py --prompts prompts_fid5k.jsonl \
    --seed-key-file keys/seed.key --slots 1 --output-dir corpus_fid5k --shard-index 0 --num-shards 1
python experiments/generation/embed_corpus.py --corpus-dir corpus_fid5k --prompts prompts_fid5k.jsonl \
    --encoder ours --checkpoint checkpoints/checkpoint-step-00019840.pt \
    --output-dir embeddings_fid5k --shard-index 0 --num-shards 1
python experiments/generation/merge_embeddings.py --shards-dir embeddings_fid5k --output cache_fid5k.npz

python experiments/baselines/generate.py --method tree_ring --prompts prompts_fid5k.jsonl --num-slots 1 \
    --official-root third_party/checkouts --master-key-file keys/baselines.key --output-dir native_fid5k/tree_ring
#   ... gaussian_shading and seal

python experiments/quality/inception_activations.py --source real --prompts prompts_fid5k.jsonl \
    --images-dir val2017 --output-dir activations_fid5k/real --shard-index 0 --num-shards 1
python experiments/quality/inception_activations.py --source corpus --corpus-dir corpus_fid5k \
    --prompts prompts_fid5k.jsonl --output-dir activations_fid5k/ours --shard-index 0 --num-shards 1
python experiments/quality/inception_activations.py --source native --corpus-dir native_fid5k/tree_ring \
    --prompts prompts_fid5k.jsonl --output-dir activations_fid5k/tree_ring --shard-index 0 --num-shards 1
#   ... gaussian_shading and seal

python experiments/quality/fid5k.py --cache cache_fid5k.npz --key-file keys/detector.key \
    --ours-activations activations_fid5k/ours --real-activations activations_fid5k/real \
    --native-activations tree_ring=activations_fid5k/tree_ring \
    gaussian_shading=activations_fid5k/gaussian_shading seal=activations_fid5k/seal \
    --output-dir results/fid5k
```

Outputs: `results/fid5k/fid.csv` (FID of each population against the real images) and
`results/fid5k/matched_changes.csv` (relative change `100 (FID_wm / FID_clean - 1)` of each
watermarked population from its matched clean population).

OpenCLIP (ViT-g/14, `laion2b_s34b_b88k`) similarity uses the first slot of the 1,000-prompt
corpus of step 1: the mean over the 1,000 prompts of the cosine similarity between the image and
its prompt. Our watermarked value is the mean over all frame and key populations, the clean value
the mean over the matched controls (`selections.npz` of step 1); a baseline compares its
watermarked and clean images of slot 1.

```bash
python experiments/quality/clipscore.py --selections selections.npz --corpus-dir corpus \
    --prompts prompts.jsonl --native-dirs tree_ring=native/tree_ring \
    gaussian_shading=native/gaussian_shading seal=native/seal --output-dir results/clipscore
```

Output: `results/clipscore/summary.csv` (watermarked mean, clean mean and relative change per
configuration and baseline).

## Notes

- The metrics are descriptive estimates of fixed-key shift and not significance tests; the 95%
  intervals of `fid10k.py` are over the keys.
- The native baselines run with one fixed method state per run (derived from `keys/baselines.key`),
  so each baseline's FID and diversity are fixed-key quantities as well.
- A corpus regenerated from fresh seed and detector keys is not key-matched to any earlier run.
