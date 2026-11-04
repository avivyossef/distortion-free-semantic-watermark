# SD2.1 candidate corpus

Produces the Stable Diffusion 2.1 candidate corpus that every detection experiment of the paper
reads (Detection and Robustness Protocols, "Prompts and datasets"): for each prompt, watermarked
candidates and an unwatermarked control generated with identical model and settings. Per
prompt there are S independent slots (S is the largest set size B of `configs/detection.yaml`),
each with N candidates (the largest N of the configurations) and one clean image; a detection set
of B images uses slots 1..B, and a configuration (N, M) uses the first N candidates of a slot.
The candidates are embedded with the semantic encoder (Section "Method", Appendix "Semantic
Encoder Training").

The embedding cache feeds `clean_detection`, `robustness`, `detector_sweep`, `encoder_ablation`,
`sampling_rules`, and `quality`; each recomputes its own selections from it with the paper's
Gumbel selection (`watermark.selection`; the appendix rule comparison in `sampling_rules` also replays
representative sampling and Tournament). The selector stream differs by experiment, as in the
paper's runs (`watermark.detection.selector_rng`): the set-size experiments (`clean_detection`)
use the pool identity `<uid>:slot:<s>`, the experiments that edit the selected images (`robustness`,
`detector_sweep`, `encoder_ablation`, `style_depth_encoders`, `generative_edits`) the bare prompt
identity, `quality` and `closed_api` their own domains.

## Prerequisites

- Environment: `environment.yml` (GPU for generation and embedding; the cache I/O
  runs on CPU).
- Encoder checkpoint: `python scripts/download_checkpoint.py` (verified by SHA-256).
- Stable Diffusion 2.1 is loaded from the Hugging Face Hub at the revision pinned in
  `configs/assets.yaml`; the cached UNet is checked against its SHA-256.
- Prompt lists, JSONL with one row per prompt: `{"prompt_uid": "...", "text": "..."}`.
  An optional `"native_key_id"` per row labels the seeds of that prompt's clean controls.
  The paper's 1,000-prompt list carries it (`python scripts/download_checkpoint.py --asset prompts
  --output-dir prompts`), so the paper's
  clean images regenerate exactly; any other list may omit it.
  The evaluation corpus uses 1,000 captions of the COCO-2017 *validation* split held out from
  encoder training; the detector sweep uses a separate development list from the COCO-2017
  *train* split (see `detector_sweep`). The prompt lists are inputs: they are not shipped.
- Two secret 32-byte key files: a generation-seed key and the detector master key.

```bash
python experiments/generation/new_key.py --output keys/seed.key
python experiments/generation/new_key.py --output keys/detector.key
```

## 1. Generate the corpus (GPU; shard over devices)

```bash
python experiments/generation/generate_corpus.py --prompts prompts.jsonl \
    --seed-key-file keys/seed.key --output-dir corpus --shard-index 0 --num-shards 1
```

Each prompt writes `corpus/<prompt_uid>/slot_NNN/{candidate_PP.png, clean.png}` and
`panel.json`. Seeds derive from the key, prompt, slot and position, so raising the number of
candidates does not change earlier ones. Add `--candidates` / `--slots` only to build a smaller
corpus (for example `--slots 1` for the development set).

## 2. Embed the images (GPU; shard over devices)

```bash
python experiments/generation/embed_corpus.py --corpus-dir corpus --prompts prompts.jsonl \
    --encoder ours --checkpoint checkpoints/checkpoint-step-00019840.pt \
    --output-dir embeddings --shard-index 0 --num-shards 1
python experiments/generation/merge_embeddings.py --shards-dir embeddings --output cache.npz
```

`cache.npz` is the embedding cache (layout in `detection_corpus.py`). The appendix encoder ablation embeds
the first slot with the other encoders (`--encoder dinov2_base|clip_vit_b32|dinov3_l16|tpips_embed
--slots 1`, one cache each); see `encoder_ablation`.

Note: a corpus regenerated from a fresh seed key, and detector results from a fresh master key,
are not key-matched to any earlier run.
