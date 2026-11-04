# GPT Image as a closed-API generator (Appendix "Closed-API Black-Box Generator Access")

Applies the watermark (Gumbel selection) to OpenAI's GPT Image API using sampling access only: per prompt, eight
independently requested candidates and one independently requested clean control (low quality,
1024 x 1024). Detection is exact-code (Hamming radius 0) at M = 8 over five hyperplane frames and
five keys; for B = 5 the 50 prompts are split into ten fixed groups of five (identical
membership for the clean and watermarked populations).

Produces: the ROC figure (`plot_roc.py`), the AUROC / TPR numbers and code-occupancy
statistics (`evaluate.py`), and the eight-candidate example grid (`umbrella_grid.py`).

Settings (N, M, radius, set sizes, partition seed, frames, keys, FPR targets) come from the
`closed_api` entry of `configs/detection.yaml`, read by `watermark.config.load_detection_config()`.

## Prerequisites

- Environment: the repository's `pyproject.toml` dependencies; a GPU for embedding. The API
  client uses the standard library only.
- An OpenAI API key in the `OPENAI_API_KEY` environment variable (never on the command line, never in a
  file). Generation makes 450 paid image requests with the model `gpt-image-2.5-sunburst`.
  Provider request identifiers are not stored.
- A prompt list `prompts.jsonl` (`{"prompt_uid": "...", "text": "..."}` per row); the first 50 rows
  in file order are used.
- The trained encoder checkpoint (`scripts/download_checkpoint.py`) and a detector master key:

```bash
python experiments/generation/new_key.py --output keys/detector.key
```

## 1. Generate the panel

Finished images are not requested again, so an interrupted run can be repeated:

```bash
export OPENAI_API_KEY=...
python experiments/closed_api/generate_panel.py --prompts prompts.jsonl --output-dir panel
```

## 2. Detect

```bash
python experiments/closed_api/evaluate.py --panel-root panel --prompts prompts.jsonl \
    --checkpoint checkpoints/checkpoint-step-00019840.pt --key-file keys/detector.key \
    --output-dir results/closed_api
```

Outputs: `summary.csv` (per set size: AUROC mean and spread over the 25 replicates; TPR and
empirical FPR at each configured FPR target), `mean_roc.csv`, `diversity.csv` (mean within-prompt
cosine similarity, mean distinct codes per prompt, selection statistics), `per_prompt.csv`
(mean distinct codes of each prompt over the five hyperplane frames) and `replicates.csv`.

## 3. Figures

```bash
python experiments/closed_api/plot_roc.py --mean-roc results/closed_api/mean_roc.csv \
    --output figures/closed_api_gpt_image_roc.pdf
python experiments/closed_api/umbrella_grid.py --panel-root panel --prompt-uid <prompt_uid> \
    --output figures/closed_api_umbrella_grid.jpg
```

## Notes

- The 25 key / frame repetitions reuse the same 50 prompts and images; they are not 25 independent
  samples. With B = 5 each repetition has only ten clean groups.
- `umbrella_grid.py` is a reconstruction from the figure caption (a 2 x 4 montage of one prompt's
  candidates).
- The closed model cannot be pinned: the API may serve different weights over time.
