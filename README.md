# Distortion-Free Semantic Watermarks for Black-Box Generative Models

Aviv Yossef, Or Patashnik, Or Zamir (Tel Aviv University)

Code for the paper. The watermark needs only sampling access to a generator:
for each request it draws N candidate images, maps each one to an M-bit code
through random hyperplanes over a semantic encoder's embedding, and returns
the candidate whose code has the highest keyed pseudorandom score. Over keys,
the output distribution equals the generator's. Detection recomputes the code
of a suspect image, also scores the codes reachable by flipping bits that lie
close to a hyperplane boundary, and compares the best keyed score (averaged
over a set of B images) with a threshold calibrated on unwatermarked images.

## Repository layout

| Path | Contents |
|---|---|
| `watermark/` | The method: keyed PRF, hyperplane codes, Algorithm 1 and the alternative sampling rules, boundary-aware detection, calibration, metrics, encoders, SD2.1 generation, the ordinary edits |
| `encoder/` | The semantic encoder: DINOv2-base with a rank-8 LoRA and an MLP head, its training data and training loop |
| `configs/detection.yaml` | Every detector setting the paper uses; all experiments read it |
| `configs/assets.yaml` | Pinned model revisions and the encoder checkpoint binding |
| `experiments/` | One directory per paper result, each with a README giving the exact commands |
| `third_party/clone.sh` | Clones the native baselines at the commits the paper used |
| `tests/` | CPU tests on synthetic data |
| `docs/` | Project page (GitHub Pages) |

## Installation

```bash
conda env create -f environment.yml
conda activate semantic-watermark
```

The native baselines (Tree-Ring, Gaussian Shading, SEAL) need their own
environments; see `experiments/baselines/README.md`. Experiments that generate
or embed images need a CUDA GPU.

Download the trained encoder and verify its SHA-256:

```bash
python scripts/download_checkpoint.py --output-dir checkpoints
```

## Keys

Watermark keys are never part of the repository. Create a master key with

```bash
python experiments/generation/new_key.py --output keys/master.key
```

and keep it private. Reruns with a new key are statistically equivalent to the
paper's runs but not image-for-image identical.

## Determinism

Every random choice is derived from a key or a fixed seed: generation seeds from
the seed key, prompt, slot and candidate position; hyperplane frames from the
frame seeds in `configs/detection.yaml`; PRF keys from the master key. The same
inputs therefore select the same candidates and give the same detection scores.
Images regenerated on a different GPU, driver or library version agree with the
originals only to floating-point tolerance, not bit for bit, so pin the
environment in `environment.yml` when comparing runs.

## Reproducing the paper

Start with `experiments/generation`, which builds the SD2.1 candidate corpus
that most experiments read. Each directory's README lists its inputs and the
exact commands.

| Paper result | Directory |
|---|---|
| Semantic encoder training | `experiments/encoder_training` |
| Same-prompt cosine and semantic entropy across encoders; entropy-vs-radius figure; stability under perturbations; distinct-code figures | `experiments/encoder_comparison` |
| SD2.1 candidate corpus (prompts, candidates, clean controls, embeddings, selections) | `experiments/generation` |
| Generation quality and diversity (FID, relative Vendi); FID against real images and CLIPScore | `experiments/quality` |
| Clean detection table, operating points, ROC figures | `experiments/clean_detection` |
| Selected detector settings | `experiments/detector_sweep` |
| Native baselines (all baseline rows and cells) | `experiments/baselines` |
| Robustness to ordinary edits; encoder similarity versus retention | `experiments/robustness` |
| Generative-edit panel | `experiments/generative_edits` |
| Semantic-encoder ablation | `experiments/encoder_ablation` |
| Alternative sampling rules (Gumbel, Tournament) | `experiments/sampling_rules` |
| Style versus depth encoders | `experiments/style_depth_encoders` |
| Closed-API generator (GPT Image) | `experiments/closed_api` |

## Tests

```bash
python -m pytest -q tests
```

The tests run on CPU with synthetic inputs and do not download models.

## Citation

```bibtex
@misc{yossef2026distortionfree,
  title  = {Distortion-Free Semantic Watermarks for Black-Box Generative Models},
  author = {Yossef, Aviv and Patashnik, Or and Zamir, Or},
  year   = {2026},
  note   = {Preprint}
}
```
