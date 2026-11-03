"""The semantic encoder: model, training pairs, training loop and checkpoint loading.

- ``encoder.model``: ``DinoLoRAMetric`` (DINOv2-base + LoRA + MLP head), ``pair_loss``.
- ``encoder.checkpoint``: ``load_encoder`` (checkpoint file -> model + image processor).
- ``encoder.attacks``: the positive-pair augmentation registry.
- ``encoder.data``: pair records, balanced sampler, pair dataset.
- ``encoder.training``: epoch loop, validation, resumable checkpoints, ``run_training``.
- ``encoder.sdxl``: SDXL final images and fifth-step previews, seed allocation.

Importing this package is cheap; torch and the model libraries are imported by the
modules that need them.
"""
