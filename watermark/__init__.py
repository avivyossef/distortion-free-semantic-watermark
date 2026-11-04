"""Black-box, distortion-free semantic watermark: method and shared evaluation utilities.

Importing this package is cheap (numpy only); torch, diffusers, transformers, open_clip
and tpips are imported inside the functions that need them.

Public API (import as ``from watermark.<module> import <name>``)
------------------------------------------------------------------

config
    ``load_detection_config(path=None)``  load and validate ``configs/detection.yaml``;
    returns a ``DetectionConfig`` (``.setting(n, m)``, ``.frame_seeds()``, ``.frame_ids()``,
    ``.key_trials()``, ``.folds``, ``.fpr_targets``, ``.set_sizes``, ``.exception(name)``).
    ``load_assets(path=None)``  checkpoint binding and pinned model revisions
    (``configs/assets.yaml``). ``HYPERPLANE_COUNT``.
prf
    ``score_code(key, bits)``  HMAC-SHA256 keyed score of a code, uniform in (0, 1).
    ``score_table(key, m)``  scores of all 2**m codes, indexed by packed code ID.
    ``derive_trial_key(master, trial)``  key of replicate ``trial`` (trial 0 = master).
    ``generate_master_key()``  fresh 32-byte master key.  ``derive_seed(master, *parts)``.
coding
    ``sample_hyperplanes(seed, dim)``  64 unit hyperplane normals; use the first M.
    ``code_embedding(vec, planes)``  bit tuple of a vector.
    ``bits_to_code_ids(bool_array)``  pack bits into integer code IDs.
    ``project(embeddings, planes, m)``  signed projections (sign = bit, abs = margin).
selection
    ``select_candidate(codes, key, rng)``  Gumbel selection over bit tuples -> ``SelectionTrace``.
    ``select_code_ids(code_ids, scores, rng)``  Gumbel selection over a score table:
    ``c* = argmax_c log F(c) / n_c`` (ties uniform), then a uniform member of ``c*``.
    ``cell_rng(*parts)``  selector RNG from a textual identity.
selection_rules
    The paper's appendix rule comparison. ``select(rule, code_ids, scores=, layer_tables=, rng=)``
    with ``rule`` in ``RULES = ("representative", "gumbel", "tournament")``;
    ``select_representative`` (representative sampling); ``detector_table(rule, ...)``;
    ``tournament_layer_tables(key, m=, layers=)``; ``tournament_layers(n)``.
detection
    ``replicate_scores(candidates, cleans, uids, planes, key, frame_id=, key_trial=, n=, m=,
    activation=, radius=, folds=, rule="gumbel", slot=None)``  select (Gumbel selection; other rules
    only in the rule comparison), calibrate delta and score one (frame, key) replicate ->
    ``ReplicateScores`` (``.clean.scores``, ``.watermarked.scores``, ``.deltas``,
    ``.selected_index``, ``.table`` ...).
    ``selector_rng(rule, frame_id, key_trial, n, m, uid, slot=None)``  the selector stream of
    the paper's runs: ``slot=None`` for experiments that edit the selected images, ``slot=s`` for
    the set-size experiments.
    ``boundary_max_score(code_id, margins, table, delta=, radius=)``  one image.
    ``boundary_scores(code_ids, margins, table, deltas, radius=)``  many images.
    ``score_embeddings(embeddings, planes, m, table, deltas, radius=)``  e.g. edited images.
    ``aggregate_scores([scores_b1, ..., scores_bB])``  set score = mean of B image scores.
calibration
    ``calibrate_delta(margins, target_activation=)``  delta for an activation target on the
    given margin population.  ``crossfit_deltas(margins, activation, folds)``  per-image
    delta from the other folds.  ``clean_threshold(clean_scores, fpr)``  frozen tau.
metrics
    ``roc_curve``, ``auroc``, ``tpr_at_fpr(roc, fpr)``, ``clean_detection_roc(clean, wm, fprs)``,
    ``interpolate_roc``, ``mean_roc(rocs, grid)``, ``paired_retention(orig, edited, tau)``.
embedding
    ``load_encoder(name, device=, checkpoint=)`` for ``ENCODER_NAMES`` (ours, dinov2_base,
    clip_vit_b32, dinov3_l16, tpips_embed, csd_style, depth_layout); every encoder has
    ``.embed(pil_images) -> (n, d)`` unit-norm float64.  ``embed_paths(encoder, paths)``.
generation
    ``panel_seed_rows(...)``, ``load_sd21_pipeline()``, ``generate_prompt_panel(...)``  the
    SD2.1 candidate corpus (see the module docstring for the on-disk layout).
attacks
    ``EDIT_IDS`` (the 20 ordinary edits), ``EDIT_GROUPS``, ``render_edit(image, edit_id)``.
io
    ``sha256_file``, ``canonical_json``, ``read_jsonl``, ``write_jsonl``, ``read_key_file``,
    ``pixel_sha256``.
"""
