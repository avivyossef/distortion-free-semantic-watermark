#!/usr/bin/env python3
"""Exact-code detection on the GPT Image panel (appendix "Closed-API Black-Box Generator Access").

Embeds the 50 panel prompts' candidates and clean controls with the trained encoder and scores
them with the ``closed_api`` settings of ``configs/detection.yaml`` (exact-code detection,
five hyperplane frames, five keys, set sizes B in the configured list, B-sets formed by a
fixed prompt partition). Key ``t`` is ``derive_trial_key(master, t)``, named ``key-0(t+1)``.

Writes to ``--output-dir``: ``summary.csv`` (per B: AUROC and TPR at the configured FPR targets
over the replicates), ``mean_roc.csv`` (mean ROC curve and deviation), ``diversity.csv``
(mean within-prompt cosine, mean distinct codes, selection statistics), ``per_prompt.csv`` (mean
distinct codes per prompt) and ``replicates.csv``.

    python experiments/closed_api/evaluate.py --panel-root panel --prompts prompts.jsonl \
        --checkpoint checkpoints/checkpoint-step-00019840.pt --key-file master.key --output-dir results/closed_api
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from closed_api_lib import MODEL_ID, PROMPT_COUNT, evaluate_embeddings, load_closed_api_panel  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402
from watermark.embedding import embed_paths, load_encoder  # noqa: E402
from watermark.io import read_jsonl, read_key_file  # noqa: E402
from watermark.prf import derive_trial_key  # noqa: E402


def _write(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True, help="32-byte master key")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_detection_config()
    setting = config.exception("closed_api")
    if setting["radius"] != 0:
        raise ValueError("the closed-API study uses exact-code detection")
    prompt_rows = read_jsonl(args.prompts)[:PROMPT_COUNT]
    panel = load_closed_api_panel(panel_root=args.panel_root, prompt_rows=prompt_rows, model_id=MODEL_ID,
                                  candidate_count=setting["N"])
    paths = [path for candidates, clean in zip(panel.candidate_paths, panel.clean_paths) for path in (*candidates, clean)]
    encoder = load_encoder("ours", device=args.device, checkpoint=args.checkpoint)
    vectors = embed_paths(encoder, paths).reshape(len(panel.prompt_uids), setting["N"] + 1, -1)
    master = read_key_file(args.key_file)
    keys = {"key-%02d" % (trial + 1): derive_trial_key(master, trial) for trial in config.key_trials()}
    result = evaluate_embeddings(
        candidates=vectors[:, :setting["N"]], clean=vectors[:, setting["N"]], prompt_uids=panel.prompt_uids,
        frame_seeds=config.frame_seeds(), keys=keys, m=setting["M"], batch_sizes=setting["set_sizes"],
        grouping_seed=setting["grouping_seed"], fpr_targets=config.fpr_targets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    summary = []
    for item in result["summary"]:
        for cell in item["fixed_fpr"]:
            summary.append({"B": item["B"], "replicates": item["replicate_count"],
                            "batches_per_replicate": item["batches_per_replicate"], "mean_auc": item["mean_auc"],
                            "std_auc": item["std_auc"], "min_auc": item["min_auc"], "max_auc": item["max_auc"],
                            **cell})
    _write(args.output_dir / "summary.csv", ("B", "replicates", "batches_per_replicate", "mean_auc", "std_auc",
                                             "min_auc", "max_auc", "requested_fpr", "mean_empirical_fpr", "mean_tpr",
                                             "std_tpr"), summary)
    _write(args.output_dir / "mean_roc.csv", ("B", "mean_auc", "fpr", "mean_tpr", "std_tpr"), result["mean_roc"])
    diversity = ("mean_pairwise_cosine_similarity", "mean_pairwise_cosine_distance", "pair_count",
                 "mean_distinct_codes", "all_distinct_rate", "mean_score_gap")
    _write(args.output_dir / "diversity.csv", ("metric", "value"), [{"metric": name, "value": result[name]}
                                                                    for name in diversity])
    distinct = defaultdict(list)
    for row in result["rows"]:
        distinct[row["prompt_uid"]].append(row["distinct_codes"])
    _write(args.output_dir / "per_prompt.csv", ("prompt_uid", "mean_distinct_codes"),
           [{"prompt_uid": uid, "mean_distinct_codes": float(np.mean(values))} for uid, values in distinct.items()])
    _write(args.output_dir / "replicates.csv", ("key_id", "frame_seed", "B", "batch_count", "auc"),
           [{name: row[name] for name in ("key_id", "frame_seed", "B", "batch_count", "auc")}
            for row in result["replicates"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
