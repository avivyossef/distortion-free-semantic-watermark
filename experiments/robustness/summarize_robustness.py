#!/usr/bin/env python3
"""Merge the robustness shards: per-edit results, category means and encoder-similarity rows.

    python experiments/robustness/summarize_robustness.py --shards-dir robustness \
        --output-dir robustness/summary [--fpr 0.2]

Writes (``--fpr`` is the nominal clean FPR of the headline tables; default: the largest FPR
target of ``configs/detection.yaml``):

- ``per_edit.csv``: per (N, M, detector, FPR target, edit): mean over the replicates of the edited
  TPR, edited FPR, paired retention and original-edited cosine (``identity`` rows give the
  unedited TPR and FPR).
- ``categories.csv``: unweighted means over the edits of each group (appearance/compression,
  geometric) of TPR, retention and FPR, plus the unedited TPR (input of the robustness table,
  category figure and detector-ablation table).
- ``edit_results.csv``: per (method, edit) rows of the boundary-aware detector at ``--fpr`` in the
  schema shared with the baselines (method, edit_id, group, edited_tpr, edited_fpr, retention,
  cosine); the input of the table, figure and similarity scripts.

Fails unless every configuration x frame x key replicate is present exactly once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _robustness_lib import EDIT_GROUPS, EDIT_IDS  # noqa: E402
from detection_corpus import replicate_cells, write_csv_rows  # noqa: E402

from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402


def load_rows(shards_dir: Path, cfg: DetectionConfig) -> List[Dict[str, Any]]:
    rows = [row for path in sorted(Path(shards_dir).glob("replicates-shard-*.jsonl")) for row in read_jsonl(path)]
    cells = {(row["N"], row["M"], row["frame_id"], row["key_trial"]) for row in rows}
    expected = {(s.n, s.m, frame_id, trial) for s in cfg.settings for frame_id, _, trial in replicate_cells(cfg)}
    per_cell = len(rows) // max(len(cells), 1)
    if cells != expected or len(rows) != len(cells) * per_cell or len(
            {(r["N"], r["M"], r["frame_id"], r["key_trial"], r["detector"], r["edit_id"], r["fpr_target"])
             for r in rows}) != len(rows):
        raise ValueError("shards do not cover the replicate grid exactly once")
    return rows


def _mean(values) -> float:
    finite = [float(v) for v in values if v is not None]
    return float(np.mean(finite)) if finite else float("nan")


def per_edit(rows: List[Dict[str, Any]], cfg: DetectionConfig) -> List[Dict[str, Any]]:
    out = []
    for setting in cfg.settings:
        for detector in sorted({r["detector"] for r in rows}):
            for target in cfg.fpr_targets:
                for edit_id in ("identity",) + tuple(EDIT_IDS):
                    members = [r for r in rows if (r["N"], r["M"], r["detector"], r["fpr_target"], r["edit_id"])
                               == (setting.n, setting.m, detector, target, edit_id)]
                    cosines = [r["cosine"] for r in members]
                    out.append({
                        "N": setting.n, "M": setting.m, "detector": detector, "fpr_target": target,
                        "edit_id": edit_id, "replicate_count": len(members),
                        "tpr": _mean(r["tpr"] for r in members), "fpr": _mean(r["fpr"] for r in members),
                        "retention": _mean(r["retention"] for r in members),
                        "cosine": _mean(cosines),
                        "cosine_sd": float(np.std(cosines, ddof=1)) if len(cosines) > 1 else 0.0})
    return out


def categories(edit_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    keys = sorted({(r["N"], r["M"], r["detector"], r["fpr_target"]) for r in edit_rows})
    for n, m, detector, target in keys:
        pool = [r for r in edit_rows if (r["N"], r["M"], r["detector"], r["fpr_target"]) == (n, m, detector, target)]
        identity = next(r for r in pool if r["edit_id"] == "identity")
        for group, members in EDIT_GROUPS.items():
            chosen = [r for r in pool if r["edit_id"] in members]
            out.append({
                "N": n, "M": m, "detector": detector, "fpr_target": target, "group": group,
                "edit_count": len(chosen), "unedited_tpr": identity["tpr"],
                "tpr": _mean(r["tpr"] for r in chosen), "retention": _mean(r["retention"] for r in chosen),
                "fpr": _mean(r["fpr"] for r in chosen)})
    return out


def group_of(edit_id: str) -> str:
    return next(group for group, members in EDIT_GROUPS.items() if edit_id in members)


def edit_results(edit_rows: List[Dict[str, Any]], cfg: DetectionConfig, target: float) -> List[Dict[str, Any]]:
    """Rows of the boundary-aware detector at ``target`` in the shared ``edit_results`` schema.

    Columns: method, edit_id, group, edited_tpr, edited_fpr, retention, cosine (and N, M).
    Baseline rows produced by ``analyze_baseline_scores.py`` use the same columns.
    """
    out = []
    for setting in cfg.settings:
        for r in edit_rows:
            if (r["N"], r["M"], r["detector"], r["fpr_target"]) == (setting.n, setting.m, "boundary", target):
                identity = r["edit_id"] == "identity"
                out.append({
                    "method": "Ours (N=%d, M=%d)" % (setting.n, setting.m), "edit_id": r["edit_id"],
                    "group": "unedited" if identity else group_of(r["edit_id"]), "edited_tpr": r["tpr"],
                    "edited_fpr": r["fpr"], "retention": r["retention"], "cosine": r["cosine"],
                    "N": setting.n, "M": setting.m})
    return out


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fpr", type=float, default=max(cfg.fpr_targets))
    args = parser.parse_args()
    if args.fpr not in cfg.fpr_targets:
        raise SystemExit("--fpr must be one of the configured FPR targets")
    rows = load_rows(args.shards_dir, cfg)
    edit_rows = per_edit(rows, cfg)
    write_csv_rows(args.output_dir / "per_edit.csv", edit_rows)
    write_csv_rows(args.output_dir / "categories.csv", categories(edit_rows))
    write_csv_rows(args.output_dir / "edit_results.csv", edit_results(edit_rows, cfg, args.fpr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
