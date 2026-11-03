#!/usr/bin/env python3
"""Merge the sweep shards into the detection-settings table.

    python experiments/detector_sweep/summarize_sweep.py --shards-dir sweep --output sweep/summary.csv

One row per (N, M, activation target, radius): means over the replicates of the AUROC, the clean
TPR, the edited TPR, edited FPR and retention at each FPR target, and their differences from
exact-code detection (activation 0, radius 0) on the same replicates. The final choice of the
three settings of ``configs/detection.yaml`` is made by reading this table: the paper favors
settings that raise clean and edited detection without inflating the false-positive rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))

from detection_corpus import replicate_cells, write_csv_rows  # noqa: E402

from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402

IDENTITY = ("N", "M", "frame_id", "key_trial")


def merge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean over replicates per grid point, with deltas from the (0, 0) exact-code reference."""
    reference = {tuple(r[k] for k in IDENTITY): r for r in rows if r["activation"] == 0.0 and r["radius"] == 0}
    metrics = [k for k in rows[0] if k not in IDENTITY + ("activation", "radius")]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["N"], r["M"], r["activation"], r["radius"]), []).append(r)
    out = []
    for (n, m, activation, radius), members in sorted(groups.items()):
        entry: Dict[str, Any] = {"N": n, "M": m, "activation": activation, "radius": radius,
                                 "replicate_count": len(members)}
        for name in metrics:
            values = [r[name] for r in members]
            entry[name] = float(np.mean([v for v in values if v is not None])) if any(
                v is not None for v in values) else float("nan")
            deltas = [r[name] - reference[tuple(r[k] for k in IDENTITY)][name] for r in members
                      if r[name] is not None and reference[tuple(r[k] for k in IDENTITY)][name] is not None]
            entry[name + "_delta_from_exact"] = float(np.mean(deltas)) if deltas else float("nan")
        out.append(entry)
    return out


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for path in sorted(args.shards_dir.glob("sweep-shard-*.jsonl")) for r in read_jsonl(path)]
    cells = {tuple(r[k] for k in IDENTITY) for r in rows}
    expected = {(s.n, s.m, frame_id, trial) for s in cfg.settings for frame_id, _, trial in replicate_cells(cfg)}
    if cells != expected:
        raise SystemExit("shards do not cover the replicate grid")
    write_csv_rows(args.output, merge(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
