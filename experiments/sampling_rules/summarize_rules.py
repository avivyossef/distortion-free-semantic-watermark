#!/usr/bin/env python3
"""Merge the sampling-rule runs into the appendix table.

    python experiments/sampling_rules/summarize_rules.py --shards-dir rules --output-dir tables

Writes ``sampling_rules_clean.csv`` (mean and standard deviation over replicates of the AUROC and
the TPR per rule, configuration and FPR target) and ``alternative_sampling_rules.tex`` (the
table: AUROC and TPR at the largest FPR target per rule, the best and second-best value of each
metric colored within each row). If ``edits-shard-*.jsonl`` files are present
(``run_rules.py --edits``), ``sampling_rules_edits.csv`` reports, per rule and configuration,
the mean over the 20 ordinary edits of the edited TPR and the paired retention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))

from detection_corpus import replicate_cells, write_csv_rows  # noqa: E402

from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402
from watermark.selection_rules import RULES  # noqa: E402

NAMES = {"representative": "Representative sampling", "gumbel": "Gumbel", "tournament": "Tournament"}


def target_label(fpr: float) -> str:
    return "%02d" % int(round(100 * fpr))


def load(directory: Path, name: str) -> List[Dict[str, Any]]:
    return [r for path in sorted(Path(directory).glob("%s-shard-*.jsonl" % name)) for r in read_jsonl(path)]


def clean_summary(rows: List[Dict[str, Any]], cfg: DetectionConfig) -> List[Dict[str, Any]]:
    expected = {(rule, s.n, s.m, f, t) for rule in RULES for s in cfg.settings for f, _, t in replicate_cells(cfg)}
    seen = [(r["rule"], r["N"], r["M"], r["frame_id"], r["key_trial"]) for r in rows]
    if len(set(seen)) != len(seen) or set(seen) != expected:
        raise ValueError("shards do not cover the replicate grid exactly once")
    fields = ["auc"] + ["%s_%s" % (kind, target_label(t)) for t in cfg.fpr_targets
                        for kind in ("tpr_roc", "tpr_interp", "frozen_tpr")] + ["mean_distinct_codes"]
    out = []
    for rule in RULES:
        for s in cfg.settings:
            members = [r for r in rows if (r["rule"], r["N"], r["M"]) == (rule, s.n, s.m)]
            entry = {"rule": rule, "N": s.n, "M": s.m, "replicate_count": len(members)}
            for field in fields:
                values = [float(r[field]) for r in members]
                entry[field + "_mean"] = float(np.mean(values))
                entry[field + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out.append(entry)
    return out


def _marks(values: List[float]) -> List[str]:
    shown = [round(v, 3) for v in values]
    levels = sorted(set(shown), reverse=True)
    return ["samplingbest" if v == levels[0] else "samplingsecond" if len(levels) > 1 and v == levels[1] else ""
            for v in shown]


def latex_table(summary: List[Dict[str, Any]], cfg: DetectionConfig) -> str:
    """Table of AUROC and TPR (at the largest FPR target) per rule with best/second colors per row."""
    label = target_label(max(cfg.fpr_targets))
    lines = ["\\begin{table}[H]", "\\centering", "\\definecolor{samplingbest}{HTML}{007A53}",
             "\\definecolor{samplingsecond}{HTML}{C76A00}", "\\small",
             "\\caption{Clean detection under alternative distortion-free sampling rules. TPR is read from the mean "
             "ROC curve at the %g\\%% clean FPR target; single-image detection, mean over frames and keys.}"
             % (100 * max(cfg.fpr_targets)), "\\label{tab:alternative_sampling_rules}",
             "\\begin{tabular}{crrrrrr}", "\\toprule",
             "& " + " & ".join("\\multicolumn{2}{c}{%s}" % NAMES[r] for r in RULES) + " \\\\",
             "\\((N,M)\\) & " + " & ".join("AUROC & TPR" for _ in RULES) + " \\\\", "\\midrule"]
    for s in cfg.settings:
        entries = [next(e for e in summary if (e["rule"], e["N"], e["M"]) == (r, s.n, s.m)) for r in RULES]
        auc = [e["auc_mean"] for e in entries]
        tpr = [e["tpr_interp_%s_mean" % label] for e in entries]
        cells = []
        for a, ma, t, mt in zip(auc, _marks(auc), tpr, _marks(tpr)):
            cells += [("\\textcolor{%s}{\\textbf{%.3f}}" % (ma, a)) if ma else "%.3f" % a,
                      ("\\textcolor{%s}{\\textbf{%.3f}}" % (mt, t)) if mt else "%.3f" % t]
        lines.append("\\((%d,%d)\\) & %s \\\\" % (s.n, s.m, " & ".join(cells)))
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n"


def edit_summary(rows: List[Dict[str, Any]], cfg: DetectionConfig) -> List[Dict[str, Any]]:
    out = []
    target = max(cfg.fpr_targets)
    for rule in RULES:
        for s in cfg.settings:
            members = [r for r in rows if (r["rule"], r["N"], r["M"], r["fpr_target"]) == (rule, s.n, s.m, target)
                       and r["edit_id"] != "identity"]
            tpr = [r["tpr"] for r in members]
            retention = [r["retention"] for r in members if r["retention"] is not None]
            out.append({"rule": rule, "N": s.n, "M": s.m, "fpr_target": target, "rows": len(members),
                        "edited_tpr": float(np.mean(tpr)) if tpr else float("nan"),
                        "retention": float(np.mean(retention)) if retention else float("nan")})
    return out


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = clean_summary(load(args.shards_dir, "clean"), cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "sampling_rules_clean.csv", summary)
    (args.output_dir / "alternative_sampling_rules.tex").write_text(latex_table(summary, cfg), encoding="utf-8")
    edits = load(args.shards_dir, "edits")
    if edits:
        write_csv_rows(args.output_dir / "sampling_rules_edits.csv", edit_summary(edits, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
