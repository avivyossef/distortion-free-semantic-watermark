#!/usr/bin/env python3
"""Merge the encoder-ablation runs: summary table and the mean-ROC figure.

    python experiments/encoder_ablation/plot_ablation.py --results-dir ablation --output-dir figures

Reads ``ablation/<encoder>/replicates-shard-*.jsonl`` for every encoder present and checks that
each covers the complete replicate grid. Writes ``encoder_ablation_summary.csv`` (mean and
standard deviation over replicates of the AUROC and the TPR at each FPR target) and
``semantic_encoder_detection_ablation.pdf`` (one panel per (N, M): the mean ROC curve of every
encoder with +-1 replicate standard deviation shaded; the dotted diagonal is random guessing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_corpus import ENCODERS, replicate_cells, write_csv_rows  # noqa: E402
from run_ablation import ROC_GRID, target_label  # noqa: E402

from watermark.config import DetectionConfig, load_detection_config  # noqa: E402
from watermark.io import read_jsonl  # noqa: E402

LABELS = {"ours": "Ours (DINOv2 + LoRA + MLP)", "dinov2_base": "DINOv2-base", "clip_vit_b32": "CLIP ViT-B/32",
          "dinov3_l16": "DINOv3-L/16", "tpips_embed": "TPIPS-Embed"}
COLORS = {"ours": "#b71c1c", "dinov2_base": "#0072b2", "clip_vit_b32": "#009e73", "dinov3_l16": "#e69f00",
          "tpips_embed": "#7b3294"}


def load_encoder_rows(directory: Path, cfg: DetectionConfig) -> List[Dict[str, Any]]:
    rows = [r for path in sorted(Path(directory).glob("replicates-shard-*.jsonl")) for r in read_jsonl(path)]
    seen = [(r["N"], r["M"], r["frame_id"], r["key_trial"]) for r in rows]
    expected = {(s.n, s.m, f, t) for s in cfg.settings for f, _, t in replicate_cells(cfg)}
    if len(set(seen)) != len(seen) or set(seen) != expected:
        raise ValueError("%s does not cover the replicate grid exactly once" % directory)
    return rows


def summarize(results: Dict[str, List[Dict[str, Any]]], cfg: DetectionConfig) -> List[Dict[str, Any]]:
    fields = ["auc"] + ["%s_%s" % (kind, target_label(t)) for t in cfg.fpr_targets
                        for kind in ("tpr_roc", "tpr_interp", "frozen_tpr", "frozen_fpr")]
    out = []
    for encoder, rows in results.items():
        for setting in cfg.settings:
            members = [r for r in rows if (r["N"], r["M"]) == (setting.n, setting.m)]
            entry: Dict[str, Any] = {"encoder": encoder, "N": setting.n, "M": setting.m,
                                     "replicate_count": len(members)}
            for field in fields:
                values = [float(r[field]) for r in members]
                entry[field + "_mean"] = float(np.mean(values))
                entry[field + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out.append(entry)
    return out


def plot(results: Dict[str, List[Dict[str, Any]]], cfg: DetectionConfig, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(cfg.settings), figsize=(14, 4.6), sharex=True, sharey=True, squeeze=False)
    for ax, setting in zip(axes[0], cfg.settings):
        for encoder, rows in results.items():
            members = [r for r in rows if (r["N"], r["M"]) == (setting.n, setting.m)]
            curves = np.asarray([r["roc_tpr"] for r in members])
            mean = curves.mean(axis=0)
            sd = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
            auc = np.mean([r["auc"] for r in members])
            ax.plot(ROC_GRID, mean, color=COLORS[encoder], lw=2.3 if encoder == "ours" else 1.6,
                    label="%s (%.3f)" % (LABELS[encoder], auc))
            ax.fill_between(ROC_GRID, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=COLORS[encoder],
                            alpha=.055)
        ax.plot([0, 1], [0, 1], ":", color=".5", lw=1)
        ax.set(title="N = %d, M = %d" % (setting.n, setting.m), xlabel="False-positive rate", xlim=(0, 1),
               ylim=(0, 1))
        ax.grid(alpha=.18)
        ax.legend(loc="lower right", fontsize=8, frameon=False, title="Encoder (mean AUROC)", title_fontsize=8)
    axes[0][0].set_ylabel("True-positive rate")
    fig.tight_layout()
    fig.savefig(str(output), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = {name: load_encoder_rows(args.results_dir / name, cfg) for name in ENCODERS
               if (args.results_dir / name).is_dir()}
    if not results:
        raise SystemExit("no encoder results found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "encoder_ablation_summary.csv", summarize(results, cfg))
    plot(results, cfg, args.output_dir / "semantic_encoder_detection_ablation.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
