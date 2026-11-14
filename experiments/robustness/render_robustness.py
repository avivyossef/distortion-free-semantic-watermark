#!/usr/bin/env python3
"""Robustness table, category figure, per-edit table and detector-ablation table.

Inputs: ``per_edit.csv`` from ``summarize_robustness.py`` (ours), the per-edit results of the native baselines
(``--baseline-per-edit NAME=per_edit.csv``, written by ``experiments/baselines/robustness.py``) and
the generative-edit summary (``--generative summary.csv``, written by
``experiments/generative_edits/summarize.py``; methods ``ours_nN_mM`` and the baseline ids). All
are brought to one row schema: ``method, edit_id, group, edited_tpr, edited_fpr, retention``
(fractions; ``group`` is ``appearance_compression``, ``geometric``, ``generative``,
``generative_all`` or ``unedited`` for the ``identity`` row holding the TPR and FPR before editing),
optionally ``label`` and ``unedited_tpr``.

    python experiments/robustness/render_robustness.py --summary-dir robustness/summary \
        --baseline-per-edit "Tree-Ring=analysis/tree_ring_robustness/per_edit.csv" \
        --baseline-per-edit "Gaussian Shading=analysis/gaussian_shading_robustness/per_edit.csv" \
        --baseline-per-edit "SEAL=analysis/seal_robustness/per_edit.csv" \
        --generative edits/summary/summary.csv --output-dir tables

Writes ``experiment3_categories.tex`` (Table: robustness at the nominal FPR),
``experiment3_robustness_categories.pdf`` (category figure), ``robustness_per_edit.tex`` (per-edit
figure content: edited TPR and paired retention, with the transformation-family exposure
during DINOv2 pretraining and our fine-tuning) and ``robustness_detector_ablation.tex``
(boundary-aware versus exact-code detection).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _robustness_lib import EDIT_GROUPS, EDIT_IDS  # noqa: E402
from summarize_robustness import edit_results  # noqa: E402

from watermark.attacks import EDITS  # noqa: E402
from watermark.config import DetectionConfig, load_detection_config  # noqa: E402

GROUP_TITLES = {"appearance_compression": "Appearance and compression", "geometric": "Geometric transformations",
                "generative": "Edits from an image-editing model"}
EDIT_LABELS = {
    "jpeg_q90": "JPEG Q90", "jpeg_q75": "JPEG Q75", "gaussian_blur_0p5": "Gaussian blur $r=0.5$",
    "gaussian_blur_1p5": "Gaussian blur $r=1.5$", "gaussian_noise_2": "Gaussian noise $\\sigma=2$",
    "gaussian_noise_6": "Gaussian noise $\\sigma=6$", "brightness_0p8": "Brightness $\\times0.8$",
    "brightness_1p2": "Brightness $\\times1.2$", "contrast_0p8": "Contrast $\\times0.8$",
    "contrast_1p2": "Contrast $\\times1.2$", "median_3": "Median $3\\times3$", "median_5": "Median $5\\times5$",
    "salt_pepper_0p01": "Salt-and-pepper $1\\%$", "salt_pepper_0p03": "Salt-and-pepper $3\\%$",
    "crop_50": "Center crop $50\\%$", "crop_75": "Center crop $75\\%$",
    "offcenter_crop_75": "Off-center crop $75\\%$", "offcenter_crop_90": "Off-center crop $90\\%$",
    "translate_32": "Shift 32 px", "translate_8": "Shift 8 px",
}
# Transformation-family exposure: (used in DINOv2 pretraining, used in our fine-tuning).
EXPOSURE = {
    "jpeg": (False, True), "gaussian_blur": (True, True), "gaussian_noise": (False, True),
    "brightness": (True, True), "contrast": (True, True), "median": (False, False),
    "salt_pepper": (False, False), "crop": (True, False), "offcenter_crop": (True, False),
    "translation": (False, True),
}
GENERATIVE_EXPOSURE = (False, False)
BASELINE_NAMES = {"tree_ring": "Tree-Ring", "gaussian_shading": "Gaussian Shading", "seal": "SEAL"}
GENERATIVE_LABELS = {"attribute_edit": "Attribute change", "background_cleanup": "Background cleanup",
                     "cat_attack": "Cat insertion", "camera_viewpoint_change": "Camera-viewpoint change"}


def _number(value: Any) -> Optional[float]:
    """A float, or ``None`` when empty or NaN (retention is undefined when nothing was detected)."""
    number = float(value) if value not in ("", None) else float("nan")
    return None if number != number else number


def read_ours_per_edit(path: Path) -> List[Dict[str, Any]]:
    """Read the ordinary-edit summary with every detector and FPR target."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "fpr_target" not in (reader.fieldnames or []):
            raise ValueError("%s must be the per_edit.csv from summarize_robustness.py" % path)
        return [
            {**row, "N": int(row["N"]), "M": int(row["M"]), "fpr_target": float(row["fpr_target"]),
             "tpr": float(row["tpr"]), "fpr": float(row["fpr"]),
             "retention": _number(row["retention"]), "cosine": _number(row["cosine"])}
            for row in reader
        ]


def ours_results_at_fpr(per_edit: List[Dict[str, Any]], cfg: DetectionConfig, fpr: float) -> List[Dict[str, Any]]:
    rows = edit_results(per_edit, cfg, fpr)
    if not rows:
        raise ValueError("our per-edit summary has no boundary-aware results at FPR target %g" % fpr)
    return rows


def read_baseline_per_edit(name: str, path: Path, fpr: float) -> List[Dict[str, Any]]:
    """``edit_results`` rows of one baseline from its ``per_edit.csv`` (``baselines/robustness.py``)."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        source = [r for r in csv.DictReader(handle) if abs(float(r["requested_fpr"]) - fpr) < 1e-12]
    if not source:
        raise ValueError("%s has no rows at FPR target %g" % (path, fpr))
    rows.append({"method": name, "edit_id": "identity", "group": "unedited",
                 "edited_tpr": float(source[0]["unedited_tpr"]), "edited_fpr": float(source[0]["calibration_fpr"]),
                 "retention": None})
    for r in source:
        rows.append({"method": name, "edit_id": r["edit_id"], "group": r["category"],
                     "edited_tpr": float(r["edited_tpr"]), "edited_fpr": float(r["edited_fpr"]),
                     "retention": _number(r["retention"]), "unedited_tpr": float(r["unedited_tpr"])})
    return rows


def display_name(raw: str) -> str:
    """``ours_n8_m4`` -> ``Ours (N=8, M=4)``; ``tree_ring`` -> ``Tree-Ring``; other names unchanged."""
    if raw.startswith("ours_n") and "_m" in raw:
        n, m = raw[len("ours_n"):].split("_m")
        return "Ours (N=%s, M=%s)" % (n, m)
    return BASELINE_NAMES.get(raw, raw)


def read_generative_summary(path: Path, fpr: float) -> List[Dict[str, Any]]:
    """``edit_results`` rows of the generative-edit summary at ``fpr`` (per category and pooled)."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if abs(float(r["fpr_target"]) - fpr) >= 1e-12:
                continue
            pooled = r["category"] == "all"
            rows.append({"method": display_name(r["method"]), "edit_id": r["category"],
                         "group": "generative_all" if pooled else "generative",
                         "label": GENERATIVE_LABELS.get(r["category"], r["category"]),
                         "edited_tpr": float(r["edited_tpr"]), "edited_fpr": float(r["edited_fpr"]),
                         "retention": _number(r["retention"]), "unedited_tpr": float(r["original_tpr"])})
    if not rows:
        raise ValueError("%s has no rows at FPR target %g" % (path, fpr))
    return rows


def method_label(name: str) -> str:
    """``Ours (N=8, M=4)`` -> ``Ours $(N{=}8,M{=}4)$``; other names unchanged."""
    if name.startswith("Ours (") and name.endswith(")"):
        return "Ours $(%s)$" % name[6:-1].replace(" ", "").replace("=", "{=}")
    return name


def methods_in_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return list(dict.fromkeys(str(row["method"]) for row in rows))


def _mean(values: Sequence[Optional[float]]) -> float:
    finite = [v for v in values if v is not None and v == v]
    return float(np.mean(finite)) if finite else float("nan")


def category_values(rows: Sequence[Dict[str, Any]], method: str, group: str) -> Dict[str, float]:
    """Unweighted mean over the edits of a group: TPR, retention, FPR and the unedited TPR."""
    chosen = [r for r in rows if r["method"] == method and r["group"] == group]
    if group == "generative":  # the summary's pooled "all" row averages the 80 edits, not the 4 categories
        chosen = [r for r in rows if r["method"] == method and r["group"] == "generative_all"] or chosen
    if not chosen:
        return {}
    identity = [r for r in rows if r["method"] == method and r["group"] == "unedited"]
    unedited = [r.get("unedited_tpr") for r in chosen]
    return {"tpr": _mean([r["edited_tpr"] for r in chosen]), "retention": _mean([r["retention"] for r in chosen]),
            "fpr": _mean([r["edited_fpr"] for r in chosen]),
            "before": _mean(unedited) if all(v is not None for v in unedited)
            else (identity[0]["edited_tpr"] if identity else float("nan"))}


def categories_table(rows: Sequence[Dict[str, Any]], fpr: float) -> str:
    """The robustness table: TPR, retention and FPR (percent) per group and method."""
    groups = [g for g in GROUP_TITLES if any(r["group"] == g for r in rows)]
    short = {"appearance_compression": "Appearance/compression", "geometric": "Geometric",
             "generative": "Generative edits"}
    lines = ["\\begin{table*}[t]", "\\centering", "\\scriptsize",
             "\\caption{Robustness at $%g\\%%$ nominal clean FPR (all values in \\%%). Columns report post-edit TPR, "
             "paired retention conditioned on original detection, and post-edit FPR on unwatermarked images.}"
             % (100 * fpr), "\\label{tab:experiment3_robustness}", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{l%s}" % "".join("ccc@{\\qquad}" if i < len(groups) - 1 else "ccc"
                                                for i in range(len(groups))), "\\toprule",
             "& " + " & ".join("\\multicolumn{3}{c}{%s}" % short[g] for g in groups) + " \\\\",
             " ".join("\\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(groups))),
             "Method & " + " & ".join("TPR & Ret. & FPR" for _ in groups) + " \\\\", "\\midrule"]
    for method in methods_in_order(rows):
        cells = []
        for group in groups:
            values = category_values(rows, method, group)
            cells.extend("%.1f" % (100 * values[k]) if values else "--" for k in ("tpr", "retention", "fpr"))
        lines.append("%s & %s \\\\" % (method_label(method), " & ".join(cells)))
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]) + "\n"


def rank_colors(values: Sequence[float]) -> List[str]:
    """``best``/``second``/``plain`` per value; ranks use one decimal and ties share a rank."""
    shown = [round(100 * v, 1) if v == v else None for v in values]
    levels = sorted({v for v in shown if v is not None}, reverse=True)
    return ["best" if v is not None and levels and v == levels[0]
            else "second" if v is not None and len(levels) > 1 and v == levels[1] else "plain" for v in shown]


def _cell(value: Optional[float], rank: str) -> str:
    if value is None or value != value:
        return "--"
    text = "%.1f" % (100 * value)
    return text if rank == "plain" else "\\textcolor{metric%s}{\\textbf{%s}}" % (rank, text)


def per_edit_tables(rows: Sequence[Dict[str, Any]], fpr: float) -> str:
    """Per-edit edited TPR and paired retention (percent) with best/second colors, as LaTeX."""
    methods = methods_in_order(rows)
    lookup = {(r["method"], r["edit_id"]): r for r in rows}
    sections = [(g, [e for e in EDIT_IDS if e in EDIT_GROUPS[g]] if g != "generative" else
                 [r["edit_id"] for r in rows if r["group"] == "generative" and r["method"] == methods[0]])
                for g in GROUP_TITLES if g != "generative" or any(r["group"] == "generative" for r in rows)]
    out = ["\\definecolor{metricbest}{HTML}{007A53}", "\\definecolor{metricsecond}{HTML}{C76A00}", "\\begingroup",
           "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\newcommand{\\trainyes}{\\(\\checkmark\\)}",
           "\\newcommand{\\trainno}{\\(\\times\\)}",
           "\\textcolor{metricbest}{\\textbf{Best}} / \\textcolor{metricsecond}{\\textbf{Second}} "
           "at %g\\%% nominal clean FPR\\par\\smallskip" % (100 * fpr)]
    for title, field in (("Edited TPR (\\%)", "edited_tpr"), ("Paired retention (\\%)", "retention")):
        out += ["\\textbf{%s}\\par\\smallskip" % title,
                "\\begin{tabular}{lcc%s}" % ("r" * len(methods)), "\\toprule",
                "Edit & \\shortstack{DINOv2\\\\pretrain} & \\shortstack{Ours\\\\fine-tune} & "
                + " & ".join(method_label(m) for m in methods) + " \\\\", "\\midrule"]
        for index, (group, edits) in enumerate(sections):
            if index:
                out.append("\\midrule")
            out.append("\\multicolumn{%d}{l}{\\textbf{%s}} \\\\" % (3 + len(methods), GROUP_TITLES[group]))
            for edit_id in edits:
                values = [lookup[(m, edit_id)][field] if (m, edit_id) in lookup else None for m in methods]
                ranks = rank_colors([float("nan") if v is None else v for v in values])
                pretrain, tuned = EXPOSURE[EDITS[edit_id].family] if edit_id in EDITS else GENERATIVE_EXPOSURE
                label = EDIT_LABELS.get(edit_id) or str(lookup[(methods[0], edit_id)].get("label") or edit_id)
                out.append("%s & %s & %s & %s \\\\" % (
                    label, "\\trainyes" if pretrain else "\\trainno", "\\trainyes" if tuned else "\\trainno",
                    " & ".join(_cell(v, r) for v, r in zip(values, ranks))))
        out += ["\\bottomrule", "\\end{tabular}", "\\par\\medskip"]
    return "\n".join(out + ["\\endgroup"]) + "\n"


def ablation_table(per_edit: Sequence[Dict[str, Any]], fpr: float) -> str:
    """Boundary-aware versus exact-code detection: means over the 20 edits at ``fpr``."""
    out = ["\\begin{table}[H]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}",
           "\\caption{Boundary-aware detector ablation at $%g\\%%$ nominal clean FPR. Values are unweighted means "
           "over the 20 transformation settings; TPR and FPR are measured after transformation, retention is "
           "conditioned on original detection. Each pair uses the same images.}" % (100 * fpr),
           "\\label{tab:robustness_detector_ablation}", "\\begin{tabular}{ccrrr}", "\\toprule",
           "$(N,M)$ & Detector & TPR & Retention & FPR \\\\", "\\midrule"]
    settings = sorted({(int(r["N"]), int(r["M"])) for r in per_edit}, key=lambda item: item[0])
    for index, (n, m) in enumerate(settings):
        if index:
            out.append("\\addlinespace")
        for position, (detector, name) in enumerate((("exact", "Exact code"), ("boundary", "Boundary aware"))):
            members = [r for r in per_edit if (int(r["N"]), int(r["M"]), r["detector"]) == (n, m, detector)
                       and abs(float(r["fpr_target"]) - fpr) < 1e-12 and r["edit_id"] != "identity"]
            out.append("%s & %s & %.2f & %.2f & %.2f \\\\" % (
                "$(%d,%d)$" % (n, m) if position == 0 else "", name,
                100 * _mean([float(r["tpr"]) for r in members]),
                100 * _mean([None if r["retention"] in ("", None) else float(r["retention"]) for r in members]),
                100 * _mean([float(r["fpr"]) for r in members])))
    return "\n".join(out + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n"


def plot_categories(rows: Sequence[Dict[str, Any]], output: Path) -> None:
    """Category figure: unedited (open) and edited (filled) TPR per method; retention annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    groups = [g for g in GROUP_TITLES if any(r["group"] == g for r in rows)]
    methods = methods_in_order(rows)
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.2, 2.5 * len(groups)), sharex=True, squeeze=False)
    for ax, group in zip(axes[:, 0], groups):
        for y, method in enumerate(methods):
            values = category_values(rows, method, group)
            if not values:
                continue
            color = "#2F6B9A" if method.startswith("Ours") else "#666666"
            ax.plot([100 * values["before"], 100 * values["tpr"]], [y, y], color=color, alpha=.45, lw=1.2)
            ax.scatter([100 * values["before"]], [y], s=28, facecolor="white", edgecolor=color, zorder=3)
            ax.scatter([100 * values["tpr"]], [y], s=28, color=color, zorder=3)
            ax.text(103, y, "retention %.1f" % (100 * values["retention"]), va="center", fontsize=8)
        ax.set_yticks(range(len(methods)), methods)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(GROUP_TITLES[group], loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="x", color="#e5e5e5", lw=.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1, 0].set_xlabel("Detection rate (%): open = before editing, filled = after editing")
    fig.tight_layout()
    fig.savefig(str(output), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> int:
    cfg = load_detection_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-dir", type=Path, required=True, help="summarize_robustness.py output")
    parser.add_argument("--baseline-per-edit", action="append", default=[], metavar="NAME=PER_EDIT_CSV")
    parser.add_argument("--generative", type=Path, help="generative_edits/summarize.py summary.csv")
    parser.add_argument("--fpr", type=float, default=max(cfg.fpr_targets))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    per_edit = read_ours_per_edit(args.summary_dir / "per_edit.csv")
    rows = ours_results_at_fpr(per_edit, cfg, args.fpr)
    for item in args.baseline_per_edit:
        name, _, path = item.partition("=")
        rows.extend(read_baseline_per_edit(name, Path(path), args.fpr))
    if args.generative is not None:
        rows.extend(read_generative_summary(args.generative, args.fpr))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment3_categories.tex").write_text(categories_table(rows, args.fpr), encoding="utf-8")
    (args.output_dir / "robustness_per_edit.tex").write_text(per_edit_tables(rows, args.fpr), encoding="utf-8")
    (args.output_dir / "robustness_detector_ablation.tex").write_text(ablation_table(per_edit, args.fpr),
                                                                       encoding="utf-8")
    plot_categories(rows, args.output_dir / "experiment3_robustness_categories.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
