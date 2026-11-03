#!/usr/bin/env python3
"""Detection, retention and embedding change of the CSD style and depth layout encoders under edits.

Appendix table "Encoder-dependent watermark robustness". Detector settings come from the
``style_depth`` entry of ``configs/detection.yaml``. For every encoder and every (hyperplane
frame, key) replicate, candidate selection and the boundary thresholds use unedited
embeddings; the score threshold targets the configured FPR on the same unedited clean images
that are evaluated, and stays fixed after editing. Results are averaged over the replicates;
retention averages only replicates in which something was detected.

Writes ``replicates.csv`` (one row per encoder, edit, frame and key) and ``table.csv``.

    python experiments/style_depth_encoders/score_edits.py --embeddings embeddings.npz \
        --key-file master.key --output-dir results/style_depth
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from style_depth_lib import EDITS, EDIT_LABELS, ENCODERS, ENCODER_LABELS, cosine_distance, score_replicate  # noqa: E402
from watermark.coding import sample_hyperplanes  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_key_file  # noqa: E402
from watermark.prf import derive_trial_key  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, required=True, help="embed_edits.py output")
    parser.add_argument("--key-file", type=Path, required=True, help="32-byte master key")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_detection_config()
    setting = config.exception("style_depth")
    if setting["calibration_population"] != "evaluation":
        raise ValueError("the style/depth study calibrates the threshold on the evaluated clean images")
    fpr = config.fpr_targets[-1]  # the largest configured target
    master = read_key_file(args.key_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    with np.load(args.embeddings, allow_pickle=False) as data:
        uids = [str(uid) for uid in data["prompt_uids"]]
        original = {encoder: data["%s__original" % encoder] for encoder in ENCODERS}
        edited = {encoder: {edit: data["%s__%s" % (encoder, edit)] for edit in EDITS} for encoder in ENCODERS}

    replicates = []
    for encoder in ENCODERS:
        for frame_index, (seed, frame_id) in enumerate(zip(config.frame_seeds(), config.frame_ids())):
            planes = sample_hyperplanes(seed, original[encoder].shape[-1])
            for trial in config.key_trials():
                result = score_replicate(
                    original[encoder], edited[encoder], uids, planes, derive_trial_key(master, trial),
                    frame_id=frame_id, key_trial=trial, n=setting["N"], m=setting["M"],
                    activation=setting["activation"], radius=setting["radius"], folds=config.folds, fpr=fpr,
                )
                for name, cell in result.items():
                    replicates.append({"encoder": encoder, "edit": name, "frame": frame_index, "key_trial": trial,
                                       **cell})
    with (args.output_dir / "replicates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("encoder", "edit", "frame", "key_trial", "tpr", "clean_fpr",
                                                    "retention"))
        writer.writeheader()
        writer.writerows(replicates)

    with (args.output_dir / "table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["encoder", "transformation", "tpr_percent", "retention_percent", "cosine_distance",
                         "clean_fpr_percent", "replicates_with_detections"])
        for encoder in ENCODERS:
            for name in ("original",) + EDITS:
                rows = [row for row in replicates if row["encoder"] == encoder and row["edit"] == name]
                retention = np.asarray([row["retention"] for row in rows], dtype=np.float64)
                kept = retention[~np.isnan(retention)]
                writer.writerow([
                    ENCODER_LABELS[encoder], "Original" if name == "original" else EDIT_LABELS[name],
                    "%.1f" % (100 * np.mean([row["tpr"] for row in rows])),
                    "" if name == "original" or not len(kept) else "%.1f" % (100 * kept.mean()),
                    "" if name == "original" else "%.3f" % cosine_distance(original[encoder], edited[encoder][name]),
                    "%.1f" % (100 * np.mean([row["clean_fpr"] for row in rows])),
                    len(kept) if name != "original" else "",
                ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
