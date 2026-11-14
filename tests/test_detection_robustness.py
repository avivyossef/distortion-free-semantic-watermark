"""Robustness experiment: edits, replicate analysis, summaries, tables, baselines and similarity."""

from __future__ import annotations

import csv
import sys

import numpy as np
import pytest
from PIL import Image

import detection_helpers as helpers
import _robustness_lib as _lib
import analyze_robustness as analysis
import detection_corpus as corpus
import edit_cosines
import embed_ordinary_edits
import plot_retention_cosine as similarity
import render_robustness as render
import summarize_robustness as summ

from watermark import detection
from watermark.attacks import EDIT_GROUPS, EDIT_IDS


@pytest.fixture(scope="module")
def world():
    cfg = helpers.small_config()
    cache = helpers.synthetic_cache(prompts=100, slots=2, n=16)
    planes = corpus.frame_planes(cfg, 24)
    keys = corpus.trial_keys(helpers.KEY, cfg)
    sources = _lib.all_selected_sources(cache, cfg, planes, keys)
    return cfg, cache, planes, keys, helpers.synthetic_edits(cache, sources)


def test_selected_indices_equal_the_detector_selection(world):
    cfg, cache, planes, keys, _ = world
    for rule in ("representative", "gumbel", "tournament"):
        for setting in cfg.settings:
            if rule == "tournament" and setting.n & (setting.n - 1):
                continue
            mine = _lib.selected_indices(cache.candidate_vectors[:, 0], cache.prompt_uids, planes["frame-01"],
                                         keys[1], frame_id="frame-01", key_trial=1, n=setting.n, m=setting.m,
                                         rule=rule)
            replay = detection.replicate_scores(
                cache.candidate_vectors[:, 0, :setting.n], cache.clean_vectors[:, 0], cache.prompt_uids,
                planes["frame-01"], keys[1], frame_id="frame-01", key_trial=1, n=setting.n, m=setting.m,
                activation=setting.activation, radius=setting.radius, folds=cfg.folds, rule=rule)
            assert np.array_equal(mine, replay.selected_index)


def test_sources_cover_clean_images_and_every_selected_candidate(world):
    cfg, cache, planes, keys, edits = world
    sources = set(edits.index)
    assert all("%s:clean" % uid in sources for uid in cache.prompt_uids)
    assert all(s.endswith(":clean") or s.split(":")[1] == "candidate" for s in sources)
    more = _lib.all_selected_sources(cache, cfg, planes, keys, ["representative", "gumbel"])
    assert sources <= set(more)


def test_unchanged_images_keep_their_detections(world):
    cfg, cache, planes, keys, _ = world
    sources = _lib.all_selected_sources(cache, cfg, planes, keys)
    identity = helpers.synthetic_edits(cache, sources, strength=0.0)
    setting = cfg.settings[1]
    rows = analysis.analyze_replicate(cache, planes["frame-00"], keys[0], setting, identity, frame_id="frame-00",
                                      key_trial=0, cfg=cfg)
    boundary = [r for r in rows if r["detector"] == "boundary" and r["fpr_target"] == 0.2]
    baseline_row = next(r for r in boundary if r["edit_id"] == "identity")
    for row in boundary:
        if row["edit_id"] != "identity":
            assert row["tpr"] == baseline_row["tpr"] and row["fpr"] == baseline_row["fpr"]
            assert row["retention"] in (None, 1.0) and row["cosine"] == pytest.approx(1.0)
    # exact-code detection uses radius 0 on the same images
    assert {r["radius"] for r in rows if r["detector"] == "exact"} == {0}
    assert baseline_row["fpr"] <= 0.2 + 1e-12


def test_strong_edits_lower_detection_and_cosine(world):
    cfg, cache, planes, keys, _ = world
    sources = _lib.all_selected_sources(cache, cfg, planes, keys)
    mild = helpers.synthetic_edits(cache, sources, strength=0.02)
    strong = helpers.synthetic_edits(cache, sources, strength=2.0)
    args = dict(frame_id="frame-00", key_trial=0, cfg=cfg)

    def mean(edits, field):
        rows = analysis.analyze_replicate(cache, planes["frame-00"], keys[0], cfg.settings[1], edits, **args)
        values = [r[field] for r in rows if r["detector"] == "boundary" and r["fpr_target"] == 0.2
                  and r["edit_id"] != "identity" and r[field] is not None]
        return np.mean(values)

    assert mean(mild, "cosine") > mean(strong, "cosine")
    assert mean(mild, "tpr") > mean(strong, "tpr")


def test_summaries_cover_the_grid_and_group_the_edits(world, tmp_path):
    cfg, cache, planes, keys, edits = world
    rows = []
    for setting in cfg.settings:
        for frame_id, _, trial in corpus.replicate_cells(cfg):
            rows.extend(analysis.analyze_replicate(cache, planes[frame_id], keys[trial], setting, edits,
                                                   frame_id=frame_id, key_trial=trial, cfg=cfg))
    corpus.write_jsonl_rows(tmp_path / "replicates-shard-000-of-001.jsonl", rows)
    loaded = summ.load_rows(tmp_path, cfg)
    per_edit = summ.per_edit(loaded, cfg)
    assert len(per_edit) == len(cfg.settings) * 2 * len(cfg.fpr_targets) * (1 + len(EDIT_IDS))
    cats = summ.categories(per_edit)
    group = next(c for c in cats if (c["N"], c["detector"], c["fpr_target"], c["group"]) ==
                 (8, "boundary", 0.2, "geometric"))
    members = [r for r in per_edit if (r["N"], r["detector"], r["fpr_target"]) == (8, "boundary", 0.2)
               and r["edit_id"] in EDIT_GROUPS["geometric"]]
    assert group["edit_count"] == 6 and group["tpr"] == pytest.approx(np.mean([r["tpr"] for r in members]))
    results = summ.edit_results(per_edit, cfg, 0.2)
    assert {r["method"] for r in results} == {"Ours (N=8, M=4)", "Ours (N=16, M=8)"}
    assert sum(r["group"] == "unedited" for r in results) == 2 and len(results) == 2 * 21
    with pytest.raises(ValueError):
        corpus.write_jsonl_rows(tmp_path / "replicates-shard-000-of-001.jsonl", rows[:-5])
        summ.load_rows(tmp_path, cfg)


def test_embedding_edits_renders_each_edit_of_each_source(tmp_path):
    class Fake:
        def embed(self, images):
            return helpers.unit(np.stack([np.asarray(im, dtype=float).reshape(-1)[:6] + 1.0 for im in images]))

    Image.new("RGB", (16, 16), (90, 120, 200)).save(tmp_path / "a.png")
    out = embed_ordinary_edits.embed_sources(Fake(), {"s": tmp_path / "a.png"}, ["s"])
    assert out.shape == (1, len(EDIT_IDS), 6)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0)


def test_edit_embeddings_load_merged_shards(tmp_path):
    ids = ["u1:clean", "u1:candidate:01", "u2:clean"]
    emb = helpers.unit(np.random.default_rng(0).normal(size=(3, len(EDIT_IDS), 4)))
    for index, part in enumerate(([0, 1], [2])):
        np.savez_compressed(str(tmp_path / ("edits-shard-%03d-of-002.npz" % index)),
                            source_ids=np.asarray([ids[i] for i in part]), edit_ids=np.asarray(EDIT_IDS),
                            embeddings=emb[part])
    loaded = _lib.load_edit_embeddings(tmp_path)
    assert np.array_equal(loaded.gather(["u2:clean", "u1:clean"]), emb[[2, 0]])
    (tmp_path / "edits-shard-001-of-002.npz").unlink()
    with pytest.raises(ValueError):
        _lib.load_edit_embeddings(tmp_path)
    with pytest.raises(ValueError):
        _lib.EditEmbeddings(ids, EDIT_IDS[:-1], emb[:, :-1])


def _results(method, tpr, retention, fpr=0.2, cosine=None):
    rows = [{"method": method, "edit_id": "identity", "group": "unedited", "edited_tpr": 0.9, "edited_fpr": 0.2,
             "retention": None, "cosine": 1.0}]
    for i, edit_id in enumerate(EDIT_IDS):
        group = next(g for g, members in EDIT_GROUPS.items() if edit_id in members)
        rows.append({"method": method, "edit_id": edit_id, "group": group, "edited_tpr": tpr - 0.01 * i,
                     "edited_fpr": fpr, "retention": retention - 0.02 * i,
                     "cosine": (cosine if cosine is not None else 0.99) - 0.02 * i})
    return rows


def _baseline_per_edit(path, fpr=0.2):
    rows = []
    for target in (0.05, fpr):
        for i, edit_id in enumerate(EDIT_IDS):
            rows.append({"requested_fpr": target, "threshold": 0.5, "calibration_fpr": 0.19, "edit_id": edit_id,
                         "category": next(g for g, m in EDIT_GROUPS.items() if edit_id in m), "unedited_tpr": 1.0,
                         "edited_tpr": 1.0 - 0.01 * i, "edited_fpr": 0.2, "retention": "nan" if i == 3 else 1.0})
    corpus.write_csv_rows(path, rows)


def test_baseline_and_generative_adapters(tmp_path):
    _baseline_per_edit(tmp_path / "per_edit.csv")
    rows = render.read_baseline_per_edit("Tree-Ring", tmp_path / "per_edit.csv", 0.2)
    assert len(rows) == 21 and rows[0]["group"] == "unedited" and rows[0]["edited_tpr"] == 1.0
    assert rows[0]["edited_fpr"] == 0.19 and rows[4]["retention"] is None
    assert {r["group"] for r in rows[1:]} == set(EDIT_GROUPS)
    with pytest.raises(ValueError):
        render.read_baseline_per_edit("Tree-Ring", tmp_path / "per_edit.csv", 0.1)
    corpus.write_csv_rows(tmp_path / "summary.csv", [
        {"method": method, "fpr_target": 0.2, "category": category, "edits": 20, "edited_tpr": 0.5,
         "retention": 0.8, "edited_fpr": 0.2, "original_tpr": 0.7, "calibration_fpr": 0.19}
        for method in ("ours_n8_m4", "tree_ring") for category in ("attribute_edit", "cat_attack", "all")])
    generative = render.read_generative_summary(tmp_path / "summary.csv", 0.2)
    assert {r["method"] for r in generative} == {"Ours (N=8, M=4)", "Tree-Ring"}
    assert {r["group"] for r in generative} == {"generative", "generative_all"}
    assert next(r for r in generative if r["edit_id"] == "cat_attack")["label"] == "Cat insertion"
    ours = _results("Ours (N=8, M=4)", 0.5, 0.9)
    pooled = render.category_values(ours + generative, "Ours (N=8, M=4)", "generative")
    assert pooled["before"] == pytest.approx(0.7) and pooled["tpr"] == pytest.approx(0.5)
    assert render.display_name("ours_n32_m12") == "Ours (N=32, M=12)" and render.display_name("seal") == "SEAL"


def test_tables_and_figure(tmp_path):
    rows = (_results("Ours (N=8, M=4)", 0.5, 0.9) + _results("Ours (N=16, M=8)", 0.6, 0.8)
            + _results("Tree-Ring", 1.0, 1.0))
    rows += [{"method": m, "edit_id": "attribute_change", "group": "generative", "label": "Attribute change",
              "edited_tpr": t, "edited_fpr": 0.2, "retention": r, "unedited_tpr": 0.7}
             for m, t, r in (("Ours (N=8, M=4)", 0.4, 0.8), ("Ours (N=16, M=8)", 0.6, 0.9), ("Tree-Ring", 0.5, 0.5))]
    table = render.categories_table(rows, 0.2)
    assert "Ours $(N{=}8,M{=}4)$ &" in table and "Tree-Ring &" in table and "Generative edits" in table
    values = render.category_values(rows, "Tree-Ring", "geometric")
    assert values["tpr"] == pytest.approx(np.mean([r["edited_tpr"] for r in rows if r["method"] == "Tree-Ring"
                                                   and r["group"] == "geometric"]))
    assert render.category_values(rows, "Tree-Ring", "generative")["before"] == pytest.approx(0.7)
    assert render.rank_colors([0.5, 0.5, 0.4, 0.3]) == ["best", "best", "second", "plain"]
    assert render.rank_colors([1.0, 0.9996, float("nan")]) == ["best", "best", "plain"]
    per_edit = render.per_edit_tables(rows, 0.2)
    assert "JPEG Q90" in per_edit and "Attribute change" in per_edit and "metricbest" in per_edit
    assert per_edit.count("\\begin{tabular}") == 2
    render.plot_categories(rows, tmp_path / "categories.pdf")
    assert (tmp_path / "categories.pdf").stat().st_size > 1000


def test_detector_ablation_table_compares_the_two_detectors():
    per_edit = []
    for n, m in ((8, 4), (16, 8)):
        for detector, tpr in (("exact", 0.4), ("boundary", 0.5)):
            for edit_id in ("identity",) + tuple(EDIT_IDS):
                per_edit.append({"N": n, "M": m, "detector": detector, "fpr_target": "0.2", "edit_id": edit_id,
                                 "tpr": tpr, "fpr": 0.18, "retention": "0.8"})
    table = render.ablation_table(per_edit, 0.2)
    assert "Exact code & 40.00 & 80.00 & 18.00" in table and "Boundary aware & 50.00" in table
    assert table.count("addlinespace") == 1


def test_similarity_points_correlations_and_figure(tmp_path):
    ours = _results("Ours (N=8, M=4)", 0.5, 0.9)
    inverted = _results("Tree-Ring", 1.0, 1.0, cosine=0.99)
    for row in inverted:
        if row["retention"] is not None:
            row["retention"] = 0.2 + 0.03 * EDIT_IDS.index(row["edit_id"]) if row["edit_id"] in EDIT_IDS else None
    rows = ours + inverted
    cosines = {("Tree-Ring", e): 0.9 - 0.01 * i for i, e in enumerate(EDIT_IDS)}
    points = similarity.edit_points(rows, cosines)
    assert len(points["Ours (N=8, M=4)"]) == 20 and all(p["family"] for p in points["Tree-Ring"])
    table = {r["method"]: r["spearman_rho"] for r in similarity.correlations(points)}
    assert table["Ours (N=8, M=4)"] == pytest.approx(1.0)  # retention and cosine are both decreasing
    assert table["Tree-Ring"] == pytest.approx(-1.0)
    similarity.plot(points, table, tmp_path / "similarity.pdf")
    assert (tmp_path / "similarity.pdf").stat().st_size > 1000


def test_pair_cosines_average_per_edit_over_watermarked_images():
    original = [{"image_id": "u1|watermarked|identity", "path": "a1"}, {"image_id": "u1|clean|identity", "path": "c1"},
                {"image_id": "u2|watermarked|identity", "path": "a2"}]
    edited = [{"image_id": "u1|watermarked|crop_50", "path": "e1"}, {"image_id": "u1|clean|crop_50", "path": "x"},
              {"image_id": "u2|watermarked|crop_50", "path": "e2"}]
    pairs = edit_cosines.watermarked_pairs(original, edited)
    assert pairs == [("u1", "crop_50", "a1", "e1"), ("u2", "crop_50", "a2", "e2")]
    a = helpers.unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    b = helpers.unit(np.array([[1.0, 0.0], [1.0, 0.0]]))
    out = edit_cosines.mean_cosines("A", [p[1] for p in pairs], a, b)
    assert out == [{"method": "A", "edit_id": "crop_50", "cosine": pytest.approx(0.5), "pair_count": 2}]


@pytest.fixture
def summary_with_two_fprs(tmp_path):
    cfg = helpers.small_config()
    setting = cfg.settings[0]
    rows = [
        {"N": setting.n, "M": setting.m, "detector": detector, "fpr_target": fpr, "edit_id": edit,
         "tpr": tpr, "fpr": fpr, "retention": retention, "cosine": cosine}
        for fpr, tpr, retentions in ((0.05, 0.1, (0.3, 0.7)), (0.20, 0.9, (0.7, 0.3)))
        for detector in ("boundary", "exact")
        for edit, retention, cosine in (("identity", None, 1.0),
                                        ("jpeg_q90", retentions[0], 0.99),
                                        ("jpeg_q75", retentions[1], 0.90))
    ]
    corpus.write_csv_rows(tmp_path / "per_edit.csv", rows)
    corpus.write_csv_rows(tmp_path / "edit_results.csv", summ.edit_results(rows, cfg, 0.20))
    return tmp_path


def test_render_robustness_uses_requested_fpr(summary_with_two_fprs, monkeypatch):
    root = summary_with_two_fprs
    output = root / "tables"
    monkeypatch.setattr(sys, "argv", ["render_robustness", "--summary-dir", str(root),
                                    "--fpr", "0.05", "--output-dir", str(output)])
    assert render.main() == 0
    table = (output / "experiment3_categories.tex").read_text()
    assert r"5\%" in table
    assert "10.0 & 50.0 & 5.0" in table


@pytest.mark.parametrize("filename", ["per_edit.csv", "edit_results.csv"])
def test_retention_plot_uses_requested_fpr(summary_with_two_fprs, monkeypatch, filename):
    root = summary_with_two_fprs
    output = root / "figures"
    monkeypatch.setattr(sys, "argv", ["plot_retention_cosine", "--results", str(root / filename),
                                    "--fpr", "0.05", "--output-dir", str(output)])
    assert similarity.main() == 0
    with (output / "retention_cosine_correlations.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["spearman_rho"]) == pytest.approx(-1.0)


def test_ours_per_edit_rejects_missing_fpr(summary_with_two_fprs):
    rows = render.read_ours_per_edit(summary_with_two_fprs / "per_edit.csv")
    with pytest.raises(ValueError, match="no boundary-aware results at FPR target 0.01"):
        render.ours_results_at_fpr(rows, helpers.small_config(), 0.01)
