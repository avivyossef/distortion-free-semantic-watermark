"""Style / depth encoder study (P16): panel, edits, scoring under edits."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "style_depth_encoders"
sys.path.insert(0, str(EXPERIMENT))

import style_depth_lib as lib  # noqa: E402
from watermark.coding import sample_hyperplanes  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402

KEY = bytes(range(32))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location("style_depth_" + name, EXPERIMENT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unit(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _setting():
    config = load_detection_config()
    exception = config.exception("style_depth")
    return config, exception


def _scores(original, edited, *, prompts=50, **overrides):
    config, setting = _setting()
    planes = sample_hyperplanes(config.frame_seeds()[0], original.shape[-1])
    arguments = dict(frame_id=config.frame_ids()[0], key_trial=0, n=setting["N"], m=setting["M"],
                     activation=setting["activation"], radius=setting["radius"], folds=config.folds,
                     fpr=config.fpr_targets[-1])
    arguments.update(overrides)
    return lib.score_replicate(original, edited, ["u%03d" % i for i in range(prompts)], planes, KEY, **arguments)


def _embeddings(prompts=50, dimension=24, seed=0):
    return unit(np.random.default_rng(seed).normal(size=(prompts, 17, dimension)))


def test_the_study_uses_the_configured_style_depth_setting():
    config, setting = _setting()
    assert (setting["N"], setting["M"], setting["radius"]) == (16, 8, 1)
    assert setting["calibration_population"] == "evaluation"
    assert lib.SLOT_CANDIDATES == setting["N"]
    assert config.fpr_targets[-1] == 0.20


def test_an_identity_edit_changes_nothing():
    original = _embeddings()
    result = _scores(original, {"same": original.copy()})
    assert result["same"]["tpr"] == result["original"]["tpr"]
    assert result["same"]["clean_fpr"] == result["original"]["clean_fpr"]
    assert result["original"]["tpr"] > 0.0 and result["same"]["retention"] == 1.0


def test_an_unrelated_edit_drops_retention_below_one_and_tpr_toward_the_clean_rate():
    original = _embeddings(seed=1)
    scrambled = _embeddings(seed=2)
    result = _scores(original, {"scrambled": scrambled})
    assert result["scrambled"]["retention"] < 1.0
    assert result["scrambled"]["tpr"] < result["original"]["tpr"]


def test_threshold_is_calibrated_on_the_evaluated_clean_images():
    original = _embeddings(seed=3)
    result = _scores(original, {})
    assert result["original"]["clean_fpr"] <= load_detection_config().fpr_targets[-1]


def test_cosine_distance_is_zero_for_identical_and_one_for_orthogonal_embeddings():
    original = _embeddings(dimension=64, seed=4)
    assert lib.cosine_distance(original, original) == pytest.approx(0.0, abs=1e-12)
    basis = np.broadcast_to(np.eye(4)[None, :2], (3, 2, 4))
    assert lib.cosine_distance(basis, np.roll(basis, 1, axis=-1)) == pytest.approx(1.0)


def test_hflip_mirrors_and_other_spatial_edits_use_the_shared_registry():
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, size=(40, 40, 3), dtype=np.uint8))
    flipped = lib.spatial_edit(image, "hflip")
    assert np.array_equal(np.asarray(flipped), np.asarray(image)[:, ::-1])
    for edit in ("translate_32", "offcenter_crop_75"):
        assert lib.spatial_edit(image, edit).size == image.size
    with pytest.raises(ValueError):
        lib.spatial_edit(image, "rotate_3")


def test_edit_registry_matches_the_paper():
    assert list(lib.STYLE_EDITS.values()) == ["Make it an oil painting", "Turn it into a watercolor painting",
                                              "Turn it into a pencil sketch"]
    assert len(lib.EDITS) == 6 and lib.ENCODERS == ("csd_style", "depth_layout")
    assert lib.edit_seed("u", "ip2p_watercolor", 3) == lib.edit_seed("u", "ip2p_watercolor", 3)
    assert lib.edit_seed("u", "ip2p_watercolor", 3) != lib.edit_seed("u", "ip2p_watercolor", 4)


def _write_panel(root: Path, uid: str) -> None:
    folder = root / uid
    (folder / "slot_001").mkdir(parents=True)
    candidates = []
    for position in range(1, 17):
        name = "slot_001/candidate_%02d.png" % position
        Image.new("RGB", (16, 16), (position, 0, 0)).save(folder / name)
        candidates.append({"candidate_position": position, "relative_path": name})
    Image.new("RGB", (16, 16), (200, 0, 0)).save(folder / "slot_001" / "clean.png")
    (folder / "panel.json").write_text(json.dumps({"batch_slots": [
        {"slot": 1, "candidate_images": candidates[::-1], "clean": {"relative_path": "slot_001/clean.png"}}]}))


def test_source_paths_are_sixteen_candidates_then_the_clean_image(tmp_path):
    _write_panel(tmp_path, "u1")
    paths = lib.source_paths(tmp_path, "u1")
    assert len(paths) == 17
    assert [int(np.asarray(Image.open(path))[0, 0, 0]) for path in paths] == list(range(1, 17)) + [200]


def test_panel_prompt_uids_are_the_first_sorted_identities(tmp_path, monkeypatch):
    for uid in ("c", "a", "b"):
        _write_panel(tmp_path, uid)
    with pytest.raises(ValueError):
        lib.panel_prompt_uids(tmp_path)
    monkeypatch.setattr(lib, "PROMPT_COUNT", 2)
    assert lib.panel_prompt_uids(tmp_path) == ["a", "b"]


class FakeEncoder:
    def __init__(self, dimension):
        self.dimension = dimension

    def embed(self, images):
        values = [np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float64).ravel() + 1.0 for image in images]
        return unit(np.stack(values))


def test_embed_edits_then_score_edits_end_to_end(tmp_path, monkeypatch):
    embed = load_script("embed_edits")
    score = load_script("score_edits")
    monkeypatch.setattr(lib, "PROMPT_COUNT", 3)
    uids = ["a", "b", "c"]
    for uid in uids:
        _write_panel(tmp_path / "panel", uid)
        for edit in lib.STYLE_EDITS:
            folder = tmp_path / "edits" / edit / uid
            folder.mkdir(parents=True)
            records = []
            for index in range(17):
                path = folder / ("%02d.png" % index)
                Image.new("RGB", (16, 16), (index * 9 % 256, 30, 60)).save(path)
                records.append({"output": path.name, "output_sha256": embed.sha256_file(path)})
            (folder / "manifest.json").write_text(json.dumps({"records": records}))
    monkeypatch.setattr(embed, "load_encoder", lambda name, device: FakeEncoder(48))
    monkeypatch.setattr(sys, "argv", ["embed_edits.py", "--panel-root", str(tmp_path / "panel"),
                                      "--edit-dir", str(tmp_path / "edits"), "--output", str(tmp_path / "emb.npz"),
                                      "--device", "cpu"])
    assert embed.main() == 0
    with np.load(tmp_path / "emb.npz") as data:
        assert data["csd_style__original"].shape == (3, 17, 48)
        assert set(data.files) == {"prompt_uids"} | {"%s__%s" % (e, n) for e in lib.ENCODERS
                                                      for n in ("original",) + lib.EDITS}

    # Scoring needs more prompts than three: synthesize a larger embedding file.
    rng = np.random.default_rng(9)
    arrays = {}
    for encoder in lib.ENCODERS:
        base = unit(rng.normal(size=(50, 17, 24)))
        arrays["%s__original" % encoder] = base
        for edit in lib.EDITS:
            arrays["%s__%s" % (encoder, edit)] = unit(base + 0.3 * rng.normal(size=base.shape))
    np.savez(tmp_path / "big.npz", prompt_uids=np.array(["u%03d" % i for i in range(50)]), **arrays)
    (tmp_path / "master.key").write_bytes(KEY)
    monkeypatch.setattr(sys, "argv", ["score_edits.py", "--embeddings", str(tmp_path / "big.npz"),
                                      "--key-file", str(tmp_path / "master.key"), "--output-dir", str(tmp_path / "out")])
    assert score.main() == 0
    config = load_detection_config()
    replicates = list(csv.DictReader((tmp_path / "out" / "replicates.csv").open()))
    assert len(replicates) == 2 * config.frames * config.keys * 7
    table = list(csv.DictReader((tmp_path / "out" / "table.csv").open()))
    assert len(table) == 2 * 7 and table[0]["transformation"] == "Original"
    assert all(row["retention_percent"] == "" for row in table if row["transformation"] == "Original")
    assert all(0.0 <= float(row["tpr_percent"]) <= 100.0 for row in table)


def test_edit_figure_renders(tmp_path, monkeypatch):
    figure = load_script("edit_figure")
    _write_panel(tmp_path / "panel", "u1")
    for edit in lib.STYLE_EDITS:
        folder = tmp_path / "edits" / edit / "u1"
        folder.mkdir(parents=True)
        Image.new("RGB", (16, 16), (10, 20, 30)).save(folder / "00.png")
    monkeypatch.setattr(sys, "argv", ["edit_figure.py", "--panel-root", str(tmp_path / "panel"),
                                      "--edit-dir", str(tmp_path / "edits"), "--prompt-uid", "u1",
                                      "--output", str(tmp_path / "figure.pdf")])
    assert figure.main() == 0
    assert (tmp_path / "figure.pdf").stat().st_size > 1000
