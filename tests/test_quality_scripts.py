"""CPU runs of the quality scripts on synthetic caches and activations."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

QUALITY = Path(__file__).resolve().parents[1] / "experiments" / "quality"
sys.path.insert(0, str(QUALITY))

import _quality_lib as lib  # noqa: E402
from detection_corpus import EmbeddingCache, write_cache  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402
from watermark.io import read_jsonl, write_jsonl  # noqa: E402

KEY = bytes(range(32))


def _load(name):
    spec = importlib.util.spec_from_file_location("quality_script_" + name, str(QUALITY / (name + ".py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(name, argv, monkeypatch):
    module = _load(name)
    monkeypatch.setattr(sys, "argv", [name] + [str(item) for item in argv])
    return module.main()


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _cache(prompts=6, slots=2, n=32, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(prompts, 1, 1, dim))
    candidates = _unit(base + 0.7 * rng.normal(size=(prompts, slots, n, dim)))
    cleans = _unit(base[:, :, 0] + 0.7 * rng.normal(size=(prompts, slots, dim)))
    uids = tuple("p%02d" % i for i in range(prompts))
    return EmbeddingCache(uids, candidates, cleans, {"encoder": "ours", "panel_sha256": "x"})


def _write(tmp_path, cache, name="cache.npz"):
    write_cache(tmp_path / name, cache.prompt_uids, cache.candidate_vectors, cache.clean_vectors,
                {"encoder": "ours", "panel_sha256": "x"})
    return tmp_path / name


def _key_file(tmp_path):
    (tmp_path / "master.key").write_bytes(KEY)
    return tmp_path / "master.key"


def test_select_populations_writes_every_configuration(tmp_path, monkeypatch):
    cfg = load_detection_config()
    path = _write(tmp_path, _cache())
    _run("select_populations", ["--cache", path, "--key-file", _key_file(tmp_path), "--num-keys", 2,
                                "--output", tmp_path / "selections.npz"], monkeypatch)
    uids, arrays, metadata = lib.load_selections(tmp_path / "selections.npz")
    assert uids == ["p%02d" % i for i in range(6)] and metadata["num_keys"] == 2
    for setting in cfg.settings:
        marked = arrays[lib.selection_key(setting.n, setting.m, "watermarked")]
        control = arrays[lib.selection_key(setting.n, setting.m, "random")]
        assert marked.shape == (cfg.frames, 2, 6, 2) and control.shape == (2, 6, 2)
        assert marked.min() >= 1 and marked.max() <= setting.n and control.max() <= setting.n
    from detection_corpus import frame_planes

    expected = lib.select_positions(_cache().candidate_vectors, frame_planes(cfg, 16)["frame-00"], KEY,
                                    frame_id="frame-00", key_trial=0, n=32, m=12,
                                    prompt_uids=["p%02d" % i for i in range(6)])
    assert np.array_equal(arrays[lib.selection_key(32, 12, "watermarked")][0, 0], expected)
    assert np.array_equal(arrays[lib.selection_key(8, 4, "random")][1], lib.matched_random_positions(uids, 1, 8, 2))


def _selection_arrays(prompts, slots, keys, frames, configurations, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    for n, m in configurations:
        arrays[lib.selection_key(n, m, "watermarked")] = rng.integers(1, n + 1, size=(frames, keys, prompts, slots))
        arrays[lib.selection_key(n, m, "random")] = rng.integers(1, n + 1, size=(keys, prompts, slots))
    return arrays


def test_fid10k_pairs_and_summary(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    prompts, slots, n, dim = 8, 3, 8, 6
    uids = ["p%d" % i for i in range(prompts)]
    arrays = _selection_arrays(prompts, slots, 3, 2, [(8, 4)])
    metadata = {"frame_ids": ["frame-00", "frame-01"], "configurations": [[8, 4]], "num_keys": 3, "slots": slots}
    lib.write_selections(tmp_path / "sel.npz", uids, arrays, metadata)
    lib.write_shard(tmp_path / "ours" / "activations-shard-000-of-001.npz", uids,
                    candidates=rng.normal(size=(prompts, slots, n, dim)), cleans=rng.normal(size=(prompts, slots, dim)))
    lib.write_shard(tmp_path / "tr" / "activations-shard-000-of-001.npz", uids,
                    watermarked=rng.normal(size=(prompts, slots, dim)), cleans=rng.normal(size=(prompts, slots, dim)))
    _run("fid10k", ["--selections", tmp_path / "sel.npz", "--ours-activations", tmp_path / "ours",
                    "--native-activations", "tree_ring=%s" % (tmp_path / "tr"), "--output-dir", tmp_path / "out"],
         monkeypatch)
    pairs = list(read_csv(tmp_path / "out" / "pairs.csv"))
    ours = [row for row in pairs if row["family"] == "ours"]
    assert len(ours) == 3 * 2 and {row["frame"] for row in ours} == {"frame-00", "frame-01"}
    summary = {row["family"]: row for row in read_csv(tmp_path / "out" / "summary.csv")}
    by_key = {}
    for row in ours:
        by_key.setdefault(row["key"], []).append(float(row["fid"]))
    expected = np.mean([np.mean(values) for values in by_key.values()])
    assert float(summary["ours"]["fid_mean"]) == pytest.approx(expected)
    assert summary["ours"]["fid_count"] == "3" and float(summary["tree_ring"]["fid_sd"]) == 0.0


def read_csv(path):
    import csv

    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_fid5k_rotating_keys_and_matched_changes(tmp_path):
    fid5k = _load("fid5k")
    cache = _cache(prompts=7, slots=1, n=32)
    uids = list(cache.prompt_uids)
    from detection_corpus import frame_planes

    cfg = load_detection_config()
    planes = frame_planes(cfg, 16)["frame-00"]
    keys = lib.trial_keys(KEY, 3)
    positions = fid5k.rotating_key_positions(cache.candidate_vectors, planes, keys, frame_id="frame-00", n=8, m=4,
                                             prompt_uids=uids)
    assert positions.shape == (7, 1) and positions.max() <= 8
    for row in range(7):
        single = lib.select_positions(cache.candidate_vectors[row:row + 1], planes, keys[row % 3],
                                      frame_id="frame-00", key_trial=row % 3, n=8, m=4, prompt_uids=[uids[row]])
        assert positions[row, 0] == single[0, 0]
    rows = fid5k.matched_changes({"a": 11.0, "a_clean": 10.0, "b": 9.0}, {"a": "a_clean", "b": "a_clean"})
    assert rows[0]["relative_change_percent"] == pytest.approx(10.0)
    assert rows[1]["relative_change_percent"] == pytest.approx(-10.0)


def test_fid5k_main_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cache = _cache(prompts=12, slots=1, n=32)
    uids = list(cache.prompt_uids)
    dim = 5
    lib.write_shard(tmp_path / "ours" / "activations-shard-000-of-001.npz", uids,
                    candidates=rng.normal(size=(12, 1, 32, dim)), cleans=rng.normal(size=(12, 1, dim)))
    lib.write_shard(tmp_path / "real" / "activations-shard-000-of-001.npz", uids, images=rng.normal(size=(12, dim)))
    lib.write_shard(tmp_path / "seal" / "activations-shard-000-of-001.npz", uids,
                    watermarked=rng.normal(size=(12, 1, dim)), cleans=rng.normal(size=(12, 1, dim)))
    _run("fid5k", ["--cache", _write(tmp_path, cache), "--key-file", _key_file(tmp_path), "--num-keys", 4,
                   "--ours-activations", tmp_path / "ours", "--real-activations", tmp_path / "real",
                   "--native-activations", "seal=%s" % (tmp_path / "seal"), "--output-dir", tmp_path / "out"],
         monkeypatch)
    populations = [row["population"] for row in read_csv(tmp_path / "out" / "fid.csv")]
    assert populations == ["ours_clean", "ours_n8_m4", "ours_n16_m8", "ours_n32_m12", "seal", "seal_clean"]
    changes = read_csv(tmp_path / "out" / "matched_changes.csv")
    assert [row["clean_population"] for row in changes] == ["ours_clean"] * 3 + ["seal_clean"]


def test_vendi_cells_and_summary():
    vendi = _load("vendi")
    cache = _cache(prompts=3, slots=6, n=32)
    evaluation = _unit(np.random.default_rng(5).normal(size=(3, 6, 32, 10)))
    from detection_corpus import frame_planes

    cfg = load_detection_config()
    planes = frame_planes(cfg, 16)
    keys = lib.trial_keys(KEY, 2)
    uids = list(cache.prompt_uids)
    rows = vendi.ours_cells(cache.candidate_vectors, evaluation, planes, keys, n=8, m=4, frame_id="frame-00",
                            prompt_uids=uids)
    assert len(rows) == 3 * 2
    controls = {row["prompt_uid"]: row["control_vendi"] for row in rows}
    assert len(controls) == 3 and all(1.0 <= value <= 6.0 + 1e-9 for value in controls.values())
    assert all(row["relative_vendi"] == pytest.approx(row["vendi"] / row["control_vendi"]) for row in rows)
    summary = vendi.summarize(rows)
    assert summary["cells"] == 6
    assert summary["change_percent"] == pytest.approx(100 * (summary["mean_relative_vendi"] - 1))
    identical = vendi.slot_vendi(np.ones((4, 3, 5)), np.array([1, 2, 3, 1]))
    assert identical == pytest.approx(1.0)


def test_clipscore_aggregation():
    clipscore = _load("clipscore")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(5, 4, 7))
    text = rng.normal(size=(5, 7))
    positions = np.array([1, 2, 3, 4, 1])
    expected = np.mean([features[i, positions[i] - 1] @ text[i]
                        / np.linalg.norm(features[i, positions[i] - 1]) / np.linalg.norm(text[i]) for i in range(5)])
    assert clipscore.candidate_population_score(features, text, positions) == pytest.approx(expected)
    selections = {lib.selection_key(4, 2, "watermarked"): rng.integers(1, 5, size=(2, 3, 5, 1)),
                  lib.selection_key(4, 2, "random"): rng.integers(1, 5, size=(3, 5, 1))}
    row = clipscore.summarize_ours(features, text, selections, 4, 2)
    assert row["watermarked_populations"] == 6 and row["clean_populations"] == 3
    assert row["relative_change_percent"] == pytest.approx(clipscore.relative_change(row["watermarked_mean"],
                                                                                    row["clean_mean"]))
    with pytest.raises(ValueError):
        clipscore.cosine_similarity_rows(np.zeros((2, 3)), np.ones((2, 3)))


def test_fid5k_panel_script(tmp_path, monkeypatch):
    captions = {"images": [{"id": i, "file_name": "%012d.jpg" % i} for i in (3, 1, 2)],
                "annotations": [{"id": 10 * i + k, "image_id": i, "caption": "caption %d %d" % (i, k)}
                                for i in (1, 2, 3) for k in range(3)]}
    (tmp_path / "captions.json").write_text(json.dumps(captions))
    images = tmp_path / "val"
    images.mkdir()
    for i in (1, 2):
        (images / ("%012d.jpg" % i)).write_bytes(b"x")
    with pytest.raises(SystemExit):
        _run("fid5k_panel", ["--captions", tmp_path / "captions.json", "--images-dir", images,
                             "--output", tmp_path / "prompts.jsonl"], monkeypatch)
    (images / ("%012d.jpg" % 3)).write_bytes(b"x")
    _run("fid5k_panel", ["--captions", tmp_path / "captions.json", "--images-dir", images,
                         "--output", tmp_path / "prompts.jsonl"], monkeypatch)
    rows = read_jsonl(tmp_path / "prompts.jsonl")
    assert [row["image_id"] for row in rows] == [1, 2, 3] and all(row["text"].startswith("caption") for row in rows)


def test_inception_script_requires_a_source_directory(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "prompts.jsonl", [{"prompt_uid": "p0", "text": "x", "file_name": "a.jpg"}])
    module = _load("inception_activations")
    monkeypatch.setattr(module, "load_inception", lambda device: object())
    monkeypatch.setattr(sys, "argv", ["x", "--source", "real", "--prompts", str(tmp_path / "prompts.jsonl"),
                                      "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        module.main()


def test_inception_script_assembles_shards_from_activations(tmp_path, monkeypatch):
    import json as _json

    root = tmp_path / "corpus" / "p0"
    for slot in (1, 2):
        (root / ("slot_%03d" % slot)).mkdir(parents=True)
    panel = {"prompt_uid": "p0", "mode": "candidates", "batch_slots": [
        {"slot": s, "candidate_images": [{"candidate_position": c, "relative_path": "slot_%03d/c%d.png" % (s, c)}
                                         for c in (1, 2, 3)],
         "clean": {"relative_path": "slot_%03d/clean.png" % s}} for s in (1, 2)]}
    (root / "panel.json").write_text(_json.dumps(panel))
    write_jsonl(tmp_path / "prompts.jsonl", [{"prompt_uid": "p0", "text": "x"}])
    module = _load("inception_activations")
    monkeypatch.setattr(module, "load_inception", lambda device: object())
    seen = []

    def fake(paths, model, **kwargs):
        seen.extend(paths)
        return np.arange(len(paths) * 2, dtype=np.float32).reshape(len(paths), 2)

    monkeypatch.setattr(module, "activations", fake)
    monkeypatch.setattr(sys, "argv", ["x", "--source", "corpus", "--prompts", str(tmp_path / "prompts.jsonl"),
                                      "--corpus-dir", str(tmp_path / "corpus"), "--output-dir", str(tmp_path / "out")])
    assert module.main() == 0
    assert len(seen) == 2 * (3 + 1)
    loaded = lib.load_shards(tmp_path / "out", ["candidates", "cleans"], ["p0"])
    assert loaded["candidates"].shape == (1, 2, 3, 2) and loaded["cleans"].shape == (1, 2, 2)
    # all candidates (slot by slot) are listed first, then the clean images; row r holds (2r, 2r + 1)
    assert loaded["candidates"][0, 0, 1, 0] == 2 and loaded["candidates"][0, 1, 0, 0] == 6
    assert loaded["cleans"][0, 0, 0] == 12 and loaded["cleans"][0, 1, 0] == 14


def test_subset_prompts_takes_the_first_identities(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "prompts.jsonl", [{"prompt_uid": "b", "text": "2"}, {"prompt_uid": "a", "text": "1"},
                                             {"prompt_uid": "c", "text": "3"}])
    _run("subset_prompts", ["--prompts", tmp_path / "prompts.jsonl", "--count", 2, "--output", tmp_path / "out.jsonl"],
         monkeypatch)
    assert [row["prompt_uid"] for row in read_jsonl(tmp_path / "out.jsonl")] == ["a", "b"]
    with pytest.raises(SystemExit):
        _run("subset_prompts", ["--prompts", tmp_path / "prompts.jsonl", "--count", 4,
                                "--output", tmp_path / "x.jsonl"], monkeypatch)


def test_embed_native_stacks_both_roles(tmp_path, monkeypatch):
    import json as _json

    for uid in ("p0", "p1"):
        root = tmp_path / "gen" / uid
        root.mkdir(parents=True)
        slots = [{"slot": s, "watermarked": {"relative_path": "slot_%03d/watermarked.png" % s},
                  "clean": {"relative_path": "slot_%03d/clean.png" % s}} for s in (1, 2, 3)]
        (root / "panel.json").write_text(_json.dumps({"prompt_uid": uid, "slots": slots}))
    write_jsonl(tmp_path / "prompts.jsonl", [{"prompt_uid": uid, "text": "x"} for uid in ("p0", "p1")])
    module = _load("embed_native")
    seen = []
    monkeypatch.setattr(module, "load_encoder", lambda name, device: name)
    monkeypatch.setattr(module, "embed_paths", lambda encoder, paths: (seen.append(list(paths)) or
                                                                       np.arange(len(paths) * 4, dtype=float).reshape(len(paths), 4)))
    monkeypatch.setattr(sys, "argv", ["x", "--native-dir", str(tmp_path / "gen"), "--prompts",
                                      str(tmp_path / "prompts.jsonl"), "--output-dir", str(tmp_path / "out")])
    assert module.main() == 0
    assert len(seen) == 2 and len(seen[0]) == 6 and seen[0][0].name == "watermarked.png"
    loaded = lib.load_shards(tmp_path / "out", ["watermarked", "cleans"], ["p0", "p1"], prefix="embeddings")
    assert loaded["watermarked"].shape == (2, 3, 4) and loaded["cleans"].shape == (2, 3, 4)
    assert loaded["cleans"][0, 0, 0] == 12
