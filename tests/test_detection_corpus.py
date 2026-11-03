"""Generation experiment helpers: prompts, panels, embedding caches, selections, key files."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import detection_helpers as helpers
import detection_corpus as corpus
import embed_corpus
import generate_corpus  # noqa: F401  (import must stay cheap: heavy imports are inside functions)
import merge_embeddings
import new_key

from watermark.config import load_detection_config


def test_prompts_are_sorted_validated_and_dealt_round_robin(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        {"prompt_uid": "b", "text": "two"}, {"prompt_uid": "a", "text": "one"}, {"prompt_uid": "c", "text": "x"}]))
    rows = corpus.read_prompts(path)
    assert [r["prompt_uid"] for r in rows] == ["a", "b", "c"]
    assert [r["prompt_uid"] for r in corpus.deal(rows, 1, 2)] == ["b"]
    assert [r["prompt_uid"] for r in corpus.deal(rows, 0, 2)] == ["a", "c"]
    with pytest.raises(ValueError):
        corpus.deal(rows, 2, 2)
    path.write_text(json.dumps({"prompt_uid": "a", "text": " "}) + "\n")
    with pytest.raises(ValueError):
        corpus.read_prompts(path)
    path.write_text(json.dumps({"prompt_uid": "a", "text": "x"}) + "\n" + json.dumps({"prompt_uid": "a", "text": "y"}))
    with pytest.raises(ValueError):
        corpus.read_prompts(path)


def test_cache_roundtrip_subset_and_overwrite_guard(tmp_path):
    cache = helpers.synthetic_cache(prompts=7, slots=2, n=4, dim=8)
    path = tmp_path / "cache.npz"
    corpus.write_cache(path, cache.prompt_uids[::-1], cache.candidate_vectors[::-1], cache.clean_vectors[::-1],
                       {"encoder": "ours"})
    loaded = corpus.load_cache(path)
    assert loaded.prompt_uids == cache.prompt_uids  # rows are stored sorted by prompt identity
    assert np.allclose(loaded.candidate_vectors, cache.candidate_vectors)
    assert loaded.metadata["encoder"] == "ours"
    sub = loaded.subset(["p0003", "p0001"])
    assert sub.prompt_uids == ("p0003", "p0001")
    assert np.array_equal(sub.clean_vectors[0], loaded.clean_vectors[3])
    with pytest.raises(ValueError):
        corpus.write_cache(path, cache.prompt_uids, cache.candidate_vectors, cache.clean_vectors, {})
    with pytest.raises(ValueError):
        loaded.index_of("missing")
    with pytest.raises(ValueError):
        corpus.write_cache(tmp_path / "bad.npz", cache.prompt_uids, cache.candidate_vectors,
                           cache.clean_vectors[:, :1], {})


def _write_panel(root, uid, slots=2, n=3):
    directory = root / uid
    directory.mkdir(parents=True)
    batch = []
    for slot in range(1, slots + 1):
        batch.append({"slot": slot, "clean": {"relative_path": "slot_%03d/clean.png" % slot},
                      "candidate_images": [{"candidate_position": p, "relative_path": "slot_%03d/candidate_%02d.png"
                                            % (slot, p)} for p in range(n, 0, -1)]})
    (directory / "panel.json").write_text(json.dumps({"prompt_uid": uid, "mode": "candidates", "batch_slots": batch}))


def test_panel_images_and_manifest_hash(tmp_path):
    _write_panel(tmp_path, "p1")
    panel = corpus.read_panel(tmp_path, "p1")
    candidates, cleans = corpus.panel_images(tmp_path, panel)
    assert [p.name for p in candidates[0]] == ["candidate_01.png", "candidate_02.png", "candidate_03.png"]
    assert cleans[1].name == "clean.png" and cleans[1].parent.name == "slot_002"
    assert len(corpus.panel_images(tmp_path, panel, 1)[0]) == 1
    first = corpus.panel_manifest_sha256(tmp_path, ["p1"])
    (tmp_path / "p1" / "panel.json").write_text(json.dumps({"prompt_uid": "p1", "mode": "candidates",
                                                              "batch_slots": []}))
    assert corpus.panel_manifest_sha256(tmp_path, ["p1"]) != first
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "panel.json").write_text(json.dumps({"prompt_uid": "p1", "mode": "candidates"}))
    with pytest.raises(ValueError):
        corpus.read_panel(tmp_path, "other")


def test_replicate_grid_comes_from_the_detection_config():
    cfg = load_detection_config()
    cells = corpus.replicate_cells(cfg)
    assert len(cells) == cfg.frames * cfg.keys
    assert {cell[0] for cell in cells} == set(cfg.frame_ids())
    planes = corpus.frame_planes(cfg, 16)
    assert set(planes) == set(cfg.frame_ids()) and planes["frame-00"].shape == (64, 16)
    keys = corpus.trial_keys(helpers.KEY, cfg)
    assert keys[0] == helpers.KEY and len(set(keys.values())) == cfg.keys


class _FakeEncoder:
    def __init__(self, dim):
        self.dim = dim

    def embed(self, images):
        rng = np.random.default_rng(len(images))
        return helpers.unit(rng.normal(size=(len(images), self.dim)))


def test_embed_prompts_geometry_and_shard_merge(tmp_path, monkeypatch):
    from PIL import Image

    for uid in ("p1", "p2", "p3"):
        _write_panel(tmp_path / "corpus", uid, slots=2, n=3)
        for slot in (1, 2):
            (tmp_path / "corpus" / uid / ("slot_%03d" % slot)).mkdir()
            for name in ["clean.png"] + ["candidate_%02d.png" % p for p in (1, 2, 3)]:
                Image.new("RGB", (8, 8)).save(tmp_path / "corpus" / uid / ("slot_%03d" % slot) / name)
    candidates, cleans = embed_corpus.embed_prompts(_FakeEncoder(5), tmp_path / "corpus", ["p1", "p2"], None)
    assert candidates.shape == (2, 2, 3, 5) and cleans.shape == (2, 2, 5)
    candidates, cleans = embed_corpus.embed_prompts(_FakeEncoder(5), tmp_path / "corpus", ["p1"], 1)
    assert candidates.shape == (1, 1, 3, 5) and cleans.shape == (1, 1, 5)
    # two shards merge into one cache with all prompts
    shards = tmp_path / "shards"
    for index, uids in enumerate((["p3", "p1"], ["p2"])):
        c, k = embed_corpus.embed_prompts(_FakeEncoder(5), tmp_path / "corpus", uids, None)
        corpus.write_cache(shards / ("embeddings-shard-%03d-of-002.npz" % index), uids, c, k,
                           {"encoder": "ours", "panel_sha256": corpus.panel_manifest_sha256(tmp_path / "corpus", uids)})
    monkeypatch.setattr(sys, "argv", ["merge", "--shards-dir", str(shards), "--output", str(tmp_path / "all.npz")])
    assert merge_embeddings.main() == 0
    merged = corpus.load_cache(tmp_path / "all.npz")
    assert merged.prompt_uids == ("p1", "p2", "p3") and merged.candidate_vectors.shape == (3, 2, 3, 5)
    monkeypatch.setattr(sys, "argv", ["merge", "--shards-dir", str(shards), "--output", str(tmp_path / "again.npz")])
    (shards / "embeddings-shard-001-of-002.npz").unlink()
    with pytest.raises(SystemExit):
        merge_embeddings.main()


def test_key_file_is_32_random_bytes_and_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "keys" / "a.key"
    monkeypatch.setattr(sys, "argv", ["new_key", "--output", str(path)])
    assert new_key.main() == 0
    assert len(corpus.read_master_key(path)) == 32
    with pytest.raises(FileExistsError):
        new_key.main()
    path.write_bytes(b"short")
    with pytest.raises(ValueError):
        corpus.read_master_key(path)
