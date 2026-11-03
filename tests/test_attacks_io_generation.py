from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from watermark import attacks, generation, io

KEY = bytes(range(32))


def _image(size: int = 64, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8))


def test_exactly_twenty_edits_in_two_groups():
    assert len(attacks.EDIT_IDS) == 20 and len(set(attacks.EDIT_IDS)) == 20
    assert len(attacks.EDIT_GROUPS["geometric"]) == 6
    assert len(attacks.EDIT_GROUPS["appearance_compression"]) == 14
    assert set(attacks.EDIT_GROUPS["geometric"]) | set(attacks.EDIT_GROUPS["appearance_compression"]) == set(attacks.EDIT_IDS)


@pytest.mark.parametrize("edit_id", attacks.EDIT_IDS)
def test_every_edit_is_deterministic_and_keeps_geometry(edit_id):
    source = _image()
    first, second = attacks.render_edit(source, edit_id), attacks.render_edit(source, edit_id)
    assert first.size == source.size and first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(source))


def test_random_edits_depend_on_the_source_pixels():
    one, two = attacks.render_edit(_image(seed=0), "gaussian_noise_6"), attacks.render_edit(_image(seed=1), "gaussian_noise_6")
    noise_one = np.asarray(one, dtype=int) - np.asarray(_image(seed=0), dtype=int)
    noise_two = np.asarray(two, dtype=int) - np.asarray(_image(seed=1), dtype=int)
    assert not np.array_equal(noise_one, noise_two)


def test_translation_pads_with_black_and_shifts_content():
    source = Image.new("RGB", (40, 40), (200, 100, 50))
    shifted = np.asarray(attacks.render_edit(source, "translate_8"))
    assert (shifted[:8] == 0).all() and (shifted[:, :8] == 0).all()
    assert (shifted[8:, 8:] == np.array([200, 100, 50])).all()


def test_unknown_edit_is_rejected():
    with pytest.raises(ValueError):
        attacks.render_edit(_image(), "rotate_15")


def test_io_helpers_round_trip(tmp_path):
    rows = [{"b": 2, "a": 1}, {"a": 3}]
    io.write_jsonl(tmp_path / "x" / "rows.jsonl", rows)
    assert io.read_jsonl(tmp_path / "x" / "rows.jsonl") == rows
    assert (tmp_path / "x" / "rows.jsonl").read_text().splitlines()[0] == '{"a":1,"b":2}'
    assert io.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert io.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(ValueError):
        io.read_jsonl(tmp_path / "missing.jsonl")
    with pytest.raises(ValueError):
        io.read_key_file(path)
    path.write_bytes(KEY)
    assert io.read_key_file(path) == KEY


def test_pixel_hash_ignores_encoding_but_not_pixels():
    image = _image()
    assert io.pixel_sha256(image) == io.pixel_sha256(image.copy())
    changed = image.copy()
    changed.putpixel((0, 0), (1, 2, 3))
    assert io.pixel_sha256(image) != io.pixel_sha256(changed)


def _rows(**kwargs):
    base = dict(prompt_uid="p1", master_key=KEY, mode="candidates", batch_size=2, candidate_count=4)
    base.update(kwargs)
    return generation.panel_seed_rows(**base)


def test_candidate_panel_has_slots_candidates_and_one_clean_per_slot():
    rows = _rows()
    assert len(rows) == 2 * (4 + 1)
    assert sorted({(r["batch_slot"], r["role"]) for r in rows}) == [(1, "candidate"), (1, "clean"), (2, "candidate"), (2, "clean")]
    assert len({r["torch_seed"] for r in rows}) == len(rows)
    assert all(r["torch_seed"] == r["seed"] % 2 ** 32 for r in rows)


def test_extending_the_candidate_count_keeps_earlier_seeds():
    short = {(r["batch_slot"], r["position"]): r["seed"] for r in _rows(candidate_count=4) if r["role"] == "candidate"}
    long = {(r["batch_slot"], r["position"]): r["seed"] for r in _rows(candidate_count=8) if r["role"] == "candidate"}
    assert all(long[key] == seed for key, seed in short.items())
    tail = _rows(candidate_count=8, candidate_start=5, include_clean=False)
    assert {r["position"] for r in tail} == {5, 6, 7, 8} and all(r["role"] == "candidate" for r in tail)
    assert {(r["batch_slot"], r["position"]): r["seed"] for r in tail} == {k: v for k, v in long.items() if k[1] >= 5}


def test_clean_seed_of_slot_one_depends_on_prompt_not_master_key():
    one = [r for r in _rows() if r["role"] == "clean" and r["batch_slot"] == 1][0]["seed"]
    other_key = [r for r in _rows(master_key=bytes(32)) if r["role"] == "clean" and r["batch_slot"] == 1][0]["seed"]
    other_prompt = [r for r in _rows(prompt_uid="p2") if r["role"] == "clean" and r["batch_slot"] == 1][0]["seed"]
    assert one == other_key and one != other_prompt
    cand = [r["seed"] for r in _rows() if r["role"] == "candidate" and r["batch_slot"] == 1]
    assert cand != [r["seed"] for r in _rows(master_key=bytes(32)) if r["role"] == "candidate" and r["batch_slot"] == 1]


def test_reference_panel_is_independent_of_candidates():
    refs = _rows(mode="references")
    assert [r["role"] for r in refs] == ["reference"] * 4
    assert not {r["seed"] for r in refs} & {r["seed"] for r in _rows()}


def test_panel_seed_validation():
    with pytest.raises(ValueError):
        _rows(mode="other")
    with pytest.raises(ValueError):
        _rows(candidate_start=9)
    with pytest.raises(ValueError):
        _rows(master_key=b"short")
