"""Encoder comparison (P2, P3, P4): statistics, panels, stability controls, distinct codes."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "encoder_comparison"
sys.path.insert(0, str(EXPERIMENT))

import comparison_lib as lib  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location("encoder_comparison_" + name, EXPERIMENT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unit(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- statistics


def test_identical_generations_have_zero_entropy_and_unit_cosine():
    same = np.tile(unit(np.random.default_rng(0).normal(size=(3, 1, 8))), (1, 16, 1))
    distances = lib.pair_distances(same)
    assert distances.shape == (3, 120)
    p, h = lib.estimate(distances, np.array([lib.TABLE_EPSILON]))
    assert p[0] == 1.0 and h[0] == 0.0
    assert lib.mean_pairwise_cosine(same) == pytest.approx(1.0)


def test_orthogonal_generations_are_never_close_so_entropy_is_infinite():
    basis = np.tile(np.eye(16)[None], (2, 1, 1))
    p, h = lib.estimate(lib.pair_distances(basis), np.array([lib.TABLE_EPSILON]))
    assert p[0] == 0.0 and np.isinf(h[0])
    assert lib.mean_pairwise_cosine(basis) == pytest.approx(0.0)


def test_estimate_is_the_negative_log_of_the_pooled_close_fraction():
    rng = np.random.default_rng(1)
    distances = rng.uniform(0, 2, size=(5, 120))
    grid = lib.EPSILON_GRID
    p, h = lib.estimate(distances, grid)
    index = int(np.argmin(np.abs(grid - 0.5)))
    assert p[index] == pytest.approx(np.mean(distances <= grid[index]))
    assert h[index] == pytest.approx(-np.log(p[index]))


def test_pair_distances_rejects_bad_input():
    with pytest.raises(ValueError):
        lib.pair_distances(np.ones((2, 1, 4)))
    with pytest.raises(ValueError):
        lib.pair_distances(np.zeros((2, 4, 4)))


def test_block_cosine_matches_the_pooled_mean():
    rng = np.random.default_rng(3)
    groups = unit(rng.normal(size=(7, 16, 10)))
    per_block = [lib.block_mean_cosine(group, block_size=16) for group in groups]
    assert lib.mean_pairwise_cosine(groups) == pytest.approx(np.mean(per_block))
    with pytest.raises(ValueError):
        lib.block_mean_cosine(groups[0][:8], block_size=16)


# --------------------------------------------------------------------------- seeds and panels


def test_flux_seeds_are_sixteen_distinct_seeds_per_prompt_and_disjoint_across_prompts():
    for variant in lib.FLUX_VARIANTS:
        seen = set()
        for index in (0, 1, lib.PROMPT_COUNT - 1):
            seeds = lib.flux_seeds(variant, index)
            assert len(seeds) == lib.GROUP_SIZE == len(set(seeds))
            assert not seen & set(seeds)
            seen |= set(seeds)
    with pytest.raises(ValueError):
        lib.flux_seeds("base", lib.PROMPT_COUNT)


def test_flux_variants_carry_the_paper_models():
    assert lib.FLUX_VARIANTS["distilled"]["repo"].endswith("FLUX.2-klein-4B")
    assert lib.FLUX_VARIANTS["base"]["repo"].endswith("FLUX.2-klein-base-4B")
    assert lib.FLUX_VARIANTS["distilled"]["steps"] < lib.FLUX_VARIANTS["base"]["steps"]


def test_select_prompts_takes_the_first_sorted_identities():
    rows = [{"prompt_uid": "p%03d" % index, "text": "t"} for index in (5, 1, 3, 2, 4)]
    assert [row["prompt_uid"] for row in lib.select_prompts(rows, 3)] == ["p001", "p002", "p003"]
    with pytest.raises(ValueError):
        lib.select_prompts(rows, 6)
    with pytest.raises(ValueError):
        lib.select_prompts(rows + rows[:1], 3)


def _image_file(path: Path, value: int = 0) -> None:
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def _write_panel(root: Path, uid: str, *, kind: str, count: int = 16) -> None:
    folder = root / uid
    folder.mkdir(parents=True)
    names = []
    for position in range(1, count + 1):
        name = ("candidate_%02d.png" if kind == "sd21" else "image_%02d.png") % position
        if kind == "sd21":
            (folder / "slot_001").mkdir(exist_ok=True)
            name = "slot_001/" + name
        _image_file(folder / name, position)
        names.append({"candidate_position": position, "relative_path": name})
    record = ({"batch_slots": [{"slot": 1, "candidate_images": list(reversed(names))},
                               {"slot": 2, "candidate_images": []}]}
              if kind == "sd21" else {"images": list(reversed(names))})
    (folder / "panel.json").write_text(json.dumps(record))


@pytest.mark.parametrize("kind", ["sd21", "flux"])
def test_panel_image_paths_follow_position_order(kind, tmp_path):
    _write_panel(tmp_path, "u1", kind=kind)
    paths = lib.panel_image_paths(tmp_path, "u1")
    assert len(paths) == 16 and all(path.is_file() for path in paths)
    assert [int(np.asarray(Image.open(path))[0, 0, 0]) for path in paths] == list(range(1, 17))


def test_panel_image_paths_reject_incomplete_panels(tmp_path):
    _write_panel(tmp_path, "u1", kind="flux", count=15)
    with pytest.raises(ValueError, match="positions"):
        lib.panel_image_paths(tmp_path, "u1")


def test_panel_prompt_uids_need_200_prompts(tmp_path, monkeypatch):
    for uid in ("c", "a", "b"):
        _write_panel(tmp_path, uid, kind="flux")
    with pytest.raises(ValueError):
        lib.panel_prompt_uids(tmp_path)
    monkeypatch.setattr(lib, "PROMPT_COUNT", 2)
    assert lib.panel_prompt_uids(tmp_path) == ["a", "b"]


# --------------------------------------------------------------------------- stability controls


def _class_tree(root: Path, classes: int, images: int) -> None:
    for split in ("train", "val"):
        for index in range(classes):
            folder = root / split / ("n%02d" % index)
            folder.mkdir(parents=True)
            for item in range(images):
                (folder / ("%s_%03d.JPEG" % (split, item))).write_bytes(b"x")


def test_perturbation_blocks_use_one_distinct_source_per_block(tmp_path):
    _class_tree(tmp_path / "imagenette", 10, 20)
    _class_tree(tmp_path / "imagewoof", 10, 20)
    blocks = lib.build_perturbation_blocks(tmp_path / "imagenette", tmp_path / "imagewoof", count=20, block_size=16)
    assert set(blocks) == set(lib.FAMILIES)
    for family in lib.FAMILIES:
        assert len(blocks[family]) == 20
        assert all(len(set(block.paths)) == 1 and len(block.paths) == 16 for block in blocks[family])
    sources = [block.paths[0] for block in blocks["jpeg"]]
    assert len(set(sources)) == 20
    assert [block.paths[0] for block in blocks["gaussian_blur"]] == sources
    again = lib.build_perturbation_blocks(tmp_path / "imagenette", tmp_path / "imagewoof", count=20, block_size=16)
    assert [block.paths[0] for block in again["jpeg"]] == sources


def test_perturbation_blocks_need_enough_images(tmp_path):
    _class_tree(tmp_path / "imagenette", 10, 5)
    _class_tree(tmp_path / "imagewoof", 10, 5)
    with pytest.raises(ValueError, match="enough"):
        lib.build_perturbation_blocks(tmp_path / "imagenette", tmp_path / "imagewoof", count=20, block_size=16)


@pytest.mark.parametrize("family", lib.FAMILIES)
def test_perturbations_keep_geometry_and_grow_with_the_index(family):
    rng = np.random.default_rng(4)
    image = Image.fromarray(rng.integers(0, 256, size=(40, 48, 3), dtype=np.uint8))
    first, last = lib._augment(image, family, 0), lib._augment(image, family, 15)
    assert first.size == last.size == image.size
    source = np.asarray(image, dtype=float)
    assert np.abs(np.asarray(last, dtype=float) - source).mean() > 0.0
    assert np.abs(np.asarray(first, dtype=float) - source).mean() != np.abs(np.asarray(last, dtype=float) - source).mean()


def test_jpeg_quality_ranges_from_35_to_95_and_blur_radius_from_quarter_to_one_and_three_quarters():
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    errors = [np.abs(np.asarray(lib._augment(image, "jpeg", index), dtype=float) - np.asarray(image, dtype=float)).mean()
              for index in range(16)]
    assert errors[0] > errors[-1]
    blurred = [np.asarray(lib._augment(image, "gaussian_blur", index), dtype=float).std() for index in range(16)]
    assert blurred[0] > blurred[-1]


def test_load_variants_returns_sixteen_images(tmp_path):
    path = tmp_path / "source.png"
    _image_file(path, 90)
    block = lib.ImageBlock("jpeg", "b", (path,) * 16)
    images = lib.load_variants(block)
    assert len(images) == 16 and all(image.mode == "RGB" for image in images)


class FakeEncoder:
    """Embeds an image by its flattened 4 x 4 thumbnail."""

    provenance = {"fake": True}

    def embed(self, images):
        values = [np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float64).ravel() + 1.0 for image in images]
        return unit(np.stack(values))


def test_stability_encode_and_table_round_trip(tmp_path, monkeypatch):
    stability = load_script("stability")
    _class_tree(tmp_path / "imagenette", 10, 20)
    _class_tree(tmp_path / "imagewoof", 10, 20)
    for root in (tmp_path / "imagenette", tmp_path / "imagewoof"):
        for path in root.rglob("*.JPEG"):
            Image.fromarray(np.random.default_rng(len(path.name)).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)).save(path, format="JPEG")
    real = lib.build_perturbation_blocks
    monkeypatch.setattr(stability, "build_perturbation_blocks",
                        lambda a, b: real(a, b, count=20, block_size=16))
    import watermark.embedding as embedding

    monkeypatch.setattr(embedding, "load_encoder", lambda *a, **k: FakeEncoder())
    out = tmp_path / "out"
    for encoder in lib.ENCODERS:
        stability.encode(SimpleNamespace(encoder=encoder, checkpoint=None, device="cpu", output_dir=out,
                                         imagenette_root=tmp_path / "imagenette",
                                         imagewoof_root=tmp_path / "imagewoof"))
    with np.load(out / "embeddings" / "ours.npz") as data:
        assert set(data.files) == set(lib.FAMILIES) and data["jpeg"].shape == (20, 16, 48)
    rows = list(csv.DictReader((out / "per_block" / "ours.csv").open()))
    assert len(rows) == 4 * 20
    monkeypatch.setattr(stability, "PROMPT_COUNT", 20)
    stability.table(SimpleNamespace(output_dir=out))
    table = list(csv.reader((out / "stability_table.csv").open()))
    assert table[0] == ["family", *lib.ENCODERS] and len(table) == 5
    assert all(0.0 < float(value) <= 1.0 for row in table[1:] for value in row[1:])


# --------------------------------------------------------------------------- semantic entropy


def _embedding_dir(root: Path, *, prompts: int = 200, dimension: int = 12, shift: int = 0) -> None:
    root.mkdir(parents=True)
    uids = np.array(["u%03d" % index for index in range(prompts)])
    for offset, encoder in enumerate(lib.ENCODERS):
        rng = np.random.default_rng(100 + offset + shift)
        np.savez(root / (encoder + ".npz"), embeddings=unit(rng.normal(size=(prompts, 16, dimension))).astype(np.float32),
                 prompt_uids=uids)


def test_semantic_entropy_writes_curves_and_table(tmp_path):
    script = load_script("semantic_entropy")
    _embedding_dir(tmp_path / "emb")
    sys.argv = ["semantic_entropy.py", "--embeddings-dir", str(tmp_path / "emb"), "--output-dir", str(tmp_path / "out")]
    assert script.main() == 0
    curves = list(csv.DictReader((tmp_path / "out" / "curves.csv").open()))
    assert len(curves) == 5 * 401 and {row["prompts"] for row in curves} == {"200"}
    table = list(csv.DictReader((tmp_path / "out" / "table.csv").open()))
    assert [row["encoder"] for row in table] == list(lib.ENCODERS)
    for row in table:
        rows = [r for r in curves if r["encoder"] == row["encoder"]]
        index = int(np.argmin([abs(float(r["epsilon"]) - lib.TABLE_EPSILON) for r in rows]))
        assert float(rows[index]["entropy"]) == pytest.approx(float(row["entropy_at_table_radius"]))
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert provenance["pairs_per_prompt"] == 120


def test_semantic_entropy_rejects_mismatched_prompts(tmp_path):
    script = load_script("semantic_entropy")
    _embedding_dir(tmp_path / "emb")
    with np.load(tmp_path / "emb" / "ours.npz") as data:
        values = data["embeddings"]
    np.savez(tmp_path / "emb" / "ours.npz", embeddings=values, prompt_uids=np.array(["z%03d" % i for i in range(200)]))
    with pytest.raises(ValueError, match="prompt identities"):
        script.load_embeddings(tmp_path / "emb")


def test_entropy_figure_renders(tmp_path):
    script = load_script("semantic_entropy")
    plot = load_script("plot_semantic_entropy_epsilon")
    paths = {}
    for name in ("sd21", "flux_distilled", "flux_base"):
        _embedding_dir(tmp_path / ("emb_" + name), shift=len(name))
        sys.argv = ["x", "--embeddings-dir", str(tmp_path / ("emb_" + name)), "--output-dir", str(tmp_path / name)]
        script.main()
        paths[name] = tmp_path / name / "curves.csv"
    sys.argv = ["x", "--sd21", str(paths["sd21"]), "--flux-distilled", str(paths["flux_distilled"]),
                "--flux-base", str(paths["flux_base"]), "--output", str(tmp_path / "figure.pdf")]
    plot.main()
    assert (tmp_path / "figure.pdf").stat().st_size > 1000


# --------------------------------------------------------------------------- distinct codes


def test_distinct_code_curve_bounds_and_extremes():
    codes = load_script("distinct_codes")
    rng = np.random.default_rng(6)
    seeds = codes.draw_seeds(2026090200)[:6]
    same = np.tile(unit(rng.normal(size=(5, 1, 32))), (1, 16, 1))
    assert np.allclose(codes.distinct_code_curve(same, seeds), 1.0)
    spread = unit(rng.normal(size=(40, 16, 32)))
    curve = codes.distinct_code_curve(spread, seeds)
    assert curve.shape == (16,) and curve[0] == 1.0
    assert np.all(np.diff(curve) >= 0) and np.all(curve <= np.arange(1, 17) + 1e-9)
    assert curve[-1] > 8.0
    with pytest.raises(ValueError):
        codes.distinct_code_curve(spread[0], seeds)


def test_distinct_code_curve_counts_collisions_exactly():
    codes = load_script("distinct_codes")
    seeds = codes.draw_seeds(2026090200)[:3]
    base = unit(np.random.default_rng(7).normal(size=(1, 2, 32)))
    pair = np.concatenate([base[:, :1], base[:, :1], base[:, 1:]], axis=1)  # members 1 and 2 identical
    curve = codes.distinct_code_curve(pair, seeds)
    assert curve[0] == 1.0 and curve[1] == 1.0 and curve[2] >= 1.0


def test_draw_seeds_are_disjoint_from_the_detector_frames():
    from watermark.config import load_detection_config

    codes = load_script("distinct_codes")
    config = load_detection_config()
    seeds = codes.draw_seeds(config.frame_seed_base)
    assert len(seeds) == len(set(seeds)) == codes.DRAWS
    assert not set(seeds) & set(config.frame_seeds())


def test_distinct_code_figures_render(tmp_path):
    codes = load_script("distinct_codes")
    stability_dir = tmp_path / "stability"
    stability_dir.mkdir()
    rng = np.random.default_rng(8)
    for encoder in lib.ENCODERS:
        np.savez(stability_dir / (encoder + ".npz"),
                 **{family: unit(rng.normal(size=(6, 16, 10))).astype(np.float32) for family in lib.FAMILIES})
    monkey_draws = codes.DRAWS
    codes.DRAWS = 4
    try:
        codes.perturbations(SimpleNamespace(embeddings_dir=stability_dir, output_dir=tmp_path / "p"))
        _embedding_dir(tmp_path / "g1", prompts=200)
        codes.generations(SimpleNamespace(panel=["A=%s" % (tmp_path / "g1")], output_dir=tmp_path / "g"))
    finally:
        codes.DRAWS = monkey_draws
    for folder, name in (("p", "unique_codes_perturbations.pdf"), ("g", "unique_codes_generations.pdf")):
        assert (tmp_path / folder / name).stat().st_size > 1000
        rows = list(csv.DictReader((tmp_path / folder / "curves.csv").open()))
        assert rows and {int(row["included"]) for row in rows} == set(range(1, 17))


# --------------------------------------------------------------------------- FLUX generation


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        value = len(self.calls) * 7 % 256
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), (value, 0, 0))])


def test_flux_panel_layout_round_trips_through_the_reader(tmp_path):
    generate = load_script("generate_flux")
    pipe = FakePipe()
    generate.write_prompt_panel(pipe, variant="base", prompt_index=3, prompt_uid="u3", text="a cat",
                                output_dir=tmp_path, device="cpu")
    assert [call["generator"].initial_seed() for call in pipe.calls] == lib.flux_seeds("base", 3)
    assert {call["num_inference_steps"] for call in pipe.calls} == {lib.FLUX_VARIANTS["base"]["steps"]}
    assert {call["guidance_scale"] for call in pipe.calls} == {lib.FLUX_VARIANTS["base"]["guidance"]}
    assert len(lib.panel_image_paths(tmp_path, "u3")) == 16
    record = json.loads((tmp_path / "u3" / "panel.json").read_text())
    assert [item["seed"] for item in record["images"]] == lib.flux_seeds("base", 3)
    with pytest.raises(ValueError, match="overwrite"):
        generate.write_prompt_panel(pipe, variant="base", prompt_index=3, prompt_uid="u3", text="a cat",
                                    output_dir=tmp_path, device="cpu")
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]
