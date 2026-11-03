"""Closed-API study (P17): request planning, panel files, exact-code detection."""

from __future__ import annotations

import base64
import csv
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "closed_api"
sys.path.insert(0, str(EXPERIMENT))

import closed_api_lib as lib  # noqa: E402
from watermark.config import load_detection_config  # noqa: E402
from watermark.prf import derive_trial_key  # noqa: E402

KEY = bytes(range(32))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location("closed_api_" + name, EXPERIMENT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unit(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _prompts(count):
    return [{"prompt_uid": "u%03d" % i, "text": "prompt %d" % i} for i in range(count)]


def _png(request):
    seed = int(request.request_uid[:8], 16)
    array = np.random.default_rng(seed).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_plan_has_n_candidates_and_one_clean_per_prompt():
    requests = lib.plan_requests(_prompts(3), candidate_count=8)
    assert len(requests) == 3 * 9
    assert len({request.request_uid for request in requests}) == len(requests)
    assert [request.role for request in requests[:9]] == ["candidate"] * 8 + ["clean"]
    assert requests[8].relative_path == "clean.png" and requests[0].relative_path == "candidate-001.png"
    with pytest.raises(ValueError):
        lib.plan_requests(_prompts(2) + _prompts(1), candidate_count=2)
    with pytest.raises(ValueError):
        lib.plan_requests([], candidate_count=2)


def test_the_paper_panel_is_450_requests_with_the_configured_candidate_count():
    n = load_detection_config().exception("closed_api")["N"]
    assert len(lib.plan_requests(_prompts(lib.PROMPT_COUNT), candidate_count=n)) == 450
    assert lib.MODEL_ID == "gpt-image-2.5-sunburst"


def test_payload_requests_one_low_quality_square_image():
    payload = lib.generation_payload(model_id="m", prompt="a cat")
    assert payload == {"model": "m", "prompt": "a cat", "n": 1, "quality": "low", "size": "1024x1024",
                       "output_format": "png"}
    with pytest.raises(ValueError):
        lib.generation_payload(model_id="", prompt="a cat")


def test_response_parsing():
    data = base64.b64encode(b"png-bytes").decode()
    assert lib.parse_generation_image({"data": [{"b64_json": data, "revised_prompt": "x"}]}) == b"png-bytes"
    for bad in ({"data": []}, {"data": [{}]}, {"data": [{"b64_json": data}, {"b64_json": data}]},
                {"data": [{"b64_json": "!!"}]}):
        with pytest.raises(ValueError):
            lib.parse_generation_image(bad)


def test_retry_delay_prefers_the_reset_duration():
    assert lib.retry_delay_seconds(headers={"retry-after": "3"}, detail="", attempt=1) == 4.0
    assert lib.retry_delay_seconds(headers={}, detail="Please try again in 250ms.", attempt=1) == pytest.approx(1.25)
    assert lib.retry_delay_seconds(headers={}, detail="", attempt=3) == 4.0
    assert lib.retry_delay_seconds(headers={}, detail="", attempt=10) == 16.0


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_generate_image_posts_with_the_bearer_key_and_returns_only_bytes(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["request"] = request
        return FakeResponse({"data": [{"b64_json": base64.b64encode(b"abc").decode()}]})

    monkeypatch.setattr(lib.urllib.request, "urlopen", fake_urlopen)
    assert lib.generate_image(api_key="k", model_id="m", prompt="p", request_uid="r") == b"abc"
    assert seen["request"].get_header("Authorization") == "Bearer k"
    assert json.loads(seen["request"].data)["n"] == 1
    with pytest.raises(ValueError):
        lib.generate_image(api_key="", model_id="m", prompt="p", request_uid="r")


def test_generate_image_retries_server_errors_but_not_client_errors(monkeypatch):
    calls = []
    monkeypatch.setattr(lib.time, "sleep", lambda seconds: None)

    def failing(code):
        def fake_urlopen(request, timeout):
            calls.append(code)
            raise urllib.error.HTTPError("u", code, "m", {}, io.BytesIO(b"{}"))
        return fake_urlopen

    monkeypatch.setattr(lib.urllib.request, "urlopen", failing(500))
    with pytest.raises(RuntimeError, match="500"):
        lib.generate_image(api_key="k", model_id="m", prompt="p", request_uid="r", maximum_attempts=3)
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(lib.urllib.request, "urlopen", failing(400))
    with pytest.raises(RuntimeError, match="400"):
        lib.generate_image(api_key="k", model_id="m", prompt="p", request_uid="r")
    assert calls == [400]


def test_panel_generation_resumes_without_new_requests_and_stores_no_provider_identifiers(tmp_path):
    calls = []

    def generate(request):
        calls.append(request.request_uid)
        return _png(request)

    prompts = _prompts(2)
    lib.generate_panel(prompts=prompts, output_root=tmp_path, model_id="m", candidate_count=3, generate=generate)
    assert len(calls) == 8
    lib.generate_panel(prompts=prompts, output_root=tmp_path, model_id="m", candidate_count=3, generate=generate)
    assert len(calls) == 8
    for path in tmp_path.rglob("*.json"):
        assert "request_id" not in path.read_text() and "usage" not in path.read_text()
    panel = lib.load_closed_api_panel(panel_root=tmp_path, prompt_rows=prompts, model_id="m", candidate_count=3)
    assert panel.prompt_uids == ("u000", "u001") and all(len(paths) == 3 for paths in panel.candidate_paths)


def test_a_tampered_image_is_rejected_and_regenerated(tmp_path):
    prompts = _prompts(1)
    lib.generate_panel(prompts=prompts, output_root=tmp_path, model_id="m", candidate_count=2, generate=_png)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / "u000" / "candidate-001.png")
    with pytest.raises(ValueError, match="hash mismatch"):
        lib.load_closed_api_panel(panel_root=tmp_path, prompt_rows=prompts, model_id="m", candidate_count=2)
    calls = []
    lib.generate_panel(prompts=prompts, output_root=tmp_path, model_id="m", candidate_count=2,
                       generate=lambda request: calls.append(request) or _png(request))
    assert len(calls) == 1
    lib.load_closed_api_panel(panel_root=tmp_path, prompt_rows=prompts, model_id="m", candidate_count=2)
    with pytest.raises(ValueError, match="identity"):
        lib.load_closed_api_panel(panel_root=tmp_path, prompt_rows=prompts, model_id="other", candidate_count=2)


def test_prompt_groups_partition_the_prompts_deterministically():
    uids = ["u%03d" % i for i in range(50)]
    groups = lib.matched_prompt_groups(uids, batch_size=5, grouping_seed=1)
    assert len(groups) == 10 and sorted(uid for group in groups for uid in group) == uids
    assert groups == lib.matched_prompt_groups(uids, batch_size=5, grouping_seed=1)
    assert groups != lib.matched_prompt_groups(uids, batch_size=5, grouping_seed=2)
    with pytest.raises(ValueError):
        lib.matched_prompt_groups(uids, batch_size=7, grouping_seed=1)


def _evaluate(prompts=20, n=8, dimension=24, **overrides):
    rng = np.random.default_rng(0)
    base = unit(rng.normal(size=(prompts, 1, dimension)))
    candidates = unit(base + 0.8 * rng.normal(size=(prompts, n, dimension)))
    clean = unit(base[:, 0] + 0.8 * rng.normal(size=(prompts, dimension)))
    arguments = dict(candidates=candidates, clean=clean, prompt_uids=["u%03d" % i for i in range(prompts)],
                     frame_seeds=(11, 12), keys={"key-01": KEY, "key-02": derive_trial_key(KEY, 1)}, m=4,
                     batch_sizes=(1, 5), grouping_seed=7, fpr_targets=(0.05, 0.2))
    arguments.update(overrides)
    return lib.evaluate_embeddings(**arguments)


def test_evaluation_structure_and_bounds():
    result = _evaluate()
    assert len(result["rows"]) == 20 * 2 * 2
    assert len(result["replicates"]) == 2 * 2 * 2
    assert [item["B"] for item in result["summary"]] == [1, 5]
    assert [item["batches_per_replicate"] for item in result["summary"]] == [20, 4]
    for item in result["summary"]:
        assert 0.0 <= item["min_auc"] <= item["mean_auc"] <= item["max_auc"] <= 1.0
        assert [cell["requested_fpr"] for cell in item["fixed_fpr"]] == [0.05, 0.2]
        assert all(cell["mean_empirical_fpr"] <= cell["requested_fpr"] + 1e-12 for cell in item["fixed_fpr"])
    assert 1.0 <= result["mean_distinct_codes"] <= 8.0
    assert result["pair_count"] == 20 * 28
    assert result["mean_pairwise_cosine_distance"] == pytest.approx(1.0 - result["mean_pairwise_cosine_similarity"])


@pytest.mark.parametrize("batch_size", [1, 5])
def test_mean_roc_runs_from_zero_to_one_monotonically(batch_size):
    curve = [row for row in _evaluate()["mean_roc"] if row["B"] == batch_size]
    assert len(curve) == lib.MEAN_ROC_POINTS and curve[0]["fpr"] == 0.0 and curve[-1]["fpr"] == 1.0
    tprs = [row["mean_tpr"] for row in curve]
    assert tprs == sorted(tprs) and tprs[-1] == 1.0
    assert all(row["std_tpr"] >= 0.0 for row in curve)


def test_evaluation_is_deterministic_and_validates_inputs():
    first, second = _evaluate(), _evaluate()
    assert first["summary"] == second["summary"]
    with pytest.raises(ValueError):
        _evaluate(keys={})
    with pytest.raises(ValueError):
        _evaluate(keys={"k": b"short"})
    with pytest.raises(ValueError):
        _evaluate(batch_sizes=(1, 1))
    with pytest.raises(ValueError):
        _evaluate(prompt_uids=["same"] * 20)


def test_aggregation_improves_detection_when_a_watermark_is_present():
    rng = np.random.default_rng(1)
    prompts, n, dimension = 40, 8, 24
    candidates = unit(rng.normal(size=(prompts, n, dimension)))
    clean = unit(rng.normal(size=(prompts, dimension)))
    result = lib.evaluate_embeddings(
        candidates=candidates, clean=clean, prompt_uids=["u%03d" % i for i in range(prompts)],
        frame_seeds=(1, 2, 3), keys={"k%d" % i: derive_trial_key(KEY, i) for i in range(3)}, m=8,
        batch_sizes=(1, 5), grouping_seed=3, fpr_targets=(0.2,),
    )
    single, batch = result["summary"]
    assert single["mean_auc"] > 0.5 and batch["mean_auc"] > single["mean_auc"]


class FakeEncoder:
    def embed(self, images):
        values = [np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float64).ravel() + 1.0 for image in images]
        return unit(np.stack(values))


def test_generate_then_evaluate_end_to_end(tmp_path, monkeypatch):
    generate = load_script("generate_panel")
    evaluate = load_script("evaluate")
    plot = load_script("plot_roc")
    grid = load_script("umbrella_grid")
    config = load_detection_config()
    prompts = _prompts(lib.PROMPT_COUNT)
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prompts))

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(sys, "argv", ["generate_panel.py", "--prompts", str(tmp_path / "prompts.jsonl"),
                                      "--output-dir", str(tmp_path / "panel")])
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        generate.main()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = []

    def fake_image(*, api_key, model_id, prompt, request_uid):
        seen.append((api_key, model_id))
        return _png(lib.ClosedApiRequest("", prompt, "", None, request_uid, ""))

    monkeypatch.setattr(generate, "generate_image", fake_image)
    assert generate.main() == 0
    assert len(seen) == 450 and set(seen) == {("test-key", lib.MODEL_ID)}

    (tmp_path / "master.key").write_bytes(KEY)
    monkeypatch.setattr(evaluate, "load_encoder", lambda *args, **kwargs: FakeEncoder())
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--panel-root", str(tmp_path / "panel"),
                                      "--prompts", str(tmp_path / "prompts.jsonl"), "--checkpoint", "unused",
                                      "--key-file", str(tmp_path / "master.key"), "--output-dir", str(tmp_path / "out"),
                                      "--device", "cpu"])
    assert evaluate.main() == 0
    sizes = config.exception("closed_api")["set_sizes"]
    summary = list(csv.DictReader((tmp_path / "out" / "summary.csv").open()))
    assert len(summary) == len(sizes) * len(config.fpr_targets)
    replicates = list(csv.DictReader((tmp_path / "out" / "replicates.csv").open()))
    assert len(replicates) == config.frames * config.keys * len(sizes)
    assert len(list(csv.DictReader((tmp_path / "out" / "per_prompt.csv").open()))) == lib.PROMPT_COUNT
    metrics = {row["metric"] for row in csv.DictReader((tmp_path / "out" / "diversity.csv").open())}
    assert "mean_distinct_codes" in metrics

    monkeypatch.setattr(sys, "argv", ["plot_roc.py", "--mean-roc", str(tmp_path / "out" / "mean_roc.csv"),
                                      "--output", str(tmp_path / "roc.pdf")])
    assert plot.main() == 0 and (tmp_path / "roc.pdf").stat().st_size > 1000
    monkeypatch.setattr(sys, "argv", ["umbrella_grid.py", "--panel-root", str(tmp_path / "panel"),
                                      "--prompt-uid", "u000", "--output", str(tmp_path / "grid.jpg")])
    assert grid.main() == 0
    with Image.open(tmp_path / "grid.jpg") as montage:
        assert montage.size == (4 * grid.TILE, 2 * grid.TILE)
