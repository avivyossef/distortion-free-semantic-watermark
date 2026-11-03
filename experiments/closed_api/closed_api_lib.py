"""GPT Image as a black-box generator: request planning, panel files, exact-code detection.

A panel holds, per prompt, ``N`` independently requested watermark candidates and one
independently requested clean control (every provider call asks for a single image). Detection
uses the exact keyed score of the M-bit hyperplane code (Hamming radius 0).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from watermark.coding import code_embedding, sample_hyperplanes
from watermark.io import canonical_json, contained_path, pixel_sha256, sha256_file
from watermark.metrics import clean_detection_roc
from watermark.prf import score_code
from watermark.selection import cell_rng, select_candidate

MODEL_ID = "gpt-image-2.5-sunburst"
PROMPT_COUNT = 50  # the first 50 prompts of the held-out registry
GENERATIONS_URL = "https://api.openai.com/v1/images/generations"
MEAN_ROC_POINTS = 101


# --------------------------------------------------------------------------- API client


def retry_delay_seconds(*, headers: Mapping[str, Any], detail: str, attempt: int) -> float:
    """Prefer the API's reset duration over a too-short generic backoff."""
    values = [
        headers.get("retry-after"),
        headers.get("x-ratelimit-reset-images"),
        headers.get("x-ratelimit-reset-input-images"),
    ]
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)(ms|s)\b", detail, re.IGNORECASE)
    if match:
        values.append(match.group(1) + match.group(2).lower())
    for raw in values:
        if raw is None:
            continue
        value = str(raw).strip().lower()
        duration = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s)?", value)
        if duration:
            seconds = float(duration.group(1))
            if duration.group(2) == "ms":
                seconds /= 1000.0
            return seconds + 1.0
    return float(min(2 ** (attempt - 1), 16))


def generation_payload(*, model_id: str, prompt: str) -> Dict[str, Any]:
    """One low-quality 1024 x 1024 PNG per request."""
    if not model_id or not prompt.strip():
        raise ValueError("model_id and prompt must be nonempty")
    return {"model": model_id, "prompt": prompt, "n": 1, "quality": "low", "size": "1024x1024",
            "output_format": "png"}


def parse_generation_image(response: Mapping[str, Any]) -> bytes:
    data = response.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("OpenAI image generation returned an unexpected result count")
    item = data[0]
    if not isinstance(item, Mapping) or not isinstance(item.get("b64_json"), str):
        raise ValueError("OpenAI image generation returned no base64 image")
    try:
        return base64.b64decode(item["b64_json"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("OpenAI image generation returned invalid image data") from error


def generate_image(
    *, api_key: str, model_id: str, prompt: str, request_uid: str,
    timeout_seconds: int = 180, maximum_attempts: int = 5,
) -> bytes:
    """Request one image; returns the PNG bytes (provider identifiers are discarded)."""
    if not api_key:
        raise ValueError("OpenAI API key is required")
    body = json.dumps(generation_payload(model_id=model_id, prompt=prompt)).encode()
    request = urllib.request.Request(
        GENERATIONS_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Client-Request-Id": request_uid,
        },
    )
    for attempt in range(1, maximum_attempts + 1):
        delay = float(min(2 ** (attempt - 1), 16))
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as result:
                response = json.loads(result.read())
            return parse_generation_image(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            quota_error = "insufficient_quota" in detail
            retryable = (error.code == 429 and not quota_error) or 500 <= error.code < 600
            if not retryable or attempt == maximum_attempts:
                raise RuntimeError(f"OpenAI API HTTP {error.code}: {detail}") from error
            delay = retry_delay_seconds(headers=error.headers, detail=detail, attempt=attempt)
        except urllib.error.URLError as error:
            if attempt == maximum_attempts:
                raise RuntimeError(f"OpenAI API request failed: {error.reason}") from error
        time.sleep(delay)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- panel generation


@dataclass(frozen=True)
class ClosedApiRequest:
    prompt_uid: str
    prompt: str
    role: str
    candidate_position: Optional[int]
    request_uid: str
    relative_path: str


def _request_uid(prompt_uid: str, role: str, position: Optional[int]) -> str:
    message = f"closed-api-panel/v1\0{prompt_uid}\0{role}\0{position or 0}".encode()
    return hashlib.sha256(message).hexdigest()


def plan_requests(prompts: Iterable[Mapping[str, Any]], *, candidate_count: int) -> Tuple[ClosedApiRequest, ...]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    output: List[ClosedApiRequest] = []
    seen = set()
    for raw in prompts:
        prompt_uid, prompt = raw.get("prompt_uid"), raw.get("text")
        if not isinstance(prompt_uid, str) or not prompt_uid or prompt_uid in seen:
            raise ValueError("prompt_uid values must be nonempty and unique")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt text must be nonempty")
        seen.add(prompt_uid)
        for position in range(1, candidate_count + 1):
            output.append(ClosedApiRequest(
                prompt_uid=prompt_uid,
                prompt=prompt,
                role="candidate",
                candidate_position=position,
                request_uid=_request_uid(prompt_uid, "candidate", position),
                relative_path=f"candidate-{position:03d}.png",
            ))
        output.append(ClosedApiRequest(
            prompt_uid=prompt_uid,
            prompt=prompt,
            role="clean",
            candidate_position=None,
            request_uid=_request_uid(prompt_uid, "clean", None),
            relative_path="clean.png",
        ))
    if not output:
        raise ValueError("at least one prompt is required")
    return tuple(output)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(dict(value)) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _materialize_image(
    root: Path, request: ClosedApiRequest, generate: Callable[[ClosedApiRequest], bytes],
) -> Dict[str, Any]:
    """Return the record of one panel image, requesting it only if no valid file exists."""
    image_path = root / request.relative_path
    metadata_path = image_path.with_suffix(".json")
    if image_path.is_file() and metadata_path.is_file():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (record.get("request_uid") == request.request_uid
                and record.get("file_sha256") == sha256_file(image_path)):
            with Image.open(image_path) as image:
                if record.get("pixel_sha256") == pixel_sha256(image):
                    return record

    image_bytes = generate(request)
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("closed API returned no image bytes")
    with Image.open(io.BytesIO(image_bytes)) as decoded:
        image = decoded.convert("RGB")
        temporary = image_path.with_suffix(".png.tmp")
        image.save(temporary, format="PNG")
    os.replace(temporary, image_path)
    with Image.open(image_path) as stored:
        record = {
            "role": request.role,
            "candidate_position": request.candidate_position,
            "request_uid": request.request_uid,
            "relative_path": request.relative_path,
            "file_sha256": sha256_file(image_path),
            "pixel_sha256": pixel_sha256(stored),
        }
    _write_json(metadata_path, record)
    return record


def generate_panel(
    *, prompts: Iterable[Mapping[str, Any]], output_root: Path, model_id: str, candidate_count: int,
    generate: Callable[[ClosedApiRequest], bytes],
) -> Dict[str, int]:
    """Write ``<output_root>/<prompt_uid>/`` with the images and a ``panel.json`` per prompt."""
    prompt_rows = tuple(dict(row) for row in prompts)
    if not model_id:
        raise ValueError("model_id must be nonempty")
    requests = plan_requests(prompt_rows, candidate_count=candidate_count)
    by_prompt = {str(row["prompt_uid"]): row for row in prompt_rows}
    output_root.mkdir(parents=True, exist_ok=True)
    for prompt_uid, prompt_row in by_prompt.items():
        root = output_root / prompt_uid
        root.mkdir(exist_ok=True)
        records = [_materialize_image(root, request, generate)
                   for request in requests if request.prompt_uid == prompt_uid]
        candidates = [record for record in records if record["role"] == "candidate"]
        clean = [record for record in records if record["role"] == "clean"]
        if len(candidates) != candidate_count or len(clean) != 1:
            raise ValueError("candidate and clean coverage is incomplete")
        _write_json(root / "panel.json", {
            "schema_version": "closed-api-candidate-panel/v1", "prompt_uid": prompt_uid,
            "prompt": prompt_row["text"], "model_id": model_id, "candidate_images": candidates, "clean": clean[0],
        })
    return {"prompt_count": len(prompt_rows), "image_count": len(requests)}


# --------------------------------------------------------------------------- panel loading


@dataclass(frozen=True)
class LoadedClosedApiPanel:
    prompt_uids: Tuple[str, ...]
    candidate_paths: Tuple[Tuple[Path, ...], ...]
    clean_paths: Tuple[Path, ...]


def _verified_path(root: Path, record: Mapping[str, Any]) -> Path:
    path = contained_path(root, record.get("relative_path"))
    if not path.is_file():
        raise ValueError("panel image is missing")
    if sha256_file(path) != record.get("file_sha256"):
        raise ValueError("panel image file hash mismatch")
    with Image.open(path) as image:
        if pixel_sha256(image) != record.get("pixel_sha256"):
            raise ValueError("panel image pixel hash mismatch")
    return path


def load_closed_api_panel(
    *, panel_root: Path, prompt_rows: Sequence[Mapping[str, Any]], model_id: str, candidate_count: int,
) -> LoadedClosedApiPanel:
    """Authenticate one complete panel, in prompt-registry order."""
    if not prompt_rows or candidate_count <= 0:
        raise ValueError("prompt registry and candidate count must be nonempty")
    prompt_uids: List[str] = []
    candidate_paths: List[Tuple[Path, ...]] = []
    clean_paths: List[Path] = []
    for prompt_row in prompt_rows:
        prompt_uid = prompt_row.get("prompt_uid")
        if not isinstance(prompt_uid, str) or not prompt_uid or prompt_uid in prompt_uids:
            raise ValueError("prompt registry has invalid identities")
        root = Path(panel_root) / prompt_uid
        record_path = root / "panel.json"
        if not record_path.is_file():
            raise ValueError(f"closed-API panel is incomplete for {prompt_uid}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("schema_version") != "closed-api-candidate-panel/v1"
                or record.get("prompt_uid") != prompt_uid
                or record.get("prompt") != prompt_row.get("text")
                or record.get("model_id") != model_id):
            raise ValueError("closed-API panel identity mismatch")
        candidates = sorted(record.get("candidate_images", []), key=lambda row: int(row.get("candidate_position", 0)))
        if [row.get("candidate_position") for row in candidates] != list(range(1, candidate_count + 1)):
            raise ValueError("closed-API panel candidate coverage mismatch")
        clean = record.get("clean")
        if not isinstance(clean, dict) or clean.get("role") != "clean":
            raise ValueError("closed-API panel clean control is missing")
        prompt_uids.append(prompt_uid)
        candidate_paths.append(tuple(_verified_path(root, row) for row in candidates))
        clean_paths.append(_verified_path(root, clean))
    return LoadedClosedApiPanel(tuple(prompt_uids), tuple(candidate_paths), tuple(clean_paths))


# --------------------------------------------------------------------------- detection


def matched_prompt_groups(
    prompt_uids: Sequence[str], *, batch_size: int, grouping_seed: int,
) -> Tuple[Tuple[str, ...], ...]:
    """Partition prompts once so clean and watermarked batches have identical membership."""
    if batch_size <= 0 or len(prompt_uids) % batch_size:
        raise ValueError("batch size must divide the closed-API prompt count")
    ordered = sorted(prompt_uids, key=lambda prompt_uid: hashlib.sha256(
        f"closed-api-detector-batch/v1\0{grouping_seed}\0{prompt_uid}".encode("utf-8")
    ).digest())
    return tuple(tuple(ordered[start:start + batch_size]) for start in range(0, len(ordered), batch_size))


def evaluate_embeddings(
    *, candidates: np.ndarray, clean: np.ndarray, prompt_uids: Sequence[str], frame_seeds: Sequence[int],
    keys: Mapping[str, bytes], m: int, batch_sizes: Sequence[int], grouping_seed: int,
    fpr_targets: Sequence[float],
) -> Dict[str, Any]:
    """Exact keyed scores over fixed frames and matched prompt batches.

    ``candidates`` is ``(prompts, N, D)`` and ``clean`` ``(prompts, D)``. Every (frame, key)
    pair is one replicate; a batch score is the mean of its prompts' scores. Returns the
    per-prompt rows, the per-replicate ROC summaries, the per-``B`` summary and mean ROC curves
    (step interpolation on 101 FPR points; deviations are population standard deviations over
    replicates) and the embedding diversity.
    """
    candidate_values = np.asarray(candidates, dtype=np.float64)
    clean_values = np.asarray(clean, dtype=np.float64)
    prompts = tuple(str(value) for value in prompt_uids)
    frames = tuple(int(value) for value in frame_seeds)
    key_items = tuple((str(key_id), value) for key_id, value in keys.items())
    batches = tuple(int(value) for value in batch_sizes)
    if candidate_values.ndim != 3 or candidate_values.shape[0] == 0 or candidate_values.shape[1] < 2:
        raise ValueError("candidates must be a prompt x candidate x dimension matrix")
    if clean_values.shape != (candidate_values.shape[0], candidate_values.shape[2]):
        raise ValueError("clean embeddings do not align with candidates")
    if len(prompts) != candidate_values.shape[0] or len(set(prompts)) != len(prompts):
        raise ValueError("prompt identities do not align with candidates")
    if not frames or not 1 <= m <= 64:
        raise ValueError("frame seeds and M are required")
    if (not key_items or len({key_id for key_id, _key in key_items}) != len(key_items)
            or any(not key_id or not isinstance(key, bytes) or len(key) != 32 for key_id, key in key_items)):
        raise ValueError("keys must map unique nonempty IDs to 32-byte values")
    if not batches or len(set(batches)) != len(batches):
        raise ValueError("batch sizes must be unique and nonempty")
    prompt_groups = {
        batch_size: matched_prompt_groups(prompts, batch_size=batch_size, grouping_seed=int(grouping_seed))
        for batch_size in batches
    }
    if not np.isfinite(candidate_values).all() or not np.isfinite(clean_values).all():
        raise ValueError("embeddings are invalid")

    norms = np.linalg.norm(candidate_values, axis=2, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("candidate embeddings must be nonzero")
    normalized = candidate_values / norms
    upper = np.triu_indices(candidate_values.shape[1], k=1)
    prompt_cosines = [(vectors @ vectors.T)[upper] for vectors in normalized]
    mean_pairwise_cosine = float(np.concatenate(prompt_cosines).mean())

    rows: List[Dict[str, Any]] = []
    for frame_seed in frames:
        planes = sample_hyperplanes(frame_seed, candidate_values.shape[2])
        codes_by_prompt = tuple(tuple(
            code_embedding(vector, planes[:m]) for vector in candidate_values[prompt_index]
        ) for prompt_index in range(len(prompts)))
        clean_codes = tuple(code_embedding(clean_values[prompt_index], planes[:m]) for prompt_index in range(len(prompts)))
        for key_id, key in key_items:
            for prompt_index, prompt_uid in enumerate(prompts):
                trace = select_candidate(
                    codes_by_prompt[prompt_index], key, cell_rng("closed-api", frame_seed, m, key_id, prompt_uid),
                )
                clean_score = score_code(key, clean_codes[prompt_index])
                rows.append({
                    "prompt_uid": prompt_uid, "key_id": key_id, "frame_seed": frame_seed,
                    "N": int(candidate_values.shape[1]), "M": m, "distinct_codes": trace.distinct_code_count,
                    "selected_position": trace.selected_index + 1, "selection_branch": trace.branch,
                    "clean_score": clean_score, "watermarked_score": trace.selected_score,
                    "score_gap": trace.selected_score - clean_score,
                })

    by_identity = {(row["prompt_uid"], row["key_id"], row["frame_seed"]): row for row in rows}
    replicates: List[Dict[str, Any]] = []
    for frame_seed in frames:
        for key_id, _key in key_items:
            for batch_size in batches:
                current = []
                for members in prompt_groups[batch_size]:
                    member_rows = [by_identity[(prompt_uid, key_id, frame_seed)] for prompt_uid in members]
                    current.append({
                        "clean_score": float(np.mean([row["clean_score"] for row in member_rows])),
                        "watermarked_score": float(np.mean([row["watermarked_score"] for row in member_rows])),
                    })
                roc = clean_detection_roc(
                    [row["clean_score"] for row in current], [row["watermarked_score"] for row in current],
                    tuple(fpr_targets),
                )
                replicates.append({
                    "key_id": key_id, "frame_seed": frame_seed, "B": batch_size, "batch_count": len(current),
                    "auc": float(roc["auc"]), "roc": roc["roc"], "fixed_fpr": roc["fixed_fpr"],
                })

    summary: List[Dict[str, Any]] = []
    mean_roc: List[Dict[str, Any]] = []
    for batch_size in batches:
        selected = [row for row in replicates if row["B"] == batch_size]
        aucs = np.asarray([row["auc"] for row in selected], dtype=np.float64)
        fixed = []
        for index, requested in enumerate(fpr_targets):
            fixed.append({
                "requested_fpr": float(requested),
                "mean_empirical_fpr": float(np.mean([row["fixed_fpr"][index]["empirical_fpr"] for row in selected])),
                "mean_tpr": float(np.mean([row["fixed_fpr"][index]["tpr"] for row in selected])),
                "std_tpr": float(np.std([row["fixed_fpr"][index]["tpr"] for row in selected])),
            })
        summary.append({
            "B": batch_size, "replicate_count": len(selected), "batches_per_replicate": int(selected[0]["batch_count"]),
            "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std()), "min_auc": float(aucs.min()),
            "max_auc": float(aucs.max()), "fixed_fpr": fixed,
        })
        for requested_fpr in np.linspace(0.0, 1.0, MEAN_ROC_POINTS):
            tprs = [max(float(point["tpr"]) for point in replicate["roc"] if float(point["fpr"]) <= float(requested_fpr))
                    for replicate in selected]
            mean_roc.append({"B": batch_size, "mean_auc": float(aucs.mean()), "fpr": float(requested_fpr),
                             "mean_tpr": float(np.mean(tprs)), "std_tpr": float(np.std(tprs))})
    return {
        "rows": rows, "replicates": replicates, "summary": summary, "mean_roc": mean_roc,
        "mean_distinct_codes": float(np.mean([row["distinct_codes"] for row in rows])),
        "all_distinct_rate": float(np.mean([row["distinct_codes"] == candidate_values.shape[1] for row in rows])),
        "unique_max_rate": float(np.mean([row["selection_branch"] == "unique_max" for row in rows])),
        "mean_score_gap": float(np.mean([row["score_gap"] for row in rows])),
        "mean_pairwise_cosine_similarity": mean_pairwise_cosine,
        "mean_pairwise_cosine_distance": 1.0 - mean_pairwise_cosine,
        "pair_count": int(len(prompts) * len(upper[0])),
    }
