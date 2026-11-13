"""Pair-only optimization, validation and resumable checkpoints for the semantic encoder."""

from __future__ import annotations

import hashlib
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import (
    BalancedPairSampler, PairDataset, PairKind, ScheduledPairUpdate, SourceCorpus, build_pair_records,
    randomized_balanced_updates,
)
from .model import NEGATIVE_CAP, DinoLoRAMetric, pair_loss, trainable_state_sha256

CHECKPOINT_VERSION = "step5_checkpoint_v2"  # the layout of the released checkpoint file
TRAINING_SEED = 20260814
UPDATES_PER_PROMPT = 4
LEARNING_RATE = 1e-4
LR_DECAY_PER_EPOCH = 0.95
MAX_GRAD_NORM = 1.0
_CHECKPOINT_CONFIG_FIELDS = frozenset({
    "model_id", "model_revision", "variant", "lora_rank", "adapted_blocks", "head_architecture",
})


class ResumeProvenanceError(ValueError):
    """Raised when a checkpoint is pointed at a different configuration."""


def validate_checkpoint_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Check the immutable model and architecture description stored in a checkpoint."""
    normalized = dict(config)
    if frozenset(normalized) != _CHECKPOINT_CONFIG_FIELDS:
        raise ResumeProvenanceError("checkpoint config must contain exactly the model and architecture fields")
    if normalized["model_id"] != "facebook/dinov2-base":
        raise ResumeProvenanceError("checkpoint model id is not the pinned DINOv2-base model")
    revision = normalized["model_revision"]
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ResumeProvenanceError("checkpoint model revision must be an immutable 40-character commit")
    if normalized["variant"] not in {"head_only", "lora_linear_head", "lora_mlp"}:
        raise ResumeProvenanceError("checkpoint metric variant is invalid")
    if normalized["lora_rank"] not in {4, 8}:
        raise ResumeProvenanceError("checkpoint LoRA rank is invalid")
    if normalized["adapted_blocks"] != [8, 9, 10, 11]:
        raise ResumeProvenanceError("checkpoint LoRA blocks must be exactly [8, 9, 10, 11]")
    expected_head = (
        "cls_768_256_l2" if normalized["variant"] == "lora_linear_head" else "cls_768_512_gelu_dropout01_256_l2"
    )
    if normalized["head_architecture"] != expected_head:
        raise ResumeProvenanceError("checkpoint head architecture does not match metric variant")
    return normalized


@dataclass(frozen=True)
class CandidateConfiguration:
    """One validation-selected ablation: head variant and LoRA rank."""

    variant: str
    lora_rank: int


def candidate_name(candidate: CandidateConfiguration) -> str:
    return "%s-rank%d" % (candidate.variant, candidate.lora_rank)


@dataclass(frozen=True)
class PairTensorBatch:
    """Preprocessed endpoints for exactly one typed 16-pair optimizer update."""

    kind: PairKind
    left_pixel_values: Tensor
    right_pixel_values: Tensor

    def validate(self, *, require_sixteen: bool = True) -> None:
        if self.left_pixel_values.ndim != 4 or self.right_pixel_values.ndim != 4:
            raise ValueError("pair tensors must have shape [pairs, channels, height, width]")
        if self.left_pixel_values.shape != self.right_pixel_values.shape:
            raise ValueError("left and right pair tensors must have identical shapes")
        if require_sixteen and self.left_pixel_values.shape[0] != 16:
            raise ValueError("updates require exactly 16 pairs of one kind")


@dataclass(frozen=True)
class EndpointBatch:
    """Normalized-health input; variants allow non-final endpoints in the report."""

    pixel_values: Tensor
    seeds: Tuple[int, ...]
    prompt_uids: Optional[Tuple[str, ...]] = None
    variants: Optional[Tuple[str, ...]] = None

    def validate(self) -> None:
        if self.pixel_values.ndim != 4 or self.pixel_values.shape[0] != len(self.seeds):
            raise ValueError("endpoint pixels and seeds must have matching batch dimension")
        if self.prompt_uids is None or len(self.prompt_uids) != len(self.seeds) or any(
            not isinstance(prompt_uid, str) or not prompt_uid for prompt_uid in self.prompt_uids
        ):
            raise ValueError("endpoint prompt_uids and seeds must have matching nonempty identities")
        variants = self.variants or ("final",) * len(self.seeds)
        if len(variants) != len(self.seeds):
            raise ValueError("endpoint variants and seeds must have matching length")


@dataclass(frozen=True)
class OrderingExample:
    """An anchor should be more similar to positive than to negative."""

    anchor_pixel_values: Tensor
    positive_pixel_values: Tensor
    negative_pixel_values: Tensor


@dataclass(frozen=True)
class ValidationData:
    pair_batches: Tuple[PairTensorBatch, ...] = ()
    attack_seed_orderings: Tuple[OrderingExample, ...] = ()
    endpoint_batches: Tuple[EndpointBatch, ...] = ()


@dataclass(frozen=True)
class EmbeddingHealth:
    coordinate_std_mean: float
    coordinate_std_min: float
    coordinate_std_max: float
    effective_rank: float
    same_prompt_different_seed_final_cosine_mean: float
    same_prompt_different_seed_final_cosine_median: float
    attack_final_cosine_mean: float
    attack_final_cosine_median: float
    collapsed: bool


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    attack_seed_ordering_accuracy: float
    health: EmbeddingHealth
    consecutive_collapse_epochs: int
    stop_for_collapse: bool

    @property
    def selection_key(self) -> Tuple[float, float]:
        """Maximize validation orderings, then minimize validation-only loss."""
        return (self.attack_seed_ordering_accuracy, -self.loss)


@dataclass(frozen=True)
class EpochTrainingResult:
    loss: float
    optimizer_steps: int
    pair_count: int
    kind_counts: Mapping[str, int]


def _device_of(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _positive(kind: PairKind) -> bool:
    return kind is PairKind.ATTACK_POSITIVE


def _cosines(model: nn.Module, left: Tensor, right: Tensor) -> Tensor:
    return F.cosine_similarity(model(pixel_values=left), model(pixel_values=right), dim=1)


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[PairTensorBatch],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    negative_cap: float = NEGATIVE_CAP,
    max_grad_norm: float = MAX_GRAD_NORM,
    mixed_precision: bool = True,
) -> EpochTrainingResult:
    """One epoch of the typed-pair cosine objective: one optimizer step per 16-pair batch.

    ``scheduler`` is stepped once at the end of the epoch.
    """
    device = _device_of(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    pair_count = 0
    optimizer_steps = 0
    kind_counts = {kind.value: 0 for kind in PairKind}
    for batch in batches:
        batch.validate()
        left = batch.left_pixel_values.to(device)
        right = batch.right_pixel_values.to(device)
        with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
            loss = pair_loss(_cosines(model, left, right), positive=_positive(batch.kind), negative_cap=negative_cap)
            loss.backward()
        total_loss += float(loss.detach().cpu())
        pair_count += left.shape[0]
        kind_counts[batch.kind.value] += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    if scheduler is not None and optimizer_steps > 0:
        scheduler.step()
    return EpochTrainingResult(
        loss=total_loss / max(1, sum(kind_counts.values())),
        optimizer_steps=optimizer_steps, pair_count=pair_count, kind_counts=kind_counts,
    )


def _ordering_accuracy(model: nn.Module, examples: Sequence[OrderingExample], device: torch.device) -> float:
    if not examples:
        return 0.0
    correct = 0
    for example in examples:
        anchor = model(pixel_values=example.anchor_pixel_values.to(device))
        positive = model(pixel_values=example.positive_pixel_values.to(device))
        negative = model(pixel_values=example.negative_pixel_values.to(device))
        correct += bool((F.cosine_similarity(anchor, positive, dim=1) > F.cosine_similarity(anchor, negative, dim=1)).all())
    return correct / len(examples)


def _cosine_summary(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 1.0, 1.0
    tensor = torch.tensor(values)
    return float(tensor.mean()), float(torch.quantile(tensor, 0.5))


def embedding_health(
    model: nn.Module, endpoint_batches: Sequence[EndpointBatch], *, device: Optional[torch.device] = None
) -> EmbeddingHealth:
    """Prompt-conditioned collapse diagnostics from evaluated validation endpoints."""
    target_device = device or _device_of(model)
    embeddings: List[Tensor] = []
    final_endpoints: List[Tuple[str, int, Tensor]] = []
    attack_endpoints: List[Tuple[str, int, Tensor]] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in endpoint_batches:
                batch.validate()
                vectors = F.normalize(model(pixel_values=batch.pixel_values.to(target_device)), dim=1).detach().float().cpu()
                embeddings.append(vectors)
                variants = batch.variants or ("final",) * len(batch.seeds)
                for vector, prompt_uid, seed, variant in zip(vectors, batch.prompt_uids, batch.seeds, variants):
                    endpoint = (prompt_uid, seed, vector)
                    if variant == "final":
                        final_endpoints.append(endpoint)
                    elif variant == "attack":
                        attack_endpoints.append(endpoint)
    finally:
        model.train(was_training)
    if not embeddings:
        return EmbeddingHealth(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, True)
    values = torch.cat(embeddings, dim=0)
    std = values.std(dim=0, unbiased=False)
    if values.shape[0] < 2:
        effective_rank = 0.0
    else:
        covariance = torch.cov(values.T)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        total = eigenvalues.sum()
        if float(total) <= 0.0:
            effective_rank = 0.0
        else:
            probabilities = eigenvalues[eigenvalues > 0] / total
            effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    same_prompt_different_seed = [
        float(F.cosine_similarity(left[None], right[None]).item())
        for index, (prompt_uid, seed, left) in enumerate(final_endpoints)
        for other_prompt_uid, other_seed, right in final_endpoints[index + 1:]
        if prompt_uid == other_prompt_uid and seed != other_seed
    ]
    final_by_identity = {(prompt_uid, seed): vector for prompt_uid, seed, vector in final_endpoints}
    attack_final = [
        float(F.cosine_similarity(vector[None], final_by_identity[(prompt_uid, seed)][None]).item())
        for prompt_uid, seed, vector in attack_endpoints
        if (prompt_uid, seed) in final_by_identity
    ]
    same_prompt_mean, same_prompt_median = _cosine_summary(same_prompt_different_seed)
    attack_mean, attack_median = _cosine_summary(attack_final)
    collapsed = effective_rank < 2.0 or same_prompt_median > 0.99
    return EmbeddingHealth(
        float(std.mean()), float(std.min()), float(std.max()), effective_rank,
        same_prompt_mean, same_prompt_median,
        attack_mean, attack_median, collapsed,
    )


@torch.no_grad()
def validate_model(
    model: nn.Module, validation: ValidationData, *, negative_cap: float = NEGATIVE_CAP,
    prior_consecutive_collapse_epochs: int = 0,
) -> ValidationResult:
    """Validation-only selection metrics and the two-epoch collapse alarm."""
    device = _device_of(model)
    was_training = model.training
    model.eval()
    losses: List[float] = []
    for batch in validation.pair_batches:
        batch.validate()
        losses.append(float(pair_loss(
            _cosines(model, batch.left_pixel_values.to(device), batch.right_pixel_values.to(device)),
            positive=_positive(batch.kind), negative_cap=negative_cap,
        ).cpu()))
    health = embedding_health(model, validation.endpoint_batches, device=device)
    consecutive = prior_consecutive_collapse_epochs + 1 if health.collapsed else 0
    result = ValidationResult(
        loss=sum(losses) / len(losses) if losses else 0.0,
        attack_seed_ordering_accuracy=_ordering_accuracy(model, validation.attack_seed_orderings, device),
        health=health,
        consecutive_collapse_epochs=consecutive, stop_for_collapse=consecutive >= 2,
    )
    model.train(was_training)
    return result


# --------------------------------------------------------------------------- checkpoints


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


@dataclass
class Checkpoint:
    """A resumable training state: trainable weights, optimizer, scheduler and RNG."""

    version: str
    model_state: Dict[str, Tensor]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    rng_state: Dict[str, Any]
    config: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)
    global_update: int = 0
    completed_epochs: int = 0
    data_manifest_sha256: Optional[str] = None

    @classmethod
    def capture(cls, *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
                history: Sequence[Mapping[str, Any]], global_update: int, completed_epochs: int,
                data_manifest_sha256: Optional[str] = None) -> "Checkpoint":
        config = validate_checkpoint_config(model.checkpoint_config())
        state = dict(model.trainable_state_dict())
        if not state:
            raise ResumeProvenanceError("checkpoint cannot omit trainable adapter/head state")
        return cls(
            CHECKPOINT_VERSION, state, deepcopy(optimizer.state_dict()),
            deepcopy(scheduler.state_dict()) if scheduler is not None else None,
            _rng_state(), deepcopy(config), deepcopy([dict(item) for item in history]),
            global_update, completed_epochs, data_manifest_sha256,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(".%s.staging" % path.name)
        try:
            torch.save(asdict(self), staging)
            staging.replace(path)
        finally:
            if staging.exists():
                staging.unlink()


def load_checkpoint_file(path: Path) -> Checkpoint:
    """Read a checkpoint file (the pickle holds optimizer and RNG state, so verify its hash first)."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ResumeProvenanceError("unsupported checkpoint version")
    known = {name for name in Checkpoint.__dataclass_fields__}
    checkpoint = Checkpoint(**{name: value for name, value in payload.items() if name in known})
    checkpoint.config = validate_checkpoint_config(checkpoint.config)
    return checkpoint


def restore_exact_trainable_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Load adapter/head tensors into a model, requiring an exact match of names and bytes."""
    expected = dict(model.trainable_state_dict())
    if set(state) != set(expected):
        raise ValueError("trainable state does not exactly match the active model")
    incompatible = model.load_state_dict(dict(state), strict=False)
    if set(incompatible.unexpected_keys) or set(incompatible.missing_keys) & set(expected):
        raise ValueError("trainable state could not be restored exactly")
    if trainable_state_sha256(model.trainable_state_dict()) != trainable_state_sha256(state):
        raise ValueError("trainable state changed while restoring tensor bytes")


def restore_checkpoint(checkpoint: Checkpoint, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any]) -> None:
    """Restore trainable state, optimizer, scheduler and RNG after checking the model matches."""
    if model.checkpoint_config() != checkpoint.config:
        raise ResumeProvenanceError("checkpoint config does not match the active adapter/head")
    expected = dict(model.trainable_state_dict())
    if set(checkpoint.model_state) != set(expected):
        raise ResumeProvenanceError("checkpoint trainable state does not exactly match the active adapter/head")
    if (scheduler is None) != (checkpoint.scheduler_state is None):
        raise ResumeProvenanceError("checkpoint scheduler presence does not match requested resume")
    incompatible = model.load_state_dict(checkpoint.model_state, strict=False)
    if set(incompatible.unexpected_keys) or set(incompatible.missing_keys) & set(expected):
        raise ResumeProvenanceError("checkpoint trainable state could not be restored exactly")
    optimizer.load_state_dict(checkpoint.optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint.scheduler_state)
    random.setstate(checkpoint.rng_state["python"])
    torch.set_rng_state(checkpoint.rng_state["torch"])
    if "cuda" in checkpoint.rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint.rng_state["cuda"])


# --------------------------------------------------------------------------- driver


def candidate_seed(training_seed: int, name: str) -> int:
    """Initialization and sampling seed of one candidate configuration."""
    return int.from_bytes(hashlib.sha256(("%s:%s" % (training_seed, name)).encode("utf-8")).digest()[:8], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pixel_tensor(processor: Any, images: Sequence[Any]) -> Tensor:
    """Preprocess PIL images with the DINOv2 image processor."""
    values = processor(images=[image.convert("RGB") for image in images], return_tensors="pt").get("pixel_values")
    if not isinstance(values, Tensor):
        raise RuntimeError("image processor did not return pixel_values")
    return values


def materialize_update(corpus: SourceCorpus, update: Any, processor: Any) -> PairTensorBatch:
    """Read one sampled same-kind update's images and preprocess them."""
    dataset = PairDataset(update.records, data_root=corpus.data_root)
    pairs = [dataset[index] for index in range(len(dataset))]
    return PairTensorBatch(
        update.kind, pixel_tensor(processor, [pair[0] for pair in pairs]), pixel_tensor(processor, [pair[1] for pair in pairs]),
    )


def training_schedule(
    corpus: SourceCorpus, *, epoch: int, seed: int, updates_per_prompt: int = UPDATES_PER_PROMPT,
) -> Tuple[ScheduledPairUpdate, ...]:
    """Globally shuffled, type-balanced updates of one epoch over all training prompts."""
    prompts = sorted({item.prompt_uid for item in corpus.finals if item.prompt_split == "train"})
    samplers = {
        prompt_uid: BalancedPairSampler(
            build_pair_records(corpus.for_prompt(prompt_uid), prompt_split="train", family_partition="train", epoch=epoch)
        )
        for prompt_uid in prompts
    }
    return randomized_balanced_updates(samplers, epoch=epoch, training_seed=seed, updates_per_prompt=updates_per_prompt)


def build_validation_data(
    corpus: SourceCorpus, processor: Any, *, updates_per_prompt: int = UPDATES_PER_PROMPT,
) -> ValidationData:
    """Validation tensors from every validation source of the corpus.

    Pair batches use the validation transformation families. The ordering metric checks
    whether a transformed image is closer to its source than a different-seed generation.
    """
    prompts = sorted({item.prompt_uid for item in corpus.finals if item.prompt_split == "validation"})
    if not prompts:
        raise ValueError("the data has no validation prompts")
    batches: List[PairTensorBatch] = []
    for prompt_uid in prompts:
        prompt_corpus = corpus.for_prompt(prompt_uid)
        sampler = BalancedPairSampler(
            build_pair_records(prompt_corpus, prompt_split="validation", family_partition="validation")
        )
        for update in sampler.iter_updates(epoch=0, update_count=updates_per_prompt):
            batches.append(materialize_update(prompt_corpus, update, processor))
    holder = PairDataset.__new__(PairDataset)
    holder.data_root = corpus.data_root
    cache: Dict[Tuple[str, str], Tensor] = {}

    def tensor_of(identity: Any) -> Tensor:
        key = (identity.source_uid, identity.attack.attack_uid if identity.attack else identity.variant)
        if key not in cache:
            cache[key] = pixel_tensor(processor, [holder.load(identity)])
        return cache[key]

    endpoints: List[Tuple[Tensor, str, int, str]] = []
    attack_examples: List[OrderingExample] = []
    for prompt_uid in prompts:
        prompt_corpus = corpus.for_prompt(prompt_uid)
        finals = list(prompt_corpus.finals)
        attacks: Dict[str, List[Any]] = {}
        for item in prompt_corpus.attacks:
            if item.attack.family_partition == "validation":
                attacks.setdefault(item.source_uid, []).append(item)
        for final in finals:
            endpoints.append((tensor_of(final), prompt_uid, final.seed, "final"))
            for attack in attacks[final.source_uid]:
                endpoints.append((tensor_of(attack), prompt_uid, final.seed, "attack"))
        for final in finals:
            other = next(item for item in finals if item.seed != final.seed)
            anchor, negative = tensor_of(final), tensor_of(other)
            for attack in attacks[final.source_uid]:
                attack_examples.append(OrderingExample(anchor, tensor_of(attack), negative))
    endpoint_batches = []
    for start in range(0, len(endpoints), 32):
        chunk = endpoints[start:start + 32]
        pixels, prompt_uids, seeds, variants = zip(*chunk)
        endpoint_batches.append(EndpointBatch(torch.cat(pixels, dim=0), seeds, prompt_uids, variants))
    return ValidationData(tuple(batches), tuple(attack_examples), tuple(endpoint_batches))


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Order-invariant digest of a source manifest, recorded in checkpoints."""
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["source_uid"]))
    return hashlib.sha256(json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_training(
    *, corpus: SourceCorpus, processor: Any, model: DinoLoRAMetric, device: torch.device,
    output_dir: Path, candidate: CandidateConfiguration, target_updates: int, training_seed: int = TRAINING_SEED,
    negative_cap: float = NEGATIVE_CAP, resume: Optional[Path] = None,
    save_every_epochs: int = 1, data_manifest_sha256: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Train ``model`` until ``target_updates`` optimizer updates (a whole number of epochs).

    One epoch is ``4 * prompts`` updates (four per training prompt, two of each pair type).
    After each epoch (or every ``save_every_epochs``) a checkpoint
    ``checkpoint-step-<updates>.pt`` is written to ``output_dir``, with validation metrics in
    ``history``. Returns the final validation result as a dict.
    """
    name = candidate_name(candidate)
    seed = candidate_seed(training_seed, name)
    seed_everything(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY_PER_EPOCH)
    history: List[Dict[str, Any]] = []
    global_update = completed_epochs = 0
    if resume is not None:
        loaded = load_checkpoint_file(resume)
        if any("preview_ordering_accuracy" in epoch for epoch in loaded.history):
            raise ResumeProvenanceError("start a new run for the paper's two-pair training recipe")
        restore_checkpoint(loaded, model=model, optimizer=optimizer, scheduler=scheduler)
        history, global_update, completed_epochs = list(loaded.history), loaded.global_update, loaded.completed_epochs
    prompts = sorted({item.prompt_uid for item in corpus.finals if item.prompt_split == "train"})
    updates_per_epoch = len(prompts) * UPDATES_PER_PROMPT
    if target_updates <= global_update or (target_updates - global_update) % updates_per_epoch:
        raise ValueError("target updates must be reachable from the current progress in whole epochs of %d updates"
                         % updates_per_epoch)
    validation = build_validation_data(corpus, processor)
    result = None
    for _ in range((target_updates - global_update) // updates_per_epoch):
        started = time.monotonic()
        schedule = training_schedule(corpus, epoch=completed_epochs, seed=seed)
        batches = (materialize_update(corpus, item.update, processor) for item in schedule)
        trained = train_one_epoch(model, batches, optimizer=optimizer, scheduler=scheduler, negative_cap=negative_cap)
        global_update += trained.optimizer_steps
        completed_epochs += 1
        result = validate_model(
            model, validation, negative_cap=negative_cap,
            prior_consecutive_collapse_epochs=history[-1]["consecutive_collapse_epochs"] if history else 0,
        )
        history.append({
            "epoch": completed_epochs - 1, "global_update": global_update, "loss": trained.loss,
            "sampler_observations": dict(trained.kind_counts),
            "embedding_health": asdict(result.health),
            "attack_seed_ordering_accuracy": result.attack_seed_ordering_accuracy,
            "validation_loss": result.loss,
            "consecutive_collapse_epochs": result.consecutive_collapse_epochs,
        })
        if completed_epochs % save_every_epochs == 0 or global_update == target_updates:
            Checkpoint.capture(
                model=model, optimizer=optimizer, scheduler=scheduler, history=history, global_update=global_update,
                completed_epochs=completed_epochs, data_manifest_sha256=data_manifest_sha256,
            ).save(Path(output_dir) / ("checkpoint-step-%08d.pt" % global_update))
        log("epoch %d  updates %d  loss %.4f  validation transformation ordering %.3f  (%.0fs)" % (
            completed_epochs, global_update, trained.loss,
            result.attack_seed_ordering_accuracy, time.monotonic() - started))
        if result.stop_for_collapse:
            raise RuntimeError("collapse alarm fired: embeddings are collapsed on two consecutive epochs")
    assert result is not None
    return {
        "candidate": name, "global_update": global_update, "selection_key": list(result.selection_key),
        "trainable_state_sha256": trainable_state_sha256(model.trainable_state_dict()),
    }
