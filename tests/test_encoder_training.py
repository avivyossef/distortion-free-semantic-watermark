from __future__ import annotations

import pytest
import torch

from encoder import checkpoint as checkpoint_module
from encoder import data, training
from encoder.model import DinoLoRAMetric
from encoder_helpers import FakeProcessor, TinyDino, tiny_model, write_sources


def _pixels(values):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)


def _same(kind):
    block = _pixels([[1, 0, 0]] * 16)
    return training.PairTensorBatch(kind, block, block.clone())


class MeanMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values):
        return torch.nn.functional.normalize(pixel_values.mean(dim=(-1, -2)) * self.scale, dim=1)


class ConstantMetric(torch.nn.Module):
    def forward(self, pixel_values):
        return torch.nn.functional.normalize(torch.ones(pixel_values.shape[0], 4), dim=1)


def test_updates_need_exactly_sixteen_pairs():
    model = MeanMetric()
    bad = training.PairTensorBatch(data.PairKind.ATTACK_POSITIVE, _pixels([[1, 0, 0]] * 15), _pixels([[1, 0, 0]] * 15))
    with pytest.raises(ValueError, match="exactly 16"):
        training.train_one_epoch(model, [bad], optimizer=torch.optim.AdamW(model.parameters()))


def test_epoch_takes_one_step_per_batch_and_steps_the_scheduler_once():
    model = MeanMetric()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    class Recorder:
        calls = 0

        def step(self):
            self.calls += 1

    scheduler = Recorder()
    batches = [_same(kind) for kind in data.PairKind] * 2
    result = training.train_one_epoch(model, batches, optimizer=optimizer, scheduler=scheduler)
    assert result.optimizer_steps == 8 and result.pair_count == 128 and scheduler.calls == 1
    assert result.kind_counts == {kind.value: 2 for kind in data.PairKind}


def test_training_pulls_positives_together_and_pushes_negatives_apart():
    torch.manual_seed(0)
    model = tiny_model("lora_linear_head")
    left = torch.randn(16, 3, 8, 8)
    right = torch.randn(16, 3, 8, 8)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    def cosine():
        with torch.no_grad():
            model.eval()
            return torch.nn.functional.cosine_similarity(model(pixel_values=left), model(pixel_values=right)).mean().item()

    before = cosine()
    for _ in range(30):
        training.train_one_epoch(model, [training.PairTensorBatch(data.PairKind.ATTACK_POSITIVE, left, right)],
                                 optimizer=optimizer)
    assert cosine() > before + 0.05


def test_validation_selects_ordering_then_loss_and_flags_collapse():
    model = MeanMetric()
    one, two = _pixels([[1, 0, 0]]), _pixels([[0, 1, 0]])
    validation = training.ValidationData(
        pair_batches=(_same(data.PairKind.PREVIEW_POSITIVE),),
        preview_orderings=(training.OrderingExample(one, one, two),),
        attack_seed_orderings=(training.OrderingExample(one, one, two),),
        endpoint_batches=(training.EndpointBatch(_pixels([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (17, 18, 19), ("p",) * 3),),
    )
    result = training.validate_model(model, validation)
    assert result.selection_key == (1.0, 1.0, -result.loss) and result.health.effective_rank >= 2
    constant = ConstantMetric()
    collapsed = training.ValidationData(endpoint_batches=validation.endpoint_batches)
    first = training.validate_model(constant, collapsed)
    second = training.validate_model(constant, collapsed, prior_consecutive_collapse_epochs=first.consecutive_collapse_epochs)
    assert first.health.collapsed and not first.stop_for_collapse and second.stop_for_collapse


def test_candidate_grid_is_the_five_ablation_configurations():
    grid = [training.CandidateConfiguration("head_only", 4),
            *(training.CandidateConfiguration(v, r) for v in ("lora_linear_head", "lora_mlp") for r in (4, 8))]
    names = [training.candidate_name(c) for c in grid]
    assert names == ["head_only-rank4", "lora_linear_head-rank4", "lora_linear_head-rank8",
                     "lora_mlp-rank4", "lora_mlp-rank8"]
    assert training.candidate_seed(1, "x") != training.candidate_seed(1, "y")


def _setup(tmp_path):
    split = lambda uid: "validation" if uid == "v" else "train"
    rows = write_sources(tmp_path, ["a", "b", "v"], 3, split_of=split)
    return rows, data.SourceCorpus.from_rows(rows, data_root=tmp_path)


def test_run_training_writes_resumable_checkpoints(tmp_path):
    rows, corpus = _setup(tmp_path)
    out = tmp_path / "run"
    candidate = training.CandidateConfiguration("lora_mlp", 4)
    # Two training prompts: one epoch is 8 updates.
    summary = training.run_training(
        corpus=corpus, processor=FakeProcessor(), model=tiny_model(), device=torch.device("cpu"), output_dir=out,
        candidate=candidate, target_updates=16, data_manifest_sha256=training.manifest_sha256(rows), log=lambda _: None)
    assert summary["global_update"] == 16 and len(summary["selection_key"]) == 3
    files = sorted(p.name for p in out.glob("*.pt"))
    assert files == ["checkpoint-step-00000008.pt", "checkpoint-step-00000016.pt"]
    saved = training.load_checkpoint_file(out / files[-1])
    assert saved.global_update == 16 and saved.completed_epochs == 2 and len(saved.history) == 2
    assert saved.data_manifest_sha256 == training.manifest_sha256(rows)
    assert saved.config["lora_rank"] == 4 and set(saved.model_state) == set(tiny_model().trainable_state_dict())
    # Resume from the first epoch and reach the same state as the uninterrupted run.
    resumed = tmp_path / "resumed"
    again = training.run_training(
        corpus=corpus, processor=FakeProcessor(), model=tiny_model(), device=torch.device("cpu"), output_dir=resumed,
        candidate=candidate, target_updates=16, resume=out / files[0], log=lambda _: None)
    assert again["trainable_state_sha256"] == summary["trainable_state_sha256"]
    with pytest.raises(ValueError, match="whole epochs"):
        training.run_training(
            corpus=corpus, processor=FakeProcessor(), model=tiny_model(), device=torch.device("cpu"),
            output_dir=tmp_path / "bad", candidate=candidate, target_updates=12, log=lambda _: None)


def test_restore_rejects_a_mismatched_model(tmp_path):
    model = tiny_model("lora_mlp")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    ckpt = training.Checkpoint.capture(model=model, optimizer=optimizer, scheduler=None, history=[],
                                       global_update=0, completed_epochs=0)
    path = tmp_path / "c.pt"
    ckpt.save(path)
    loaded = training.load_checkpoint_file(path)
    other = tiny_model("lora_linear_head")
    with pytest.raises(training.ResumeProvenanceError):
        training.restore_checkpoint(loaded, model=other, optimizer=torch.optim.AdamW(
            [p for p in other.parameters() if p.requires_grad]), scheduler=None)
    forged = tmp_path / "forged.pt"
    torch.save({**vars(loaded), "version": "other"}, forged)
    with pytest.raises(training.ResumeProvenanceError):
        training.load_checkpoint_file(forged)


def test_load_encoder_verifies_the_hash_and_restores_exact_weights(tmp_path):
    model = tiny_model()
    model.metric_head[0].weight.data.fill_(0.25)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    path = tmp_path / "ckpt.pt"
    training.Checkpoint.capture(model=model, optimizer=optimizer, scheduler=None, history=[],
                                global_update=7, completed_epochs=1).save(path)
    digest = checkpoint_module.file_sha256(path)
    with pytest.raises(ValueError, match="SHA-256"):
        checkpoint_module.load_encoder(path, expected_sha256="0" * 64, model=tiny_model(), processor=object())
    fresh = tiny_model()
    assert not torch.equal(fresh.metric_head[0].weight, model.metric_head[0].weight)
    loaded, processor, provenance = checkpoint_module.load_encoder(
        path, expected_sha256=digest, model=fresh, processor="processor")
    assert torch.equal(loaded.metric_head[0].weight, model.metric_head[0].weight)
    assert not loaded.training and processor == "processor"
    assert provenance["checkpoint_sha256"] == digest and provenance["global_update"] == 7
    with pytest.raises(ValueError, match="configuration"):
        checkpoint_module.load_encoder(path, expected_sha256=digest, model=tiny_model("lora_linear_head"),
                                       processor=object())
