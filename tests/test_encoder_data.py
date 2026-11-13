from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from PIL import Image

from encoder import attacks, data, sdxl
from encoder_helpers import write_sources


def _rows(tmp_path, prompts=("a", "b"), seeds=3, split="train"):
    return write_sources(tmp_path, prompts, seeds, split_of=lambda _: split)


def test_attack_registry_has_six_training_and_two_validation_families():
    partitions = Counter(spec.partition for spec in attacks.ATTACK_REGISTRY.values())
    assert partitions == {"train": 6, "validation": 2}
    assert all(set(spec.severities) == {"mild", "medium", "severe"} for spec in attacks.ATTACK_REGISTRY.values())


@pytest.mark.parametrize("family", sorted(attacks.ATTACK_REGISTRY))
@pytest.mark.parametrize("severity", ["mild", "severe"])
def test_every_attack_is_deterministic_and_preserves_size(family, severity):
    source = Image.fromarray(np.random.default_rng(1).integers(0, 256, size=(48, 48, 3), dtype=np.uint8))
    choice = attacks.attack_choice("p:seed_16", family, severity)
    first, second = attacks.render_attack(source, choice), attacks.render_attack(source, choice)
    assert first.size == source.size and np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(source))


def test_attack_seed_depends_on_source_family_and_severity():
    base = attacks.attack_seed("p:seed_16", "affine", "mild")
    assert base == attacks.attack_seed("p:seed_16", "affine", "mild")
    assert len({base, attacks.attack_seed("p:seed_17", "affine", "mild"),
                attacks.attack_seed("p:seed_16", "affine", "severe"),
                attacks.attack_seed("p:seed_16", "photometric", "mild")}) == 4


def test_render_attack_rejects_parameters_that_differ_from_the_registry():
    choice = attacks.attack_choice("p:seed_16", "gaussian_blur", "mild")
    forged = attacks.AttackChoice(choice.family, choice.severity, {"radius": 9.0}, choice.deterministic_seed)
    with pytest.raises(ValueError):
        attacks.render_attack(Image.new("RGB", (8, 8)), forged)


def test_corpus_builds_one_final_and_all_attack_cells_per_source(tmp_path):
    corpus = data.SourceCorpus.from_rows(_rows(tmp_path), data_root=tmp_path)
    assert len(corpus.finals) == 6
    assert len(corpus.attacks) == 6 * 24  # 8 families x 3 severities
    assert {item.attack.family_partition for item in corpus.attacks} == {"train", "validation"}


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(source_uid="wrong"),
    lambda row: row.update(prompt_split="final_test"),
    lambda row: row.update(seed=3, source_uid="a:seed_03"),
    lambda row: row.update(final_path="/abs/path.png"),
])
def test_corpus_rejects_malformed_rows(tmp_path, mutate):
    rows = _rows(tmp_path)
    mutate(rows[0])
    with pytest.raises(data.LeakageError):
        data.SourceCorpus.from_rows(rows, data_root=tmp_path)
    with pytest.raises(data.LeakageError):
        data.SourceCorpus.from_rows([], data_root=tmp_path)


def test_pair_records_cover_only_the_two_paper_types_for_every_source(tmp_path):
    rows = _rows(tmp_path)
    rows[0]["preview_path"] = "unused-legacy-preview.png"
    corpus = data.SourceCorpus.from_rows(rows, data_root=tmp_path)
    records = data.build_pair_records(corpus, prompt_split="train", family_partition="train")
    counts = Counter(record.kind for record in records)
    assert counts == {data.PairKind.ATTACK_POSITIVE: 6 * 18, data.PairKind.FINAL_SEED_NEGATIVE: 6}
    for record in records:
        assert record.target_similarity == (1.0 if "positive" in record.kind.value else 0.0)
        if "negative" in record.kind.value:
            assert record.left.variant == record.right.variant == "final"
            assert record.left.prompt_uid == record.right.prompt_uid and record.left.seed != record.right.seed
        else:
            assert record.left.variant == "final" and record.right.variant == "attack"
            assert record.left.source_uid == record.right.source_uid
    assert {r.right.attack.family_partition for r in records if r.right.attack} == {"train"}
    with pytest.raises(data.LeakageError):
        data.build_pair_records(corpus, prompt_split="train", family_partition="validation")
    with pytest.raises(data.LeakageError):
        data.build_pair_records(corpus, prompt_split="final_test", family_partition="final_test")


def test_negatives_need_at_least_two_seeds(tmp_path):
    corpus = data.SourceCorpus.from_rows(_rows(tmp_path, seeds=1), data_root=tmp_path)
    with pytest.raises(data.LeakageError):
        data.build_pair_records(corpus, prompt_split="train", family_partition="train")


def test_sampler_updates_hold_sixteen_pairs_of_one_kind_and_are_reproducible(tmp_path):
    corpus = data.SourceCorpus.from_rows(_rows(tmp_path), data_root=tmp_path)
    sampler = data.BalancedPairSampler(data.build_pair_records(corpus, prompt_split="train", family_partition="train"))
    updates = list(sampler.iter_updates(epoch=0, update_count=8))
    assert Counter(update.kind for update in updates) == {kind: 4 for kind in data.PairKind}
    assert all(len(update.records) == 16 and {r.kind for r in update.records} == {update.kind} for update in updates)
    again = list(sampler.iter_updates(epoch=0, update_count=8))
    assert [u.records for u in updates] == [u.records for u in again]
    attack = next(u for u in updates if u.kind is data.PairKind.ATTACK_POSITIVE)
    assert len({(r.right.attack.family, r.right.attack.severity) for r in attack.records}) == 16
    with pytest.raises(data.LeakageError):
        data.BalancedPairSampler([], pairs_per_update=16)
    with pytest.raises(data.LeakageError):
        data.BalancedPairSampler(sampler._queues[data.PairKind.ATTACK_POSITIVE], pairs_per_update=8)


def test_global_schedule_balances_prompts_and_kinds(tmp_path):
    corpus = data.SourceCorpus.from_rows(_rows(tmp_path, prompts=("a", "b", "c")), data_root=tmp_path)
    samplers = {uid: data.BalancedPairSampler(data.build_pair_records(
        corpus.for_prompt(uid), prompt_split="train", family_partition="train")) for uid in "abc"}
    schedule = data.randomized_balanced_updates(samplers, epoch=2, training_seed=5, updates_per_prompt=4)
    assert len(schedule) == 12
    assert Counter(item.prompt_uid for item in schedule) == {"a": 4, "b": 4, "c": 4}
    assert Counter(item.update.kind for item in schedule) == {kind: 6 for kind in data.PairKind}
    assert all(r.left.prompt_uid == item.prompt_uid for item in schedule for r in item.update.records)
    assert [i.update for i in schedule] == [i.update for i in data.randomized_balanced_updates(
        samplers, epoch=2, training_seed=5, updates_per_prompt=4)]
    other = data.randomized_balanced_updates(samplers, epoch=3, training_seed=5, updates_per_prompt=4)
    assert [i.prompt_uid for i in schedule] != [i.prompt_uid for i in other]
    with pytest.raises(data.LeakageError):
        data.randomized_balanced_updates(samplers, epoch=0, training_seed=0, updates_per_prompt=3)


def test_dataset_loads_files_and_renders_attacks_from_the_final_image(tmp_path):
    corpus = data.SourceCorpus.from_rows(_rows(tmp_path), data_root=tmp_path)
    records = data.build_pair_records(corpus, prompt_split="train", family_partition="train")
    dataset = data.PairDataset(records, data_root=tmp_path)
    attack_index = next(i for i, r in enumerate(records) if r.kind is data.PairKind.ATTACK_POSITIVE)
    left, right, target = dataset[attack_index]
    assert left.size == right.size == (32, 32) and target == 1.0
    assert not np.array_equal(np.asarray(left), np.asarray(right))
    with pytest.raises(data.LeakageError):
        data.PairDataset([], data_root=tmp_path)
    escaped = data.ImageIdentity("a:seed_16", "a", "train", 16, "final", "../outside.png")
    with pytest.raises(data.LeakageError):
        dataset.load(escaped)


def test_seed_allocation_is_disjoint_reproducible_and_prefix_stable():
    prompts = ["p%d" % i for i in range(5)]
    first = sdxl.allocate_seeds(prompts, registry_seed=11, seeds_per_prompt=8)
    assert first == sdxl.allocate_seeds(list(reversed(prompts)), registry_seed=11, seeds_per_prompt=8)
    flat = [seed for seeds in first.values() for seed in seeds]
    assert len(flat) == len(set(flat)) == 40 and min(flat) >= 16 and max(flat) < 2 ** 63
    assert first != sdxl.allocate_seeds(prompts, registry_seed=12, seeds_per_prompt=8)
    # A larger stage only appends for the first prompt; later prompts shift, so extend per stage via the prefix.
    assert sdxl.allocate_seeds(prompts, registry_seed=11, seeds_per_prompt=16)["p0"][:8] == first["p0"]
    assert sdxl.source_uid("p0", 16) == "p0:seed_16"
    with pytest.raises(ValueError):
        sdxl.allocate_seeds(["a", "a"], registry_seed=1)


def test_sdxl_shard_writes_only_final_images_and_reuses_them(tmp_path):
    class Generator:
        calls = []

        def generate(self, prompt, seed):
            self.calls.append((prompt, seed))
            return Image.new("RGB", (8, 8), "red")

    generator = Generator()
    settings = dict(prompt_uid="p", prompt_split="train", text="a red image", seeds=[16], output_dir=tmp_path)
    rows = sdxl.write_prompt_shard(generator, **settings)
    assert rows == [{
        "source_uid": "p:seed_16", "prompt_uid": "p", "prompt_split": "train", "seed": 16,
        "final_path": "p/final/seed_16.png",
    }]
    assert list(tmp_path.rglob("*.png")) == [tmp_path / rows[0]["final_path"]]
    assert sdxl.write_prompt_shard(generator, **settings) == rows
    assert generator.calls == [("a red image", 16)]
