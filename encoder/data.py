"""Training pairs for the semantic encoder.

Sources are SDXL images: for every (prompt, seed) a final image and the preview decoded
from the fifth denoising step (see :mod:`encoder.sdxl`). Four pair types are balanced across
optimizer updates, each update holding 16 pairs of one type:

- ``attack_positive``: a final image and a transformed copy of it (one cell of the attack
  registry, :mod:`encoder.attacks`);
- ``preview_positive``: the fifth-step preview and the final image of the same seed;
- ``final_seed_negative``: finals of two different seeds of the same prompt;
- ``preview_seed_negative``: the preview of one seed and the final of another seed of the
  same prompt.

A source manifest is a JSONL file with one row per source image::

    {"source_uid": "<prompt_uid>:seed_<seed:02d>", "prompt_uid", "prompt_split": "train" |
     "validation", "seed", "final_path", "preview_path"}      # paths relative to the data root
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from .attacks import ATTACK_REGISTRY, AttackChoice, attack_choice, render_attack


class LeakageError(ValueError):
    """Raised before an invalid split or malformed source reaches training."""


class PairKind(str, Enum):
    ATTACK_POSITIVE = "attack_positive"
    PREVIEW_POSITIVE = "preview_positive"
    FINAL_SEED_NEGATIVE = "final_seed_negative"
    PREVIEW_SEED_NEGATIVE = "preview_seed_negative"


@dataclass(frozen=True)
class AttackRef:
    """One attack-registry cell applied to one source image."""

    attack_uid: str
    family: str
    family_partition: str
    severity: str
    choice: AttackChoice


@dataclass(frozen=True)
class ImageIdentity:
    source_uid: str
    prompt_uid: str
    prompt_split: str
    seed: int
    variant: str  # "final", "preview" or "attack"
    path: Optional[str]  # relative to the data root; attacks are rendered from ``source_path``
    source_path: Optional[str] = None
    attack: Optional[AttackRef] = None


@dataclass(frozen=True)
class PairRecord:
    kind: PairKind
    left: ImageIdentity
    right: ImageIdentity
    target_similarity: float


@dataclass(frozen=True)
class SamplerUpdate:
    epoch: int
    update_index: int
    kind: PairKind
    records: Tuple[PairRecord, ...]


@dataclass(frozen=True)
class ScheduledPairUpdate:
    """One globally shuffled optimizer update and its source-prompt queue."""

    prompt_uid: str
    update: SamplerUpdate


@dataclass(frozen=True)
class SourceCorpus:
    """Normalized source identities; no image bytes are read while indexing."""

    data_root: Path
    finals: Tuple[ImageIdentity, ...]
    previews: Tuple[ImageIdentity, ...]
    attacks: Tuple[ImageIdentity, ...]

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], *, data_root: Path) -> "SourceCorpus":
        if not rows:
            raise LeakageError("source manifest is empty")
        finals: List[ImageIdentity] = []
        previews: List[ImageIdentity] = []
        attacks: List[ImageIdentity] = []
        seen = set()
        for row in rows:
            try:
                source_uid, prompt_uid = str(row["source_uid"]), str(row["prompt_uid"])
                prompt_split, seed = str(row["prompt_split"]), int(row["seed"])
                final_path, preview_path = str(row["final_path"]), str(row["preview_path"])
            except (KeyError, TypeError, ValueError) as error:
                raise LeakageError("source manifest row is missing a field") from error
            if source_uid != "%s:seed_%02d" % (prompt_uid, seed) or prompt_split not in {"train", "validation"}:
                raise LeakageError("source identity is invalid")
            if seed < 16:
                raise LeakageError("sources require seed >= 16")
            if source_uid in seen:
                raise LeakageError("source manifest contains duplicate source identities")
            if Path(final_path).is_absolute() or Path(preview_path).is_absolute():
                raise LeakageError("source paths must be relative to the data root")
            seen.add(source_uid)
            finals.append(ImageIdentity(source_uid, prompt_uid, prompt_split, seed, "final", final_path))
            previews.append(ImageIdentity(source_uid, prompt_uid, prompt_split, seed, "preview", preview_path))
            for family in sorted(ATTACK_REGISTRY):
                specification = ATTACK_REGISTRY[family]
                for severity in sorted(specification.severities):
                    reference = AttackRef(
                        "%s:%s:%s" % (source_uid, family, severity), family, specification.partition,
                        severity, attack_choice(source_uid, family, severity),
                    )
                    attacks.append(ImageIdentity(
                        source_uid, prompt_uid, prompt_split, seed, "attack", None,
                        source_path=final_path, attack=reference,
                    ))
        return cls(
            data_root=Path(data_root).resolve(strict=False),
            finals=tuple(sorted(finals, key=lambda item: item.source_uid)),
            previews=tuple(sorted(previews, key=lambda item: item.source_uid)),
            attacks=tuple(sorted(attacks, key=lambda item: item.attack.attack_uid if item.attack else "")),
        )

    def for_prompt(self, prompt_uid: str) -> "SourceCorpus":
        """The sub-corpus of one prompt."""
        return SourceCorpus(
            self.data_root,
            tuple(item for item in self.finals if item.prompt_uid == prompt_uid),
            tuple(item for item in self.previews if item.prompt_uid == prompt_uid),
            tuple(item for item in self.attacks if item.prompt_uid == prompt_uid),
        )


def _select_other(
    sources: Sequence[ImageIdentity], *, epoch: int, kind: PairKind, source: ImageIdentity
) -> ImageIdentity:
    digest = hashlib.sha256(
        ("%s\0%s\0%s\0%s" % (source.prompt_uid, source.seed, epoch, kind.value)).encode("utf-8")
    ).digest()
    position = int.from_bytes(digest, "big") % len(sources)
    for offset in range(len(sources)):
        candidate = sources[(position + offset) % len(sources)]
        if candidate.seed != source.seed:
            return candidate
    raise LeakageError("same-prompt negative requires another seed")


def build_pair_records(
    corpus: SourceCorpus, *, prompt_split: str, family_partition: str, epoch: int = 0
) -> Tuple[PairRecord, ...]:
    """Build typed pair records without opening any image file."""
    if prompt_split not in {"train", "validation"}:
        raise LeakageError("training data never reads other prompt splits")
    if family_partition != prompt_split:
        raise LeakageError("prompt split and attack-family partition must match")
    finals = [item for item in corpus.finals if item.prompt_split == prompt_split]
    previews = {item.source_uid: item for item in corpus.previews if item.prompt_split == prompt_split}
    attacks_by_source: Dict[str, List[ImageIdentity]] = defaultdict(list)
    for attack in corpus.attacks:
        if attack.prompt_split == prompt_split and attack.attack and attack.attack.family_partition == family_partition:
            attacks_by_source[attack.source_uid].append(attack)
    finals_by_prompt: Dict[str, List[ImageIdentity]] = defaultdict(list)
    for final in finals:
        finals_by_prompt[final.prompt_uid].append(final)
    if not finals:
        raise LeakageError("no sources are eligible for the requested prompt split")
    prompt_source_counts = {prompt: len(items) for prompt, items in finals_by_prompt.items()}
    if len(set(prompt_source_counts.values())) != 1:
        raise LeakageError("prompts must have equal eligible source counts")
    records: List[PairRecord] = []
    for final in finals:
        preview = previews.get(final.source_uid)
        eligible = sorted(finals_by_prompt[final.prompt_uid], key=lambda item: item.seed)
        if preview is None:
            raise LeakageError("eligible clean source lacks its preview")
        if len(eligible) < 2:
            raise LeakageError("each eligible prompt requires at least two seeds")
        attacks = sorted(attacks_by_source[final.source_uid], key=lambda item: item.attack.attack_uid if item.attack else "")
        records.extend(PairRecord(PairKind.ATTACK_POSITIVE, final, attack, 1.0) for attack in attacks)
        records.append(PairRecord(PairKind.PREVIEW_POSITIVE, preview, final, 1.0))
        records.append(PairRecord(
            PairKind.FINAL_SEED_NEGATIVE, final,
            _select_other(eligible, epoch=epoch, kind=PairKind.FINAL_SEED_NEGATIVE, source=final), 0.0,
        ))
        records.append(PairRecord(
            PairKind.PREVIEW_SEED_NEGATIVE, preview,
            _select_other(eligible, epoch=epoch, kind=PairKind.PREVIEW_SEED_NEGATIVE, source=final), 0.0,
        ))
    return tuple(records)


class PairDataset:
    """PIL-backed dataset; attack images are rendered on the fly from their final image."""

    def __init__(self, records: Sequence[PairRecord], *, data_root: Path) -> None:
        self.records = tuple(records)
        if not self.records:
            raise LeakageError("pair dataset cannot be empty")
        self.data_root = Path(data_root).resolve(strict=False)

    def __len__(self) -> int:
        return len(self.records)

    def _open(self, relative: str) -> Image.Image:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise LeakageError("image path must be relative to the data root")
        with Image.open(self.data_root / path) as opened:
            return opened.convert("RGB").copy()

    def load(self, identity: ImageIdentity) -> Image.Image:
        if identity.variant != "attack":
            if not identity.path:
                raise LeakageError("image identity lacks a path")
            return self._open(identity.path)
        if identity.attack is None or not identity.source_path:
            raise LeakageError("attack identity is incomplete")
        return render_attack(self._open(identity.source_path), identity.attack.choice)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image, float]:
        record = self.records[index]
        return self.load(record.left), self.load(record.right), record.target_similarity


def _record_key(record: PairRecord) -> Tuple[str, str, str]:
    right_attack = record.right.attack.attack_uid if record.right.attack else ""
    return record.left.source_uid, record.right.source_uid, right_attack


def _interleave_attack_cells(records: Sequence[PairRecord]) -> Tuple[PairRecord, ...]:
    """Cycle registry cells so every 16-pair attack update is near-uniform."""
    cells: Dict[Tuple[str, str], List[PairRecord]] = defaultdict(list)
    for record in records:
        if record.right.attack is None:
            raise LeakageError("attack-positive record lacks attack provenance")
        cells[(record.right.attack.family, record.right.attack.severity)].append(record)
    ordered_cells = sorted(cells)
    queues = {cell: sorted(cells[cell], key=_record_key) for cell in ordered_cells}
    if len({len(queue) for queue in queues.values()}) != 1:
        raise LeakageError("attack family/severity cells must have equal record counts")
    return tuple(
        queues[cell][position]
        for position in range(len(next(iter(queues.values()))))
        for cell in ordered_cells
    )


class BalancedPairSampler:
    """Four deterministic queues; every update has one kind and 16 pair records."""

    def __init__(self, records: Sequence[PairRecord], *, pairs_per_update: int = 16) -> None:
        if pairs_per_update != 16:
            raise LeakageError("optimizer updates must contain exactly 16 pairs")
        queues: Dict[PairKind, Tuple[PairRecord, ...]] = {}
        for kind in PairKind:
            queue = tuple(record for record in records if record.kind is kind)
            if not queue:
                raise LeakageError("enabled pair type is empty: %s" % kind.value)
            queues[kind] = (
                _interleave_attack_cells(queue)
                if kind is PairKind.ATTACK_POSITIVE
                else tuple(sorted(queue, key=_record_key))
            )
        self._queues = queues
        self.pairs_per_update = pairs_per_update

    def sample_update(self, *, epoch: int, update_index: int, kind: PairKind, nonce: int) -> SamplerUpdate:
        """Draw a reproducibly randomized same-kind update from one prompt queue."""
        if update_index < 0:
            raise LeakageError("update index must be nonnegative")
        if not isinstance(nonce, int):
            raise LeakageError("sampler nonce must be an integer")
        seed = int.from_bytes(
            hashlib.sha256(("%s\0%s\0%s\0%s" % (epoch, kind.value, update_index, nonce)).encode("utf-8")).digest(), "big"
        )
        generator = random.Random(seed)
        queue = self._queues[kind]
        if kind is PairKind.ATTACK_POSITIVE:
            cells: Dict[Tuple[str, str], List[PairRecord]] = defaultdict(list)
            for record in queue:
                assert record.right.attack is not None  # constructor queue invariant
                cells[(record.right.attack.family, record.right.attack.severity)].append(record)
            cell_order = sorted(cells)
            start = generator.randrange(len(cell_order))
            chosen_cells = [cell_order[(start + position) % len(cell_order)] for position in range(self.pairs_per_update)]
            batch = tuple(generator.choice(cells[cell]) for cell in chosen_cells)
        else:
            batch = tuple(generator.choice(queue) for _ in range(self.pairs_per_update))
        return SamplerUpdate(epoch, update_index, kind, batch)

    def iter_updates(self, *, epoch: int, update_count: int):
        """Per-prompt type-balanced updates (used by validation)."""
        if update_count < 0:
            raise LeakageError("update count must be nonnegative")
        kinds = tuple(PairKind)
        offset = int.from_bytes(hashlib.sha256(str(epoch).encode("ascii")).digest(), "big") % len(kinds)
        for update_index in range(update_count):
            kind = kinds[(offset + update_index) % len(kinds)]
            yield self.sample_update(epoch=epoch, update_index=update_index, kind=kind, nonce=0)


def randomized_balanced_updates(
    samplers: Mapping[str, BalancedPairSampler], *, epoch: int, training_seed: int, updates_per_prompt: int = 4,
) -> Tuple[ScheduledPairUpdate, ...]:
    """Randomize prompt/pair choice while retaining exact per-epoch coverage."""
    if updates_per_prompt <= 0 or updates_per_prompt % len(PairKind):
        raise LeakageError("updates per prompt must be a positive multiple of the four pair kinds")
    if not samplers:
        raise LeakageError("randomized schedule requires at least one prompt sampler")
    prompt_uids = tuple(sorted(samplers))
    seed_material = ("%s\0%s\0%s\0%s" % (training_seed, epoch, ",".join(prompt_uids), updates_per_prompt)).encode("utf-8")
    generator = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest(), "big"))
    prompt_kind_updates = [
        (prompt_uid, kind)
        for prompt_uid in prompt_uids
        for _ in range(updates_per_prompt // len(PairKind))
        for kind in PairKind
    ]
    generator.shuffle(prompt_kind_updates)
    return tuple(
        ScheduledPairUpdate(
            prompt_uid=prompt_uid,
            update=samplers[prompt_uid].sample_update(
                epoch=epoch, update_index=update_index, kind=kind, nonce=generator.getrandbits(128),
            ),
        )
        for update_index, (prompt_uid, kind) in enumerate(prompt_kind_updates)
    )
