"""Experiment constants and the loader for ``configs/detection.yaml``.

This is the only module that reads ``configs/detection.yaml``. Every experiment
asks :func:`load_detection_config` for its detector settings instead of
hard-coding N, M, activation, radius, FPR targets, folds, frames or keys.

PyYAML is used when installed; otherwise a small parser for the YAML subset the
shipped configs use (block mappings, block lists, flow mappings and lists,
scalars, comments) takes over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


HYPERPLANE_COUNT = 64

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_DETECTION_PATH = _CONFIG_DIR / "detection.yaml"
DEFAULT_ASSETS_PATH = _CONFIG_DIR / "assets.yaml"


# --------------------------------------------------------------------------- YAML subset


_INT = re.compile(r"^[-+]?\d+$")
_FLOAT = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")


def _scalar(text: str) -> Any:
    text = text.strip()
    if text in ("", "~", "null", "Null", "NULL"):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if _INT.match(text):
        return int(text)
    if _FLOAT.match(text):
        return float(text)
    return text


def _split_top_level(text: str) -> List[str]:
    parts, depth, current = [], 0, []
    for char in text:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def _split_key(text: str) -> Tuple[str, str]:
    index = text.find(":")
    while index != -1 and index + 1 < len(text) and text[index + 1] != " ":
        index = text.find(":", index + 1)
    if index == -1:
        raise ValueError("expected 'key: value' in %r" % text)
    return text[:index].strip(), text[index + 1:].strip()


def _flow(text: str) -> Any:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        result: Dict[str, Any] = {}
        for item in _split_top_level(text[1:-1]):
            key, value = _split_key(item.strip())
            result[key] = _flow(value)
        return result
    if text.startswith("[") and text.endswith("]"):
        return [_flow(item) for item in _split_top_level(text[1:-1])]
    return _scalar(text)


def _strip_comment(line: str) -> str:
    quote: Optional[str] = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line


def _parse_block(lines: Sequence[Tuple[int, str]], start: int, indent: int) -> Tuple[Any, int]:
    if lines[start][1].startswith("- "):
        items: List[Any] = []
        index = start
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            items.append(_flow(lines[index][1][2:]))
            index += 1
        return items, index
    mapping: Dict[str, Any] = {}
    index = start
    while index < len(lines) and lines[index][0] == indent:
        key, value = _split_key(lines[index][1])
        index += 1
        if value:
            mapping[key] = _flow(value)
        elif index < len(lines) and lines[index][0] > indent:
            mapping[key], index = _parse_block(lines, index, lines[index][0])
        else:
            mapping[key] = None
    return mapping, index


def parse_yaml(text: str) -> Any:
    """Parse a config file (PyYAML if available, else the built-in subset parser)."""
    try:
        import yaml  # type: ignore
    except ImportError:
        lines = []
        for raw in text.splitlines():
            stripped = _strip_comment(raw).rstrip()
            if stripped.strip():
                lines.append((len(stripped) - len(stripped.lstrip(" ")), stripped.strip()))
        if not lines:
            return None
        value, end = _parse_block(lines, 0, lines[0][0])
        if end != len(lines):
            raise ValueError("unsupported YAML structure near %r" % lines[end][1])
        return value
    return yaml.safe_load(text)


# --------------------------------------------------------------------------- detection config


@dataclass(frozen=True)
class DetectorSetting:
    """One paper configuration: N candidates, M hash bits, delta activation, Hamming radius."""

    n: int
    m: int
    activation: float
    radius: int


@dataclass(frozen=True)
class DetectionConfig:
    prf: str
    settings: Tuple[DetectorSetting, ...]
    activation_population: str
    calibration_population: str
    folds: int
    fpr_targets: Tuple[float, ...]
    frames: int
    keys: int
    frame_seed_base: int
    set_sizes: Tuple[int, ...]
    aggregation: str
    exceptions: Mapping[str, Mapping[str, Any]]

    def setting(self, n: int, m: int) -> DetectorSetting:
        """Return the configured setting for ``(N, M)``."""
        for item in self.settings:
            if (item.n, item.m) == (n, m):
                return item
        raise KeyError("no detector setting for (N, M) = (%d, %d)" % (n, m))

    def frame_seeds(self) -> Tuple[int, ...]:
        """Hyperplane-frame seeds: ``frame_seed_base + f`` for each of the ``frames`` frames."""
        return tuple(self.frame_seed_base + index for index in range(self.frames))

    def frame_ids(self) -> Tuple[str, ...]:
        """Frame identifiers used inside selector RNG identities (``frame-00`` ...)."""
        return tuple("frame-%02d" % index for index in range(self.frames))

    def key_trials(self) -> Tuple[int, ...]:
        """Key trial indices ``0 .. keys-1`` (see :func:`watermark.prf.derive_trial_key`)."""
        return tuple(range(self.keys))

    def exception(self, name: str) -> Mapping[str, Any]:
        """Settings of a named exception (``closed_api`` or ``style_depth``)."""
        return self.exceptions[name]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("invalid detection config: " + message)


def _positive_int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             "%s must be a positive integer" % name)
    return value


def _unit_fraction(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 < float(value) <= 1.0,
             "%s must lie in (0, 1]" % name)
    return float(value)


def validate_detection_config(raw: Mapping[str, Any]) -> DetectionConfig:
    """Validate a parsed ``detection.yaml`` mapping and return the typed config."""
    _require(isinstance(raw, Mapping), "top level must be a mapping")
    _require(raw.get("prf") == "hmac-sha256", "prf must be hmac-sha256")
    _require(raw.get("activation_population") == "clean_watermarked",
             "activation_population must be clean_watermarked")
    _require(raw.get("aggregation") == "mean", "aggregation must be mean")
    entries = raw.get("configs")
    _require(isinstance(entries, list) and entries, "configs must be a nonempty list")
    settings = []
    for entry in entries:
        _require(isinstance(entry, Mapping) and set(entry) == {"N", "M", "activation", "radius"},
                 "each config needs exactly N, M, activation, radius")
        n, m = _positive_int(entry["N"], "N"), _positive_int(entry["M"], "M")
        _require(m <= HYPERPLANE_COUNT, "M exceeds the hyperplane count")
        radius = entry["radius"]
        _require(isinstance(radius, int) and not isinstance(radius, bool) and 0 <= radius <= m,
                 "radius must lie between 0 and M")
        settings.append(DetectorSetting(n, m, _unit_fraction(entry["activation"], "activation"), radius))
    _require(len({(item.n, item.m) for item in settings}) == len(settings), "duplicate (N, M)")
    calibration = raw.get("calibration")
    _require(isinstance(calibration, Mapping) and calibration.get("population") == "clean",
             "calibration.population must be clean")
    folds = _positive_int(calibration.get("folds"), "calibration.folds")
    _require(folds >= 2, "calibration.folds must be at least 2")
    targets = calibration.get("fpr_targets")
    _require(isinstance(targets, list) and targets, "calibration.fpr_targets must be a nonempty list")
    fpr_targets = tuple(_unit_fraction(value, "fpr target") for value in targets)
    replication = raw.get("replication")
    _require(isinstance(replication, Mapping) and set(replication) == {"frames", "keys", "frame_seed_base"},
             "replication needs frames, keys, frame_seed_base")
    seed_base = replication["frame_seed_base"]
    _require(isinstance(seed_base, int) and not isinstance(seed_base, bool) and seed_base >= 0,
             "frame_seed_base must be a nonnegative integer")
    sizes = raw.get("set_sizes")
    _require(isinstance(sizes, list) and sizes, "set_sizes must be a nonempty list")
    set_sizes = tuple(_positive_int(value, "set size") for value in sizes)
    exceptions = raw.get("exceptions") or {}
    _require(isinstance(exceptions, Mapping), "exceptions must be a mapping")
    return DetectionConfig(
        prf="hmac-sha256", settings=tuple(settings), activation_population="clean_watermarked",
        calibration_population="clean", folds=folds, fpr_targets=fpr_targets,
        frames=_positive_int(replication["frames"], "frames"),
        keys=_positive_int(replication["keys"], "keys"), frame_seed_base=seed_base,
        set_sizes=set_sizes, aggregation="mean",
        exceptions={str(name): dict(value) for name, value in exceptions.items()},
    )


def load_detection_config(path: Optional[Path] = None) -> DetectionConfig:
    """Load and validate ``configs/detection.yaml`` (or ``path``)."""
    source = Path(path) if path is not None else DEFAULT_DETECTION_PATH
    return validate_detection_config(parse_yaml(source.read_text(encoding="utf-8")))


def load_assets(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``configs/assets.yaml``: checkpoint binding and pinned model revisions."""
    source = Path(path) if path is not None else DEFAULT_ASSETS_PATH
    assets = parse_yaml(source.read_text(encoding="utf-8"))
    if not isinstance(assets, dict) or "checkpoint" not in assets or "models" not in assets:
        raise ValueError("invalid assets config: needs checkpoint and models")
    return assets
