from __future__ import annotations

import copy

import pytest

from watermark import config


def test_shipped_detection_config_matches_the_paper_settings():
    cfg = config.load_detection_config()
    assert [(s.n, s.m, s.activation, s.radius) for s in cfg.settings] == [
        (8, 4, 0.10, 1), (16, 8, 0.50, 1), (32, 12, 0.75, 1)]
    assert cfg.activation_population == "clean_watermarked"
    assert (cfg.folds, cfg.fpr_targets) == (5, (0.01, 0.05, 0.20))
    assert (cfg.frames, cfg.keys) == (5, 5)
    assert cfg.set_sizes == (1, 5, 10) and cfg.aggregation == "mean"
    assert cfg.setting(16, 8).activation == 0.5
    assert cfg.exception("closed_api")["radius"] == 0 and cfg.exception("closed_api")["M"] == 8
    assert cfg.exception("style_depth")["calibration_population"] == "evaluation"


def test_frame_and_key_replication_identifiers():
    cfg = config.load_detection_config()
    assert cfg.frame_seeds() == tuple(cfg.frame_seed_base + i for i in range(5))
    assert cfg.frame_ids() == ("frame-00", "frame-01", "frame-02", "frame-03", "frame-04")
    assert cfg.key_trials() == (0, 1, 2, 3, 4)
    with pytest.raises(KeyError):
        cfg.setting(64, 32)


def test_builtin_yaml_parser_handles_the_subset_used_by_the_configs():
    text = """
# comment
a: 1
b: {x: 2, y: [1, 2.5, three], z: {k: true}}   # trailing comment
c:
  - {N: 8, M: 4}
  - {N: 16, M: 8}
d:
  e: hello
  f:
    g: null
url: https://example.org/a#b
hash: 5e252aa4af18e90bb93c65e6dee80e98070ceea8523e777f0b0dee6b6d41c614
"""
    parsed = config.parse_yaml(text)
    assert parsed == {
        "a": 1, "b": {"x": 2, "y": [1, 2.5, "three"], "z": {"k": True}},
        "c": [{"N": 8, "M": 4}, {"N": 16, "M": 8}], "d": {"e": "hello", "f": {"g": None}},
        "url": "https://example.org/a#b",
        "hash": "5e252aa4af18e90bb93c65e6dee80e98070ceea8523e777f0b0dee6b6d41c614",
    }


def _raw():
    return {
        "prf": "hmac-sha256", "configs": [{"N": 8, "M": 4, "activation": 0.1, "radius": 1}],
        "activation_population": "clean_watermarked",
        "calibration": {"population": "clean", "folds": 5, "fpr_targets": [0.01, 0.2]},
        "replication": {"frames": 5, "keys": 5, "frame_seed_base": 7}, "set_sizes": [1],
        "aggregation": "mean", "exceptions": {},
    }


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.update(prf="sha1"),
    lambda raw: raw.update(aggregation="max"),
    lambda raw: raw.update(activation_population="clean"),
    lambda raw: raw["configs"][0].update(radius=5),
    lambda raw: raw["configs"][0].update(activation=0.0),
    lambda raw: raw["configs"][0].update(M=65),
    lambda raw: raw["configs"].append(dict(raw["configs"][0])),
    lambda raw: raw["calibration"].update(folds=1),
    lambda raw: raw["calibration"].update(population="evaluation"),
    lambda raw: raw["replication"].pop("keys"),
    lambda raw: raw.update(set_sizes=[]),
])
def test_validation_rejects_malformed_configs(mutate):
    raw = _raw()
    assert config.validate_detection_config(copy.deepcopy(raw)).folds == 5
    mutate(raw)
    with pytest.raises(ValueError):
        config.validate_detection_config(raw)


def test_assets_bind_the_checkpoint_hash_and_pinned_revisions():
    assets = config.load_assets()
    assert assets["checkpoint"]["sha256"] == "5e252aa4af18e90bb93c65e6dee80e98070ceea8523e777f0b0dee6b6d41c614"
    for name in ("dinov2_base", "clip_vit_b32", "dinov3_l16", "csd_vit_l", "depth_anything_v2_small",
                 "tpips_embed", "sdxl", "sd21"):
        revision = assets["models"][name]["revision"]
        assert len(revision) == 40 and set(revision) <= set("0123456789abcdef")
