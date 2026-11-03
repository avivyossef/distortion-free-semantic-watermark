"""Release hardening: pinned weight hashing before load, path containment, URL and checkout checks (CPU)."""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "baselines"))

import _baselines_lib as lib  # noqa: E402
import detection_helpers  # noqa: E402,F401
import detection_corpus as corpus  # noqa: E402
from watermark import generation  # noqa: E402
from watermark.io import contained_path, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_contained_path_rejects_escapes(tmp_path):
    assert contained_path(tmp_path, "a/b.png") == (tmp_path / "a" / "b.png").resolve()
    for bad in ("/etc/passwd", "../x.png", "a/../../x.png", "", None):
        with pytest.raises(ValueError):
            contained_path(tmp_path, bad)


def test_panel_images_reject_manifest_escape(tmp_path):
    panel = {"prompt_uid": "p", "batch_slots": [
        {"slot": 1, "clean": {"relative_path": "../../evil.png"}, "candidate_images": []}]}
    with pytest.raises(ValueError):
        corpus.panel_images(tmp_path, panel)


def test_sd21_snapshot_hash_checked_before_load(tmp_path, monkeypatch):
    unet = tmp_path / "unet" / "diffusion_pytorch_model.bin"
    unet.parent.mkdir()
    unet.write_bytes(b"weights")
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_download))
    model = {"repo": "r/m", "revision": "abc", "unet_sha256": sha256_file(unet)}
    assert generation.verified_sd21_snapshot(model) == tmp_path
    assert calls["revision"] == "abc"
    with pytest.raises(ValueError):
        generation.verified_sd21_snapshot(dict(model, unet_sha256="0" * 64))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://example.org/x", "ftp://example.org/x"])
def test_download_checkpoint_requires_https(tmp_path, monkeypatch, url):
    spec = importlib.util.spec_from_file_location("download_checkpoint", ROOT / "scripts" / "download_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_assets", lambda path=None: {"checkpoint": {"url": url, "sha256": "0", "file": "f"}})
    monkeypatch.setattr(sys, "argv", ["x", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="https"):
        module.main()


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                   text=True).strip()


def test_verify_checkout_is_strict(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f").write_text("x")
    _git(tmp_path, "add", "f")
    _git(tmp_path, "commit", "-qm", "c")
    head = _git(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(SystemExit):  # wrong commit
        lib.verify_checkout(tmp_path, "seal")
    monkeypatch.setitem(lib.OFFICIAL_COMMITS, "seal", head)
    lib.verify_checkout(tmp_path, "seal")
    (tmp_path / "f").write_text("changed")
    with pytest.raises(SystemExit):  # dirty tree
        lib.verify_checkout(tmp_path, "seal")
    (tmp_path / "f").write_text("x")

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(lib.subprocess, "check_output", no_git)
    with pytest.raises(SystemExit):  # git unavailable
        lib.verify_checkout(tmp_path, "seal")
