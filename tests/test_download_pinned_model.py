import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "pinned_model", Path(__file__).parents[1] / "scripts/download_pinned_model.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


@pytest.fixture
def locked_model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    payloads = {
        "config.json": b"{}", "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}", "chat_template.jinja": b"{{messages}}",
        "merges.txt": b"#version: 0.2\n",
        "model.safetensors.index.json": json.dumps({"weight_map": {"layer": "model.safetensors"}}).encode(),
        "model.safetensors": b"weight fixture", "README.md": b"model information",
    }
    files = []
    for name, data in payloads.items():
        (root / name).write_bytes(data)
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    lock = {"repo_id": "fixture/model", "revision": "fixed-revision", "local_path": str(root),
            "download_manifest": str(tmp_path / "reports/model.json"), "files": files}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    return path, lock


def invoke(monkeypatch, path, *options):
    monkeypatch.setattr(sys, "argv", ["download_pinned_model.py", "--lock", str(path), *options])
    download.main()


def test_tokenizer_verification_uses_separate_manifest_without_weights(locked_model, monkeypatch, capsys):
    path, lock = locked_model
    root = Path(lock["local_path"])
    (root / "model.safetensors").unlink()
    (root / "model.safetensors.index.json").unlink()
    invoke(monkeypatch, path, "--verify-only", "--tokenizer-only")
    full = Path(lock["download_manifest"])
    assert not full.exists()
    manifest = json.loads(full.with_name("model.tokenizer.json").read_text())
    assert manifest["status"] == "publisher_verified_tokenizer"
    assert manifest["verification_scope"] == "tokenizer_config_only"
    assert manifest["weights_verified"] is False
    assert {r["path"] for r in manifest["files"]} == {
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt"}
    assert json.loads(capsys.readouterr().out)["weights_verified"] is False
    with pytest.raises(ValueError, match="Missing"):
        invoke(monkeypatch, path, "--verify-only")


@pytest.mark.parametrize("checksum", ["sha256", "git_blob_sha1"])
def test_tokenizer_mode_rejects_same_size_wrong_hash(locked_model, monkeypatch, checksum):
    path, lock = locked_model
    if checksum == "git_blob_sha1":
        record = next(r for r in lock["files"] if r["path"] == "tokenizer.json")
        record.pop("sha256")
        record[checksum] = hashlib.sha1(b"blob 2\0{}").hexdigest()
        path.write_text(json.dumps(lock))
    (Path(lock["local_path"]) / "tokenizer.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="mismatch"):
        invoke(monkeypatch, path, "--verify-only", "--tokenizer-only")
    assert not Path(lock["download_manifest"]).exists()
    assert not Path(lock["download_manifest"]).with_name("model.tokenizer.json").exists()


def test_tokenizer_download_never_requests_weight_files(locked_model, monkeypatch):
    path, lock = locked_model
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kw: calls.append(kw)))
    full = Path(lock["download_manifest"])
    full.parent.mkdir()
    full.write_text("previous complete manifest")
    invoke(monkeypatch, path, "--tokenizer-only")
    assert calls[0]["revision"] == lock["revision"]
    assert set(calls[0]["allow_patterns"]) == {
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt"}
    assert full.read_text() == "previous complete manifest"


def test_full_mode_keeps_complete_checkpoint_verification(locked_model, monkeypatch):
    path, lock = locked_model
    invoke(monkeypatch, path, "--verify-only")
    result = json.loads(Path(lock["download_manifest"]).read_text())
    assert result["status"] == "publisher_verified_complete"
    assert result["weights_verified"] is True
    assert len(result["files"]) == len(lock["files"])
