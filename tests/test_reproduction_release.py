"""Exercise release verification and corrupted cached-download rejection."""
import importlib.util
import hashlib
import json
import io
from pathlib import Path
import tarfile

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"release_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cached_download_rejects_changed_bytes_before_using_data(tmp_path):
    module = script("prepare_data")
    path = tmp_path / "source.json"
    path.write_bytes(b"original")
    source = {"url": "https://example.invalid/source", "bytes": 8,
              "sha256": hashlib.sha256(b"original").hexdigest()}
    module.download(source, path, offline=True)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        module.download(source, path, offline=True)


def test_release_memberships_reject_cross_split_ids_and_changed_prepared_bytes(tmp_path):
    module = script("verify_reproduction")
    folder = tmp_path / "data/reproduction"
    folder.mkdir(parents=True)
    dataset = {"dataset": "fixture", "counts": {}, "splits": {}}
    for split, uid in (("development", "dev"), ("audit", "aud")):
        path = tmp_path / f"data/processed/fixture/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": uid}) + "\n")
        membership = {"dataset": "fixture", "split": split, "count": 1, "ids": [uid],
                      "processed_path": str(path.relative_to(tmp_path)),
                      "bytes": path.stat().st_size, "sha256": module.digest(path)}
        relative = f"data/reproduction/{split}.json"
        (tmp_path / relative).write_text(json.dumps(membership))
        dataset["splits"][split] = relative
        dataset["counts"][split] = 1
    (folder / "catalog.json").write_text(json.dumps({"datasets": [dataset]}))
    assert module.verify(tmp_path, prepared=True)["members"] == 2
    path.write_text(json.dumps({"id": "corrupted"}) + "\n")
    with pytest.raises(ValueError, match="hash/size"):
        module.verify(tmp_path, prepared=True)
    member = tmp_path / dataset["splits"]["audit"]
    record = json.loads(member.read_text())
    record["ids"] = ["dev"]
    member.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cross splits"):
        module.verify(tmp_path)


def test_release_paths_cannot_escape_repository(tmp_path):
    module = script("verify_reproduction")
    with pytest.raises(ValueError, match="escapes"):
        module.local_path(tmp_path, "../outside")


def test_optional_upstream_download_verifies_archive_and_cached_source(tmp_path):
    module = script("prepare_upstream")
    payload = b"source fixture\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("upstream-commit/package.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_path = tmp_path / "source.tar.gz"
    archive_path.write_bytes(buffer.getvalue())
    locks = tmp_path / "configs/baselines"
    locks.mkdir(parents=True)
    lock = {"commit": "fixture", "local_path": "third_party/flashevolve",
            "archive_url": archive_path.as_uri(),
            "archive_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "files": {"package.py": hashlib.sha256(payload).hexdigest()}}
    (locks / "flash_evolve.lock.json").write_text(json.dumps(lock))
    assert module.prepare("flash-evolve", tmp_path)["files_verified"] == 1
    (tmp_path / "third_party/flashevolve/package.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed locked upstream"):
        module.prepare("flash-evolve", tmp_path, verify_only=True)
