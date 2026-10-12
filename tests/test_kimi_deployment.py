import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_startup_verification_rejects_same_size_corruption(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_kimi_verify.py"
    spec = importlib.util.spec_from_file_location("kimi_verify", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "weights.safetensors"
    weight.write_bytes(b"original pinned bytes")
    lock = {"repo_id": "fixture", "revision": "pinned", "local_path": str(model),
            "files": [{"path": weight.name, "bytes": weight.stat().st_size,
                       "sha256": hashlib.sha256(weight.read_bytes()).hexdigest()}]}
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    result = module.verify_checkpoint(lock_path, workers=1)
    assert result["status"] == "all_bytes_sha256_verified"
    weight.write_bytes(b"modified pinned bytes")
    assert weight.stat().st_size == lock["files"][0]["bytes"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.verify_checkpoint(lock_path, workers=1)
