"""Check local model bytes against the downloaded, pinned model manifest."""
import hashlib
import json
from pathlib import Path


def verify_model(lock_path, manifest_path):
    lock = json.loads(Path(lock_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if lock["revision"] != manifest["revision"] or lock["repo_id"] != manifest["repo_id"]:
        raise ValueError("Model manifest and configured revision differ")
    root = Path(lock["local_path"])
    listed = {entry["path"] for entry in manifest["files"]}
    index = json.loads((root / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required <= listed:
        raise ValueError("Model hash manifest does not cover all required files")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or truncated model file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Model hash mismatch: {path}")
    return {"revision": lock["revision"], "verified_files": len(listed),
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
