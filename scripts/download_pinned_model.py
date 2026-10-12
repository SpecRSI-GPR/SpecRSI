#!/usr/bin/env python3
"""Download a frozen publisher manifest and verify every checkpoint byte."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time


def verify_file(root: Path, record: dict) -> dict:
    path = root / record["path"]
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Missing or incorrect-size publisher file: {path}")
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(b"blob " + str(record["bytes"]).encode() + b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            if "git_blob_sha1" in record:
                sha1.update(chunk)
    if record.get("sha256") and record["sha256"] != sha256.hexdigest():
        raise ValueError(f"Publisher SHA256 mismatch: {path}")
    if record.get("git_blob_sha1") and record["git_blob_sha1"] != sha1.hexdigest():
        raise ValueError(f"Publisher Git blob mismatch: {path}")
    return {**record, "sha256": sha256.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    root = Path(lock["local_path"])
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"], local_dir=str(root),
                          allow_patterns=[record["path"] for record in lock["files"]], max_workers=args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(lambda record: verify_file(root, record), lock["files"]))
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = {record["path"] for record in files}
    if not set(index["weight_map"].values()) <= names:
        raise ValueError("Weight index contains an unverified shard")
    report = {"status": "publisher_verified_complete", "repo_id": lock["repo_id"],
              "revision": lock["revision"], "local_path": str(root.resolve()),
              "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "files": files,
              "bytes": sum(record["bytes"] for record in files), "verified_at_unix": time.time()}
    destination = Path(lock["download_manifest"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({key: report[key] for key in ("status", "repo_id", "revision", "bytes")}), flush=True)


if __name__ == "__main__":
    main()
