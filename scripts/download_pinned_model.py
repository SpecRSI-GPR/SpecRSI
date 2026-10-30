#!/usr/bin/env python3
"""Download and verify pinned checkpoint files or tokenizer/configuration files."""
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


def tokenizer_records(files: list[dict]) -> list[dict]:
    """Select text-tokenizer/config assets; weight indexes do not verify weights."""
    records = []
    for record in files:
        path = Path(record["path"])
        if path.name.endswith(".index.json"):
            continue
        if path.suffix in {".json", ".jinja"} or path.name in {"merges.txt", "vocab.txt"}:
            records.append(record)
    if not records:
        raise ValueError("Publisher lock has no tokenizer/configuration files")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Verify tokenizer/config files only; write a separate manifest without claiming weights are ready")
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    root = Path(lock["local_path"])
    records = tokenizer_records(lock["files"]) if args.tokenizer_only else lock["files"]
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"], local_dir=str(root),
                          allow_patterns=[record["path"] for record in records], max_workers=args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(lambda record: verify_file(root, record), records))
    if not args.tokenizer_only:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        names = {record["path"] for record in files}
        if not set(index["weight_map"].values()) <= names:
            raise ValueError("Weight index contains an unverified shard")
    report = {"status": "publisher_verified_tokenizer" if args.tokenizer_only else "publisher_verified_complete",
              "verification_scope": "tokenizer_config_only" if args.tokenizer_only else "full_checkpoint",
              "weights_verified": not args.tokenizer_only, "repo_id": lock["repo_id"],
              "revision": lock["revision"], "local_path": str(root.resolve()),
              "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "files": files,
              "bytes": sum(record["bytes"] for record in files), "verified_at_unix": time.time()}
    destination = Path(lock["download_manifest"])
    if args.tokenizer_only:
        destination = destination.with_name(destination.stem + ".tokenizer.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({key: report[key] for key in
                      ("status", "verification_scope", "weights_verified", "repo_id", "revision", "bytes")}), flush=True)


if __name__ == "__main__":
    main()
