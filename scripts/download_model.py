#!/usr/bin/env python3
"""Download the pinned public Qwen model; verify every shard and record SHA256."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/qwen35_35b_a3b_bf16.lock.json")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    # Publisher manifests bind bytes before downloading. Use the strict path
    # for the current reference model; preserve the old manifest format only
    # for an explicitly supplied historical model lock.
    configured = json.loads(Path(args.lock).read_text())
    if configured.get("files") and configured.get("download_manifest"):
        import subprocess
        import sys
        subprocess.run([sys.executable, str(Path(__file__).with_name("download_pinned_model.py")),
                        "--lock", args.lock, "--workers", str(args.workers)], check=True)
        return
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    lock = json.loads(Path(args.lock).read_text())
    destination = Path(lock["local_path"])
    snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"], local_dir=str(destination),
                      max_workers=args.workers, allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"])
    # Older hub clients create absolute/cache-dependent symlinks. Keep the shared
    # workspace self-contained so a future node does not need this host's /root cache.
    for path in destination.iterdir():
        if path.is_symlink():
            temporary = path.with_suffix(path.suffix + ".copy")
            shutil.copyfile(path.resolve(), temporary)
            temporary.replace(path)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    for shard in shards:
        if not (destination / shard).is_file():
            raise RuntimeError(f"Missing shard {shard}")
    files = []
    for path in sorted(destination.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {**lock, "files": files, "status": "downloaded_and_hashed"}
    Path("reports/model_download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(files), "path": str(destination)}))


if __name__ == "__main__":
    main()
