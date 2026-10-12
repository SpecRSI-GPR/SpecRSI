#!/usr/bin/env python3
"""Download all pinned official Kimi weights and verify publisher SHA256 hashes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/kimi_k26.lock.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    lock = json.loads(Path(args.lock).read_text())
    root = Path(lock["local_path"])
    root.mkdir(parents=True, exist_ok=True)
    report = Path("reports/kimi_download_progress.json")
    started = time.time()
    done = []

    def fetch(entry):
        filename = entry["path"]
        destination = Path(hf_hub_download(lock["repo_id"], filename, revision=lock["revision"], local_dir=root))
        if destination.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"size mismatch for {filename}")
        checksum = hashlib.sha256()
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(32 << 20), b""):
                checksum.update(chunk)
        actual = checksum.hexdigest()
        if entry.get("sha256") and actual != entry["sha256"]:
            raise RuntimeError(f"publisher SHA256 mismatch for {filename}")
        return {"path": filename, "bytes": destination.stat().st_size, "sha256": actual,
                "publisher_sha256_verified": bool(entry.get("sha256"))}

    def progress(status):
        payload = {"status": status, "repo_id": lock["repo_id"], "revision": lock["revision"],
                   "started_at_unix": started, "elapsed_seconds": time.time() - started,
                   "downloaded_verified_bytes": sum(x["bytes"] for x in done),
                   "total_bytes": sum(x["bytes"] for x in lock["files"]),
                   "completed_files": len(done), "total_files": len(lock["files"]), "files": sorted(done, key=lambda x: x["path"])}
        temporary = report.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(report)
        print(json.dumps({k: v for k, v in payload.items() if k != "files"}), flush=True)
        return payload

    progress("downloading")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fetch, entry) for entry in lock["files"]]
        for job in as_completed(jobs):
            done.append(job.result())
            progress("downloading")
    payload = progress("verified_complete")
    Path("reports/kimi_download.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
