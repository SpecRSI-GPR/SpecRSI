#!/usr/bin/env python3
"""Resumable parallel byte-range download of the pinned public Kimi checkpoint.

Each completed file is checked against the publisher's SHA256. Partial block maps
are merely restart hints; they never authorize serving unverified model weights.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import requests


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/kimi_k26.lock.json")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()
    lock = json.loads(Path(args.lock).read_text())
    root = Path(lock["local_path"])
    root.mkdir(parents=True, exist_ok=True)
    parts = root / ".download-parts"
    parts.mkdir(exist_ok=True)
    manifest_path = Path("reports/kimi_download_progress.json")
    chunk_size = args.chunk_mib << 20
    completed = []
    started = time.time()
    previous = 0.
    local = threading.local()
    verification_pool = ThreadPoolExecutor(max_workers=4)
    pending_verification = []

    def progress(status, current=None, current_bytes=0):
        nonlocal previous
        now = time.time()
        payload = {"status": status, "repo_id": lock["repo_id"], "revision": lock["revision"],
                   "started_at_unix": started, "elapsed_seconds": now - started,
                   "downloaded_verified_bytes": sum(x["bytes"] for x in completed),
                   "current_file": current, "current_completed_block_bytes": current_bytes,
                   "total_bytes": sum(x["bytes"] for x in lock["files"]),
                   "completed_files": len(completed), "total_files": len(lock["files"]),
                   "files": completed, "parallel_ranges": args.workers}
        atomic_json(manifest_path, payload)
        if now - previous >= 10 or status != "downloading":
            print(json.dumps({k: v for k, v in payload.items() if k != "files"}), flush=True)
            previous = now
        return payload

    def finish_part(part, destination, blockmap, name, size, expected):
        actual = checksum(part)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Publisher SHA256 mismatch after complete download: {name}")
        part.replace(destination)
        blockmap.unlink()
        return {"path": name, "bytes": size, "sha256": actual,
                "publisher_sha256_verified": expected is not None}

    def drain_verification():
        for future in list(pending_verification):
            if future.done():
                completed.append(future.result())
                pending_verification.remove(future)

    for entry in lock["files"]:
        drain_verification()
        name, size = entry["path"], entry["bytes"]
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("sha256")
        if destination.exists() and destination.stat().st_size == size:
            actual = checksum(destination)
            if expected is not None and actual != expected:
                raise RuntimeError(f"Existing file checksum mismatch: {name}")
        elif size < chunk_size:
            url = f"https://huggingface.co/{lock['repo_id']}/resolve/{lock['revision']}/{name}"
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            data = response.content
            if len(data) != size:
                raise RuntimeError(f"File size mismatch: {name}")
            temporary = destination.with_suffix(destination.suffix + ".download")
            temporary.write_bytes(data)
            actual = checksum(temporary)
            if expected is not None and actual != expected:
                raise RuntimeError(f"Publisher checksum mismatch: {name}")
            temporary.replace(destination)
        else:
            part = parts / (name.replace("/", "__") + ".part")
            blockmap = part.with_suffix(".blocks.json")
            block_count = math.ceil(size / chunk_size)
            blocks = set()
            if blockmap.exists():
                saved = json.loads(blockmap.read_text())
                if saved["sha256"] != expected or saved["chunk_size"] != chunk_size or not part.exists():
                    raise RuntimeError(f"Incompatible partial download: {name}")
                blocks = set(saved["completed"])
            elif not part.exists() and expected:
                # Recover the contiguous prefix from a stopped hf_hub_download.
                candidates = list((root / ".cache/huggingface/download").glob(f"*.{expected}.incomplete"))
                if candidates:
                    candidate = candidates[0]
                    prefix = candidate.stat().st_size
                    candidate.replace(part)
                    blocks = set(range(min(prefix // chunk_size, block_count)))
            fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
            os.ftruncate(fd, size)
            public_url = f"https://huggingface.co/{lock['repo_id']}/resolve/{lock['revision']}/{name}"
            response = requests.head(public_url, allow_redirects=True, timeout=90)
            response.raise_for_status()
            remote_url = response.url

            def save_blocks():
                atomic_json(blockmap, {"sha256": expected, "chunk_size": chunk_size, "completed": sorted(blocks)})

            def fetch(index):
                begin = index * chunk_size
                end = min(size - 1, begin + chunk_size - 1)
                if not hasattr(local, "session"):
                    local.session = requests.Session()
                for attempt in range(6):
                    try:
                        with local.session.get(remote_url, headers={"Range": f"bytes={begin}-{end}"},
                                               stream=True, timeout=(30, 120)) as result:
                            result.raise_for_status()
                            expected_range = f"bytes {begin}-{end}/{size}"
                            if result.status_code != 206 or result.headers.get("Content-Range") != expected_range:
                                raise RuntimeError(f"Range not honored for {name} block {index}")
                            offset = begin
                            for data in result.iter_content(4 << 20):
                                position = 0
                                while position < len(data):
                                    written = os.pwrite(fd, data[position:], offset)
                                    offset += written
                                    position += written
                            if offset != end + 1:
                                raise RuntimeError(f"Truncated range: {name} block {index}")
                        return index
                    except (requests.RequestException, RuntimeError):
                        if attempt == 5:
                            raise
                        time.sleep(min(16, 2 ** attempt))

            save_blocks()
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(fetch, index) for index in range(block_count) if index not in blocks]
                    for future in as_completed(futures):
                        blocks.add(future.result())
                        save_blocks()
                        progress("downloading", name, sum(min(chunk_size, size - index * chunk_size) for index in blocks))
                os.fsync(fd)
            finally:
                os.close(fd)
            pending_verification.append(verification_pool.submit(finish_part, part, destination, blockmap, name, size, expected))
            progress("downloading")
            continue
        completed.append({"path": name, "bytes": size, "sha256": actual,
                          "publisher_sha256_verified": expected is not None})
        progress("downloading")
    for future in as_completed(pending_verification):
        completed.append(future.result())
        progress("verifying")
    verification_pool.shutdown()
    payload = progress("verified_complete")
    atomic_json(Path("reports/kimi_download.json"), payload)


if __name__ == "__main__":
    main()
