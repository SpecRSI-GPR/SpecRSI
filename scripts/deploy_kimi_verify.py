#!/usr/bin/env python3
"""Verify every pinned checkpoint byte and warm the read cache before launch."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def verify_checkpoint(lock_path, *, workers=16):
    if not 1 <= workers <= 16:
        raise ValueError("Startup verification uses 1 to 16 bounded workers")
    lock = json.loads(Path(lock_path).read_text())
    root = Path(lock["local_path"])
    total = sum(entry["bytes"] for entry in lock["files"])
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
    if total > available * .5:
        raise RuntimeError("Byte verification/cache warmup requires available host RAM >= twice the checkpoint size")
    started = time.monotonic()

    def check(entry):
        path = root / entry["path"]
        if not entry.get("sha256") or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Unpinned or incomplete model file: {entry['path']}")
        digest = hashlib.sha256()
        buffer = bytearray(32 << 20)
        before = path.stat()
        count = 0
        with path.open("rb", buffering=0) as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            while size := handle.readinto(buffer):
                digest.update(memoryview(buffer)[:size])
                count += size
        after = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if count != entry["bytes"] or digest.hexdigest() != entry["sha256"] or identity(before) != identity(after):
            raise ValueError(f"Model bytes changed or SHA256 mismatch: {entry['path']}")
        return {"path": entry["path"], "bytes": count, "sha256": digest.hexdigest(),
                "file_identity": list(identity(after))}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(check, entry) for entry in lock["files"]]):
            records.append(future.result())
    # Reject concurrent modifications that occurred after an early file's pass.
    for record in records:
        stat = (root / record["path"]).stat()
        identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        if identity != record["file_identity"]:
            raise ValueError(f"Model file changed during startup verification: {record['path']}")
    result = {"status": "all_bytes_sha256_verified", "repo_id": lock["repo_id"], "revision": lock["revision"],
              "workers": workers, "buffer_bytes_per_worker": 32 << 20, "bytes_verified": total,
              "available_ram_at_start": available, "elapsed_seconds": time.monotonic()-started,
              "completed_at_unix": time.time(), "files": sorted(records, key=lambda x: x["path"])}
    path = Path("reports/kimi_startup_verification.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    temporary.replace(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/kimi_k26.lock.json")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    result = verify_checkpoint(args.lock, workers=args.workers)
    print(json.dumps({k:v for k,v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
