#!/usr/bin/env python3
"""Bounded read-only sequential checkpoint prefetch for shared-filesystem mmap."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--mapped-libraries", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1 to 8 bounded prefetch workers")
    files = ([Path(p) for p, _ in json.loads(Path("reports/kimi_mapped_libraries.json").read_text())["files"]]
             if args.mapped_libraries else sorted(Path("models/Kimi-K2.6").glob("model-*-of-*.safetensors")))
    output_stem = "kimi_library_prefetch" if args.mapped_libraries else "kimi_prefetch"
    total = sum(p.stat().st_size for p in files)
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
    if not files or total > available * .5:
        raise RuntimeError("Prefetch requires complete checkpoint and at least twice its size in available host RAM")
    files = files[args.start_shard-1:] + files[:args.start_shard-1]
    started = time.monotonic()
    completed = []

    def prefetch(path):
        begun = time.monotonic()
        buffer = bytearray(32 << 20)
        read = 0
        with path.open("rb", buffering=0) as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            while count := handle.readinto(buffer):
                read += count
        return {"path": path.name, "bytes": read, "seconds": time.monotonic()-begun}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(prefetch, p) for p in files]):
            completed.append(future.result())
            report = {"status": "complete" if len(completed) == len(files) else "prefetching",
                      "method": "read_only_32MiB_sequential_reads", "workers": args.workers,
                      "available_ram_at_start": available, "total_checkpoint_bytes": total,
                      "bytes_prefetched": sum(x["bytes"] for x in completed),
                      "elapsed_seconds": time.monotonic()-started, "files": completed}
            temp = Path(f"reports/{output_stem}.tmp")
            temp.write_text(json.dumps(report, indent=2)+"\n")
            temp.replace(f"reports/{output_stem}.json")
            print(json.dumps({k:v for k,v in report.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
