#!/usr/bin/env python3
"""Download immutable optional baseline sources; verify all locked file bytes.

Baseline source is fetched from the upstream release rather than bundled with
benchmark data or upstream experiment outputs in the SpecRSI code archive.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCKS = {"flash-evolve": "flash_evolve.lock.json",
         "textgrad": "secondary_textgrad.lock.json", "trace": "secondary_trace.lock.json"}


def prepare(name: str, root: Path = ROOT, *, verify_only: bool = False) -> dict:
    lock = json.loads((root / "configs/baselines" / LOCKS[name]).read_text())
    target = root / lock["local_path"]
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Baseline source destination escapes repository")

    def verify(directory: Path):
        for relative, expected in lock["files"].items():
            path = directory / relative
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"Unsafe locked upstream path: {relative}")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Missing or changed locked upstream file: {relative}")

    if target.exists():
        verify(target)
    else:
        if verify_only:
            raise FileNotFoundError(target)
        url = lock.get("archive_url") or lock["repository"].replace("https://github.com/", "https://codeload.github.com/") + "/tar.gz/" + lock["commit"]
        with urllib.request.urlopen(url, timeout=180) as response:
            archive_bytes = response.read()
        if lock.get("archive_sha256") and hashlib.sha256(archive_bytes).hexdigest() != lock["archive_sha256"]:
            raise ValueError(f"Upstream archive checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".baseline-", dir=target.parent) as temporary:
            stage = Path(temporary) / "source"
            stage.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                for member in archive:
                    parts = Path(member.name).parts[1:]
                    if not parts or member.isdir():
                        continue
                    relative = Path(*parts)
                    path = stage / relative
                    if not member.isfile() or not path.resolve().is_relative_to(stage.resolve()):
                        raise ValueError(f"Unsafe upstream archive member: {member.name}")
                    if relative.as_posix() not in lock["files"]:
                        raise ValueError(f"Unregistered upstream archive file: {relative}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source:
                        path.write_bytes(source.read())
            verify(stage)
            stage.rename(target)
    return {"baseline": name, "commit": lock["commit"], "files_verified": len(lock["files"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", choices=list(LOCKS), default=list(LOCKS))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        print(json.dumps(prepare(name, args.root, verify_only=args.verify_only)))


if __name__ == "__main__":
    main()
