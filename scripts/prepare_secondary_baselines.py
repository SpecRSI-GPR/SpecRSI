#!/usr/bin/env python3
"""Install immutable original baseline sources using the committed source locks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

import requests

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("textgrad", "trace")


def verify(target: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        path = target / name
        if not path.resolve().is_relative_to(target.resolve()) or path.is_symlink():
            raise ValueError(f"Unsafe source path: {name}")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed pinned source: {path}")


def prepare(name: str, root: Path = ROOT, verify_only: bool = False) -> dict:
    lock = json.loads((root / f"configs/baselines/secondary_{name}.lock.json").read_text())
    target = root / lock["local_path"]
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source target escapes repository")
    if target.exists():
        verify(target, lock["files"])
        action = "verified_existing"
    else:
        if verify_only:
            raise FileNotFoundError(target)
        response = requests.get(lock["archive_url"], timeout=(30, 180))
        response.raise_for_status()
        archive_bytes = response.content
        if hashlib.sha256(archive_bytes).hexdigest() != lock["archive_sha256"]:
            raise ValueError(f"Pinned archive checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
            stage = Path(temporary) / "source"
            stage.mkdir()
            extracted: set[str] = set()
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                for member in archive:
                    parts = Path(member.name).parts[1:]
                    if not parts or member.isdir():
                        continue
                    relative = Path(*parts)
                    destination = stage / relative
                    if not member.isfile() or not destination.resolve().is_relative_to(stage.resolve()):
                        raise ValueError(f"Unsafe archive member: {member.name}")
                    if relative.as_posix() not in lock["files"]:
                        raise ValueError(f"Unregistered archive member: {member.name}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source:
                        destination.write_bytes(source.read())
                    extracted.add(relative.as_posix())
            if extracted != set(lock["files"]):
                raise ValueError(f"Incomplete original source archive: {name}")
            verify(stage, lock["files"])
            stage.rename(target)
        action = "downloaded_and_verified"
    return {"name": name, "commit": lock["commit"], "archive_sha256": lock["archive_sha256"],
            "files_verified": len(lock["files"]), "path": str(target), "action": action}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Repository root containing the committed source locks")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare, name, args.root, args.verify_only) for name in NAMES]
        for future in futures:
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
