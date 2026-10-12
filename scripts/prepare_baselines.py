#!/usr/bin/env python3
"""Fetch official baseline source snapshots with commit and archive checksums.

Fetching source does not assert that an SpecRSI-specific matched adaptation exists.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile
import requests

REPOS = {"gepa": "gepa-ai/gepa", "dspy": "stanfordnlp/dspy", "hgm": "metauto-ai/HGM", "aflow": "FoundationAgents/AFlow"}
ROOT = Path(__file__).resolve().parents[1]


def fetch(entry):
    name, repo = entry
    target = ROOT / "third_party" / name
    lockfile = ROOT / "configs/baselines" / f"{name}.lock.json"
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(lockfile.read_text()) if lockfile.exists() else None
    if previous:
        commit = previous["commit"]
    else:
        r = requests.get(f"https://api.github.com/repos/{repo}/commits/HEAD", timeout=45)
        r.raise_for_status()
        commit = r.json()["sha"]
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    checksum = hashlib.sha256(r.content).hexdigest()
    if previous and previous["archive_sha256"] != checksum:
        raise RuntimeError(f"Archive digest changed for pinned {repo}")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            destination = target.joinpath(*parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError("Archive traversal")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    destination.write_bytes(source.read())
    lock = {"name": name, "repository": "https://github.com/" + repo, "commit": commit,
            "archive_sha256": checksum, "source_downloaded": True,
            "specrsi_matched_adaptation_validated": False, "path": str(target.relative_to(ROOT))}
    lockfile.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, REPOS.items()):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
