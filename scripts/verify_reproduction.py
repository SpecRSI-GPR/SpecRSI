#!/usr/bin/env python3
"""Verify release files and exact task memberships without downloading datasets.

With --prepared, also check the locally prepared JSONL bytes and ordered IDs.
The release contains membership IDs and fingerprints; benchmark inputs, sealed
answers, evaluation outputs and model weights are obtained or produced locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def local_path(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Release path escapes repository: {name}")
    return path


def verify(root: Path, *, prepared: bool = False, subsets: bool = False) -> dict:
    manifest_path = root / "release_manifest.json"
    checked = 0
    if manifest_path.is_file():
        for record in json.loads(manifest_path.read_text())["files"]:
            path = local_path(root, record["path"])
            if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError(f"Release file hash/size mismatch: {record['path']}")
            checked += 1
    configuration_references = set()
    for path in (root / "configs").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}:
            for relative in re.findall(r"configs/[A-Za-z0-9_./-]+\.(?:json|yaml|yml|txt|toml)", path.read_text()):
                if not local_path(root, relative).is_file():
                    raise ValueError(f"Missing referenced configuration: {relative}")
                configuration_references.add(relative)
    profile_count = 0
    profiles_path = root / "configs/reproduction/training_profiles.json"
    if profiles_path.is_file():
        profiles = json.loads(profiles_path.read_text())
        for profile in [*profiles["profiles"], *profiles["deployed_configurations"]]:
            if profile.get("config") and profile.get("sha256"):
                if digest(local_path(root, profile["config"])) != profile["sha256"]:
                    raise ValueError(f"Training profile SHA256 mismatch: {profile['config']}")
                profile_count += 1
    science_count = 0
    science_path = root / "data/reproduction/science_scenarios.json"
    if science_path.is_file():
        for scenario in json.loads(science_path.read_text())["scenarios"]:
            registration = scenario["complete_registration"]
            for name, expected in registration["source_sha256"].items():
                if digest(local_path(root, f"src/specrsi/science/{name}")) != expected:
                    raise ValueError(f"Scientific evaluator source SHA256 mismatch: {name}")
            actual = hashlib.sha256(json.dumps(registration, ensure_ascii=False,
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            if actual != scenario["registration_sha256"]:
                raise ValueError(f"Scientific registration SHA256 mismatch: {scenario['domain']}/{scenario['case_id']}")
            science_count += 1
    catalog = json.loads((root / "data/reproduction/catalog.json").read_text())
    datasets, splits, members = 0, 0, 0
    for dataset in catalog["datasets"]:
        datasets += 1
        seen: set[str] = set()
        for role, relative in dataset["splits"].items():
            record = json.loads(local_path(root, relative).read_text())
            ids = record["ids"]
            if record["dataset"] != dataset["dataset"] or record["split"] != role:
                raise ValueError(f"Membership identity mismatch: {relative}")
            if len(ids) != record["count"] or len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate or incorrect membership count: {relative}")
            membership_hash = hashlib.sha256(json.dumps(ids, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if record.get("id_list_sha256") and membership_hash != record["id_list_sha256"]:
                raise ValueError(f"Ordered membership SHA256 mismatch: {relative}")
            if seen.intersection(ids):
                raise ValueError(f"Task IDs cross splits: {dataset['dataset']}/{role}")
            seen.update(ids)
            if len(ids) != dataset["counts"][role]:
                raise ValueError(f"Catalog count mismatch: {relative}")
            if prepared:
                path = local_path(root, record["processed_path"])
                if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError(f"Prepared dataset hash/size mismatch: {path}")
                with path.open(encoding="utf-8", newline="\n") as handle:
                    actual = [str(json.loads(line)["id"]) for line in handle if line.strip()]
                if actual != ids:
                    raise ValueError(f"Prepared dataset ordered IDs mismatch: {path}")
            splits += 1
            members += len(ids)
    subset_files = 0
    if subsets:
        metadata = json.loads((root / "data/reproduction/materialized_subsets.json").read_text())
        for subset in metadata["subsets"]:
            for record in subset["files"]:
                path = local_path(root, record["processed_path"])
                if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError(f"Materialized subset hash/size mismatch: {path}")
                with path.open(encoding="utf-8", newline="\n") as handle:
                    actual = [str(json.loads(line)["id"]) for line in handle if line.strip()]
                if actual != record["ids"] or len(actual) != record["count"]:
                    raise ValueError(f"Materialized subset ordered IDs mismatch: {path}")
                subset_files += 1
    return {"release_files_verified": checked, "datasets": datasets,
            "splits": splits, "members": members, "prepared_bytes_verified": prepared,
            "subset_files_verified": subset_files,
            "configuration_references_verified": len(configuration_references),
            "training_profiles_verified": profile_count, "science_registrations_verified": science_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--subsets", action="store_true", help="Also verify materialized subset bytes and IDs")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, prepared=args.prepared, subsets=args.subsets), indent=2))


if __name__ == "__main__":
    main()
