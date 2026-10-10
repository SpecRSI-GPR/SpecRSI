from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .base import BenchmarkExample, MissingArtifactError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: str | Path, *, root: str | Path = ".") -> dict:
    manifest = json.loads(Path(path).read_text())
    def verify_record(record):
        artifact = Path(root) / record["path"]
        if not artifact.is_file():
            raise MissingArtifactError(f"Missing dataset artifact: {artifact}")
        if artifact.stat().st_size != record["bytes"] or sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"Dataset artifact hash/size mismatch: {artifact}")
    records=manifest.get('files',[])
    if len(records)>32:
        # Shared research filesystems have substantial per-file latency. Bound
        # concurrent independent reads while still hashing every original byte.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as pool:
            for _ in pool.map(verify_record,records): pass
    else:
        for record in records: verify_record(record)
    return manifest


def assign_split(content_hash: str, seed: int, *, development_fraction: float,
                 selection_fraction: float) -> str:
    """Assign a stable split using caller-supplied seed and partition fractions."""
    import math
    fractions = (development_fraction, selection_fraction)
    if any(not math.isfinite(x) or x < 0 for x in fractions) or sum(fractions) > 1:
        raise ValueError("Split fractions must be finite, nonnegative, and sum to at most one")
    value = int(hashlib.sha256(f"{seed}:{content_hash}".encode()).hexdigest()[:16], 16) / 2**64
    return ("development" if value < development_fraction else "selection"
            if value < development_fraction + selection_fraction else "audit")


def validate_disjoint_splits(splits: Mapping[str, Iterable[BenchmarkExample]]) -> dict[str, int]:
    ids: dict[tuple[str, str], str] = {}
    contents: dict[str, str] = {}
    groups: dict[str, str] = {}
    counts = collections.Counter()
    for split, examples in splits.items():
        for example in examples:
            if example.split != split:
                raise ValueError(f"Example {example.id} declares {example.split}, loaded as {split}")
            key = (example.dataset, example.id)
            if key in ids:
                raise ValueError(f"Duplicate example ID: {key}, {ids[key]} and {split}")
            previous = contents.get(example.content_hash)
            if previous is not None and previous != split:
                raise ValueError(f"Input leakage between {previous} and {split}: {example.id}")
            group = example.metadata.get("split_group")
            if group and group in groups and groups[group] != split:
                raise ValueError(f"Underlying instance leakage between {groups[group]} and {split}: {group}")
            if group:
                groups[group] = split
            ids[key] = split
            contents[example.content_hash] = split
            counts[split] += 1
    return dict(counts)


def load_examples(dataset: str, split: str, *, data_root: str | Path = "data", verify: bool = True) -> list[BenchmarkExample]:
    if split not in {"development", "selection", "audit"}:
        raise ValueError("Split must be development, selection, or audit")
    root = Path(data_root)
    manifest_path = root / "manifests" / f"{dataset}.json"
    if not manifest_path.exists():
        raise MissingArtifactError(f"Provide a dataset manifest and local examples for {dataset}")
    if verify:
        verify_manifest(manifest_path, root=root.parent)
    path = root / "processed" / dataset / f"{split}.jsonl"
    if not path.exists():
        manifest = json.loads(manifest_path.read_text())
        raise MissingArtifactError(f"{dataset}: {manifest.get('blockers', ['normalized examples unavailable'])}")
    examples = []
    # JSONL records are separated by physical LF/CRLF, not every character
    # recognized by str.splitlines(). U+0085/U+2028/U+2029 are valid characters
    # inside JSON strings and must remain part of the benchmark text.
    with path.open(encoding="utf-8", newline="\n") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL in {path} at physical line {line_number}, "
                                 f"JSON column {error.colno}: {error.msg}") from error
            try:
                examples.append(BenchmarkExample.from_dict(raw))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid benchmark record in {path} at physical line {line_number}: {error}") from error
    validate_disjoint_splits({split: examples})
    return examples


def exclude_audit_overlap(selection: Iterable[BenchmarkExample], audit: Iterable[BenchmarkExample]) -> tuple[list[BenchmarkExample], list[str]]:
    """Filter overlap explicitly, returning excluded IDs for a registered audit."""
    selected = list(selection)
    ids = {e.id for e in selected}
    hashes = {e.content_hash for e in selected}
    keep, removed = [], []
    for example in audit:
        if example.id in ids or example.content_hash in hashes:
            removed.append(example.id)
        else:
            keep.append(example)
    return keep, removed
