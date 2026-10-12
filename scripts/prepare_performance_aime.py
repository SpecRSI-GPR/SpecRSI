#!/usr/bin/env python3
"""Pin an earlier-year tuning corpus and a separate 2025/2026 AIME audit.

The 2024 problems supply disjoint development and selection partitions; the
2025 and 2026 problems form the frozen audit. AIME25 remains a separate version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from specrsi.benchmarks import BenchmarkExample, validate_disjoint_splits

SOURCES = {
    2024: ("math-ai/aime24", "83a7f387baaa524a8bda0022eac0541582297103", "test-00000-of-00001.parquet"),
    2025: ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", "test.jsonl"),
    2026: ("math-ai/aime26", "79037aebdb6580008fb960d17cb21fd3099083e3", "aime2026.jsonl"),
}


def prepare(root: Path):
    import requests
    import pyarrow.parquet as pq
    dataset = "aime_2025_2026"
    raw = root / "raw" / dataset
    processed = root / "processed" / dataset
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifests" / (dataset + ".json")
    if manifest_path.exists():
        raise FileExistsError("Prepared dataset already registered; do not overwrite its manifest")
    files, by_year = [], {}

    def record(path, **extra):
        files.append({"path": str(path.relative_to(root.parent)), "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra})

    for year, (repo, revision, filename) in SOURCES.items():
        path = raw / f"{year}-{filename}"
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.content
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Pinned raw artifact changed")
        path.write_bytes(data)
        record(path, url=url, revision=revision)
        if filename.endswith("parquet"):
            rows = pq.read_table(path).to_pylist()
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 30:
            raise ValueError(f"Expected exactly 30 AIME {year} questions, found {len(rows)}")
        examples = []
        for i, row in enumerate(rows):
            prompt = row.get("problem", row.get("Problem"))
            answer = row.get("answer", row.get("Answer", row.get("solution")))
            boxed = re.fullmatch(r"\\boxed\{([0-9]{1,3})\}", str(answer).strip())
            if boxed:
                answer = boxed.group(1)
            if not isinstance(prompt, str) or not prompt.strip() or not str(answer).isdigit() or not 0 <= int(answer) <= 999:
                raise ValueError(f"Invalid problem/answer schema for {year} row {i}; keys={list(row)}")
            identity = f"aime{year}:{row.get('id', i)}"
            examples.append(BenchmarkExample(identity, dataset, "audit", prompt, str(int(answer)),
                {"year": year, "source_repo": repo, "source_revision": revision,
                 "source_row": i, "split_group": identity}))
        by_year[year] = examples
    earlier = sorted(by_year[2024], key=lambda e: e.content_hash)
    from dataclasses import replace
    splits = {"development": [replace(e, split="development") for e in earlier[:15]],
              "selection": [replace(e, split="selection") for e in earlier[15:]],
              "audit": by_year[2025] + by_year[2026]}
    counts = validate_disjoint_splits(splits)
    for split, examples in splits.items():
        path = processed / f"{split}.jsonl"
        path.write_text("".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in examples))
        record(path)
    manifest = {"schema_version": 1, "dataset": dataset, "counts": counts, "files": files,
        "status": "prepared", "replication_status": "prospective_reconstruction_not_author_split",
        "split_policy": "2024 normalized input hash first15 development/rest15 selection; all2025+2026 audit",
        "prior_exposure": "Local AIME2025 has historical development/evaluation use; audit is held out from this campaign only. Global untouchedness is not claimed.",
        "source_authority": "Math-AI benchmark-maintainer releases; not MAA-issued machine-readable data",
        "sota_comparability": False}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(manifest_path), "counts": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/performance_extension"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root), indent=2))
