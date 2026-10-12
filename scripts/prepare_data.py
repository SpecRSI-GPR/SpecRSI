#!/usr/bin/env python3
"""Download pinned public sources, verify bytes, and freeze disjoint task splits."""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from specrsi.benchmarks.base import BenchmarkExample, canonical_json
from specrsi.benchmarks.data import assign_split, sha256_file, validate_disjoint_splits, verify_manifest


def download(source: dict, target: Path, *, offline: bool = False):
    target.parent.mkdir(parents=True, exist_ok=True)
    def verify(path: Path):
        # Release source records bind the downloaded bytes as well as the URL
        # revision. Check cached files too: an existing name is not evidence of
        # an intact dataset, and --force must not bless changed source bytes.
        if source.get("bytes") is not None and path.stat().st_size != source["bytes"]:
            raise ValueError(f"Pinned source size mismatch: {path}")
        if source.get("sha256") is not None and sha256_file(path) != source["sha256"]:
            raise ValueError(f"Pinned source SHA256 mismatch: {path}")
    if target.exists():
        verify(target)
        return
    if offline:
        raise FileNotFoundError(f"Offline mode: missing {target}")
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(3):
        try:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "SpecRSI-research-data-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1 << 20)
            if partial.stat().st_size == 0:
                raise ValueError("Downloaded empty file")
            verify(partial)
            partial.replace(target)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def extract_zip(archive: Path, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"Archive symlink is not accepted: {item.filename}")
        zf.extractall(directory)


def rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        # Arrow timestamps are datetime objects; preserve ISO values in JSONL.
        return json.loads(json.dumps(pq.read_table(path).to_pylist(), default=lambda value: value.isoformat()))
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def normalize(name: str, config: dict, raw: Path):
    fmt = config["format"]
    source_files = [raw / f["filename"] for f in config["files"] if f["filename"].endswith((".json", ".jsonl", ".csv", ".parquet"))]
    for path in source_files:
        for index, row in enumerate(rows(path)):
            metadata = {"source_file": path.name, "source_index": index}
            reference = row
            if fmt == "game24":
                numbers = [int(v) for v in row["Puzzles"].split()]
                key, prompt, reference = str(row.get("Rank", index)), f"Use each number exactly once with +, -, *, / and parentheses to obtain 24: {' '.join(map(str,numbers))}. Return only an arithmetic expression.", numbers
            elif fmt == "mbpp":
                key, prompt = str(row["task_id"]), row["text"]
                # Public prompt examples use only the first test; full tests stay in reference.
                metadata["native_task_id"] = row["task_id"]
            elif fmt == "ifbench":
                key, prompt = str(row["key"]), row["prompt"]
            elif fmt == "hover":
                key, prompt = str(row["uid"]), row["claim"]
                metadata["retrieval_required"] = True
            elif fmt == "pupa":
                key, prompt = str(row.get("id", f"{path.stem}:{index}")), row["user_query"]
                metadata["variant"] = path.stem
            elif fmt == "aime25":
                key, prompt, reference = str(row.get("id", index)), row["problem"], row["answer"]
            elif fmt == "math":
                key, prompt = f"{path.stem}:{index}", row["problem"]
                metadata.update(subject=row["type"], level=row["level"], source_split="train" if "train" in path.name else "test")
            elif fmt == "livebench_math":
                key, prompt = str(row["question_id"]), row["turns"][0]
                metadata["native_task"] = row["task"]
            elif fmt == "swe":
                key, prompt = row["instance_id"], row["problem_statement"]
                metadata.update(repo=row["repo"], base_commit=row["base_commit"])
            elif fmt == "hotpotqa":
                key = row.get("_id", row.get("id"))
                native_context = row["context"]
                if isinstance(native_context, dict):
                    native_context = list(zip(native_context["title"], native_context["sentences"]))
                context = "\n\n".join(title + ":\n" + " ".join(sentences) for title, sentences in native_context)
                prompt = f"{context}\n\nQuestion: {row['question']}\nAnswer:"
                supporting = row["supporting_facts"]
                if isinstance(supporting, dict):
                    supporting = list(zip(supporting["title"], supporting["sent_id"]))
                reference = {"answer": row["answer"], "supporting_facts": supporting}
                metadata["question"] = row["question"]
            else:
                raise ValueError(f"Unsupported normalization format: {fmt}")
            yield BenchmarkExample(key, name, "unassigned", prompt, reference, metadata)
    if fmt == "polyglot":
        for config_path in sorted((raw / "source").rglob(".meta/config.json")):
            exercise = config_path.parent.parent
            relative = exercise.relative_to(next((raw / "source").iterdir()))
            instructions = exercise / ".docs/instructions.md"
            if not instructions.is_file():
                continue
            meta = json.loads(config_path.read_text())
            files = meta.get("files", {})
            yield BenchmarkExample(str(relative), name, "unassigned", instructions.read_text(),
                                   {"exercise_path": str(exercise.relative_to(ROOT)), "files": files},
                                   {"language": relative.parts[0], "native_harness_required": True})
    elif fmt == "planbench":
        # The native one-shot examples rotate through evaluation instance IDs.
        # A prospective split must remove those demonstrations. Retain every
        # original task in raw/source; only the explicitly named zero-shot plan
        # generation track is normalized here, without inventing SpecRSI's subset.
        for path in sorted((raw / "source").glob("*/plan-bench/prompts/*/task_1_plan_generation.json")):
            document = json.loads(path.read_text())
            if not isinstance(document, dict):
                continue
            for index, row in enumerate(document.get("instances", [])):
                prompt = row.get("query", row.get("prompt", ""))
                if not prompt:
                    continue
                statements = prompt.split("[STATEMENT]")
                if len(statements) < 2:
                    raise ValueError(f"Cannot identify PlanBench statement boundary: {path}")
                prompt = statements[0] + "[STATEMENT]" + statements[-1]
                key = f"{path.parent.name}/{path.stem}/{index}"
                yield BenchmarkExample(key, name, "unassigned", prompt, row,
                                       {"domain": path.parent.name, "task": path.stem,
                                        "track": "prospective_zero_shot_plan_generation",
                                        "split_group": f"planbench-instance:{row['instance_id']}",
                                        "source_file": str(path.relative_to(ROOT))})


def prepare(name: str, config: dict, seed: int, *, offline: bool = False, force: bool = False) -> dict:
    raw = ROOT / "data/raw" / name
    manifest_path = ROOT / "data/manifests" / f"{name}.json"
    if manifest_path.exists() and not force:
        manifest = verify_manifest(manifest_path, root=ROOT)
        if manifest.get("status") not in {"download_failed"}:
            print(json.dumps({"dataset": name, "status": "verified_cached", "counts": manifest.get("counts", {})}), flush=True)
            return manifest
    elif manifest_path.exists():
        # Re-normalization may change processed files, but never bless modified
        # downloaded bytes or evaluator source merely because --force was used.
        previous = json.loads(manifest_path.read_text())
        for record in previous.get("files", []):
            if record["path"].startswith(f"data/raw/{name}/"):
                path = ROOT / record["path"]
                if not path.is_file() or sha256_file(path) != record["sha256"]:
                    raise ValueError(f"Pinned raw artifact changed: {path}")
    files = []
    manifest = {"schema_version": 1, "dataset": name, "family": config["family"],
                "created_at": datetime.now(timezone.utc).isoformat(), "source_page": config["source_page"],
                "replication_status": "pinned_sources_and_deterministic_splits",
                "split_policy": {"algorithm": "sha256-group-or-normalized-input-v2", "group_metadata_field": "split_group",
                                 "seed": seed, "fractions": [0.2, 0.4, 0.4]},
                "files": files}
    if config.get("blockers"):
        manifest["blockers"] = config["blockers"]
    try:
        for source in config["files"]:
            path = raw / source["filename"]
            download(source, path, offline=offline)
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size,
                          "url": source["url"], "revision": source["revision"]})
            if source.get("extract"):
                extract_zip(path, raw / source.get("extract_to", "source"))
                # Hash extracted evaluator files too, so scoring cannot silently use
                # modified code while a source ZIP still has its expected digest.
                for extracted in sorted((raw / "source").rglob("*")):
                    if extracted.is_file() and "__pycache__" not in extracted.parts and extracted.suffix != ".pyc":
                        files.append({"path": str(extracted.relative_to(ROOT)), "sha256": sha256_file(extracted),
                                      "bytes": extracted.stat().st_size})
        partitions = {key: [] for key in ["development", "selection", "audit"]}
        seen_ids = set()
        for example in normalize(name, config, raw):
            if example.id in seen_ids:
                raise ValueError(f"Duplicate native ID in {name}: {example.id}")
            seen_ids.add(example.id)
            split_group = example.metadata.get("split_group")
            split_hash = hashlib.sha256(split_group.encode()).hexdigest() if split_group else example.content_hash
            split = assign_split(split_hash, seed)
            if config["format"] == "math":
                # Keep the original 5,000 test questions exclusively for audit.
                split = "audit" if example.metadata["source_split"] == "test" else ("development" if int(split_hash[:8], 16) % 2 == 0 else "selection")
            partitions[split].append(dataclasses.replace(example, split=split))
        if config["format"] == "math":
            audit_hashes = {example.content_hash for example in partitions["audit"]}
            removed = []
            for split in ("development", "selection"):
                removed.extend(example.id for example in partitions[split] if example.content_hash in audit_hashes)
                partitions[split] = [example for example in partitions[split] if example.content_hash not in audit_hashes]
            manifest["split_policy"] = {"algorithm": "original-test-audit; train-sha256-parity-development-selection", "seed": seed,
                                        "removed_training_test_overlap_ids": removed}
        counts = validate_disjoint_splits(partitions)
        if counts:
            for split, examples in partitions.items():
                path = ROOT / "data/processed" / name / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(canonical_json(e.to_dict()) + "\n" for e in examples))
                files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest.update(status="prepared" if counts else "source_only" if files else "missing_source_artifacts", counts=counts)
    except Exception as exc:
        manifest.update(status="download_failed", error=f"{type(exc).__name__}: {exc}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"dataset": name, "status": manifest["status"], "counts": manifest.get("counts", {}), "error": manifest.get("error")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate prospective manifests from pinned sources")
    parser.add_argument("--verify", action="store_true", help="Verify hashes only; no network or writes")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/datasets/sources.json").read_text())
    names = list(config["datasets"]) if args.datasets == ["all"] else args.datasets
    unknown = set(names) - config["datasets"].keys()
    if unknown:
        parser.error(f"Unknown datasets: {sorted(unknown)}")
    if args.verify:
        for name in names:
            manifest = verify_manifest(ROOT / "data/manifests" / f"{name}.json", root=ROOT)
            print(f"{name}: {manifest['status']}; {len(manifest['files'])} hashes verified")
        return
    from concurrent.futures import ThreadPoolExecutor
    def run(name):
        return prepare(name, config["datasets"][name], config["split_seed"], offline=args.offline, force=args.force)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        manifests = list(pool.map(run, names))
    if any(m["status"] == "download_failed" for m in manifests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
