#!/usr/bin/env python3
"""Prepare, verify, or score the complete 30-problem AIME 2025 evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import download
from specrsi.benchmarks.base import BenchmarkExample
from specrsi.benchmarks.data import validate_disjoint_splits
from specrsi.benchmarks.scorers import aime_score

DEFAULT_CONFIG = ROOT / "configs/evaluation/aime2025_full.json"


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalize(raw, config):
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    expected = config["source_ids"]
    if len(expected) != 30 or len(set(expected)) != 30:
        raise ValueError("The full evaluation requires exactly 30 distinct source IDs")
    ids = [str(row["id"]) for row in records]
    if len(ids) != 30 or set(ids) != set(expected):
        raise ValueError("AIME 2025 source must contain each registered ID exactly once")
    by_id = dict(zip(ids, records))
    examples = []
    for key in expected:
        row = by_id[key]
        prompt, answer = row["problem"], str(row["answer"])
        if not isinstance(prompt, str) or not prompt.strip() or not re.fullmatch(r"\d{1,3}", answer):
            raise ValueError(f"Invalid problem or integer answer for source ID {key}")
        examples.append(BenchmarkExample(
            f"aime2025:{key}", config["dataset"], "audit", prompt, str(int(answer)),
            {"year": 2025, "source_id": key, "source_revision": config["source_revision"],
             "split_group": f"aime2025:{key}"}))
    if len({e.content_hash for e in examples}) != 30:
        raise ValueError("Duplicate problem text in AIME 2025 source")
    validate_disjoint_splits({"audit": examples})
    return examples


def artifacts(config, data_root):
    if config.get("year") != 2025 or config.get("evaluation_only") is not True:
        raise ValueError("This command requires the evaluation-only 2025 profile")
    raw_dir = data_root / "raw" / config["dataset"]
    source_files = []
    for source in config["files"]:
        path = raw_dir / source["filename"]
        # Always check cached bytes against the committed source hashes.
        download(source, path, offline=True)
        source_files.append(path)
    examples = normalize((raw_dir / "test.jsonl").read_bytes(), config)
    prepared_dir = data_root / "processed" / config["dataset"]
    payloads = {prepared_dir / f"{split}.jsonl": b"" for split in ("development", "selection")}
    payloads[prepared_dir / "audit.jsonl"] = b"".join(
        (json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode() for e in examples)
    files = [{"path": str(p.relative_to(data_root.parent)), "bytes": p.stat().st_size,
              "sha256": sha256(p.read_bytes())} for p in source_files]
    files += [{"path": str(p.relative_to(data_root.parent)), "bytes": len(raw), "sha256": sha256(raw)}
              for p, raw in payloads.items()]
    manifest = {"schema_version": 1, "dataset": config["dataset"], "status": "prepared",
                "counts": {"development": 0, "selection": 0, "audit": 30},
                "profile_sha256": sha256(encode(config)), "files": files,
                "source_authority": config["source_authority"],
                "source_revision": config["source_revision"],
                "split_policy": "All 30 AIME 2025 I/II problems are evaluation-only.",
                "members": [{"id": e.id, "source_id": e.metadata["source_id"],
                             "input_sha256": e.content_hash} for e in examples]}
    payloads[data_root / "manifests" / (config["dataset"] + ".json")] = encode(manifest)
    return examples, manifest, payloads


def prepare(config, data_root, *, offline=False):
    data_root = Path(data_root)
    for source in config["files"]:
        download(source, data_root / "raw" / config["dataset"] / source["filename"], offline=offline)
    _, manifest, payloads = artifacts(config, data_root)
    # Reject changed prepared artifacts before writing anything.
    for path, raw in payloads.items():
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Prepared AIME artifact changed: {path}")
    for path, raw in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return {"dataset": config["dataset"], "counts": manifest["counts"],
            "profile_sha256": manifest["profile_sha256"], "verified": True}


def verify(config, data_root):
    examples, manifest, payloads = artifacts(config, Path(data_root))
    for path, raw in payloads.items():
        if not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"Prepared AIME artifact is missing or changed: {path}")
    return examples, manifest


def score_predictions(config, data_root, rows):
    examples, manifest = verify(config, data_root)
    by_id = {e.id: e for e in examples}
    seeds = {}
    for row in rows:
        seed, key, prediction = row.get("seed"), row.get("id"), row.get("prediction")
        candidate = row.get("candidate_sha256", "")
        if type(seed) is not int or key not in by_id or not isinstance(prediction, str):
            raise ValueError("Each prediction requires an integer seed, a registered ID, and a string prediction")
        if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate):
            raise ValueError("Each prediction must identify its fixed candidate_sha256")
        current = seeds.setdefault(seed, {"candidate_sha256": candidate, "scores": {}})
        if current["candidate_sha256"] != candidate:
            raise ValueError("Candidate changed within a seed's evaluation")
        if key in current["scores"]:
            raise ValueError(f"Duplicate prediction for seed {seed}: {key}")
        current["scores"][key] = aime_score(prediction, by_id[key].reference).score
    if not seeds:
        raise ValueError("No predictions supplied")
    summaries = []
    for seed, current in sorted(seeds.items()):
        if set(current["scores"]) != set(by_id):
            raise ValueError(f"Seed {seed} must include all 30 problems; include failures with an empty prediction")
        correct = sum(current["scores"].values())
        summaries.append({"seed": seed, "candidate_sha256": current["candidate_sha256"],
                          "correct": int(correct), "total": 30, "score_percent": correct / 30 * 100})
    values = [s["score_percent"] for s in summaries]
    return {"dataset": config["dataset"], "profile_sha256": manifest["profile_sha256"],
            "seeds": summaries, "mean_percent": statistics.mean(values),
            "sd_percent": statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "score"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_root = args.data_root or Path(config["data_root"])
    if args.action == "prepare":
        result = prepare(config, data_root, offline=args.offline)
    elif args.action == "verify":
        _, manifest = verify(config, data_root)
        result = {"verified": True, "counts": manifest["counts"],
                  "profile_sha256": manifest["profile_sha256"]}
    else:
        if args.predictions is None:
            parser.error("score requires --predictions")
        rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
        result = score_predictions(config, data_root, rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
