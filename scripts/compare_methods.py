#!/usr/bin/env python3
"""Strict prospective comparison of registered methods from observed run artifacts.

Usage:
  python scripts/compare_methods.py --schema
  python scripts/compare_methods.py registration.json runs.jsonl --output comparison.json

Register profiles and repetitions before running the experiment. Each JSONL row
identifies one completed run and supplies its frozen comparison contract. Score
and actual_cost are read from SHA256-checked JSON artifacts; optional values in
the JSONL row must agree exactly. Source paths are relative to the JSONL file.
JSON pointers follow RFC 6901, with optional item_pointer and mean/sum reduction
for arrays. Missing, null, or interrupted outcomes cannot be averaged away.

This checks recorded metric provenance and declared matching conditions. File
hashes cannot prove that runs were conducted as declared or that an audit was
never consulted while tuning. The output explicitly limits its conclusion to
the registered comparisons, never a claim to outperform every existing method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from specrsi.statistics import compare_registered_methods


def json_pointer(document, pointer: str, *, require_complete: bool = False):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("Use an RFC 6901 JSON pointer (empty string selects the document)")
    result = document
    for token in pointer.split("/")[1:]:
        if (require_complete and isinstance(result, dict)
                and "status" in result and result["status"] != "complete"):
            raise ValueError("Incomplete audit outcomes cannot be treated as completed observations")
        # Reject malformed escapes rather than silently selecting a different key.
        for index, char in enumerate(token):
            if char == "~" and (index + 1 == len(token) or token[index + 1] not in "01"):
                raise ValueError("Malformed JSON pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(result, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ValueError("JSON array pointers require a nonnegative canonical integer")
            index = int(token)
            if index >= len(result):
                raise ValueError(f"JSON pointer array index out of range: {pointer}")
            result = result[index]
        elif isinstance(result, dict) and token in result:
            result = result[token]
        else:
            raise ValueError(f"JSON pointer does not identify an observed field: {pointer}")
    if (require_complete and isinstance(result, dict)
            and "status" in result and result["status"] != "complete"):
        raise ValueError("Incomplete audit outcomes cannot be treated as completed observations")
    return result


def observed_value(source: dict, base: Path, cache: dict):
    if not isinstance(source, dict) or not {"path", "sha256", "pointer"} <= source.keys():
        raise ValueError("Each observed metric needs path, sha256, and pointer provenance")
    path = (base / source["path"]).resolve()
    digest = source["sha256"]
    if path not in cache:
        raw = path.read_bytes()
        cache[path] = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
    actual_digest, document = cache[path]
    if digest != actual_digest:
        raise ValueError(f"Raw run artifact hash mismatch: {path}")
    # Completion belongs to the run/outcome, not just the selected numeric leaf.
    # A scalar pointer must not bypass an interrupted status in an ancestor.
    value = json_pointer(document, source["pointer"], require_complete=True)
    reduction = source.get("reduction", "identity")
    if reduction not in {"identity", "mean", "sum"}:
        raise ValueError("Metric reduction must be identity, mean, or sum")
    if reduction == "identity":
        if "item_pointer" in source:
            raise ValueError("item_pointer requires an array reduction")
        values = [value]
    else:
        if not isinstance(value, list) or not value:
            raise ValueError("A reduction needs a nonempty array of observed outcomes")
        if any(isinstance(item, dict) and "status" in item and item["status"] != "complete" for item in value):
            raise ValueError("Incomplete audit outcomes cannot be treated as completed observations")
        values = [json_pointer(item, source.get("item_pointer", ""), require_complete=True) for item in value]
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item) for item in values):
        raise ValueError("Every observed outcome must be finite; missing audit outcomes cannot be dropped")
    result = math.fsum(values) / len(values) if reduction == "mean" else math.fsum(values)
    identity = (str(path), digest, source["pointer"], source.get("item_pointer", ""), reduction)
    return result, identity


def load_observed_records(path: Path):
    cache = {}
    identities = set()
    rows = []
    with path.open(encoding="utf-8", newline="\n") as handle:
        return _load_observed_lines(handle, path, cache, identities, rows)


def _load_observed_lines(handle, path, cache, identities, rows):
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL in {path} at physical line {line_number}, "
                             f"JSON column {error.colno}: {error.msg}") from error
        evidence = row.get("evidence", {})
        for field in ("score", "actual_cost"):
            if field not in evidence:
                raise ValueError(f"Line {line_number}: {field} must cite a raw observed artifact")
            value, source_identity = observed_value(evidence[field], path.parent, cache)
            if field in row and row[field] != value:
                raise ValueError(f"Line {line_number}: supplied {field} differs from its raw artifact")
            row[field] = value
            if field == "score":
                # One result may support different endpoints/profiles, but cannot
                # be duplicated as additional independent runs within a profile.
                identity = (row.get("profile_id"), source_identity)
                if identity in identities:
                    raise ValueError("A source outcome was duplicated as multiple registered runs")
                identities.add(identity)
        rows.append(row)
    return rows, {str(file): digest for file, (digest, _) in sorted(cache.items())}


def schema_example():
    contract = {"dataset": "declared-dataset", "model_revision": "frozen-model-revision",
                "scorer": "frozen-native-scorer-revision", "permission_band": "D+H",
                "endpoint": "audit_score", "evaluation_split": "audit",
                "selection_manifest_sha256": "0" * 64, "evaluation_manifest_sha256": "1" * 64,
                "initial_artifact_sha256": "2" * 64, "execution_contract_sha256": "3" * 64,
                "resource_profile": "declared_hardware_workers_and_runtime",
                "budget": {"cap": 100000, "unit": "tokens",
                           "accounting": "search_including_all_proposals_checks_and_failed_calls"}}
    return {"registration": {"schema_version": 1, "candidate": "specrsi", "alpha": .05,
                             "bootstrap_resamples": 10000, "random_seed": 0,
                             "profiles": [{"id": "example", "baselines": ["baseline"],
                                           "training_seeds": [0, 1, 2, 3, 4], "search_seeds": [0, 1, 2],
                                           "direction": "higher", "minimum_effect": 0, "contract": contract}]},
            "one_jsonl_row": {"profile_id": "example", "method": "specrsi", "training_seed": 0,
                              "search_seed": 0, "run_id": "example-specrsi-0-0", "status": "complete",
                              "contract": contract,
                              "evidence": {"score": {"path": "run/audit.json", "sha256": "REPLACE_WITH_ACTUAL_SHA256",
                                                     "pointer": "", "item_pointer": "/score", "reduction": "mean"},
                                           "actual_cost": {"path": "run/search.json", "sha256": "REPLACE_WITH_ACTUAL_SHA256",
                                                           "pointer": "/total_cost"}}},
            "notes": ["The example is a schema, not evidence or a manuscript reproduction registration.",
                      "Supply every registered method x training_seed x search_seed for every profile.",
                      "Choose budgets, seed counts, and hypotheses before observing evaluation results.",
                      "Five seeds yield a minimum exact one-sided p-value of 1/32; multiple baselines may need more independent seeds.",
                      "Matching manifest hashes and metadata do not prove audit isolation or faithful baseline execution."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registration", nargs="?", type=Path)
    parser.add_argument("records", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--schema", action="store_true", help="Print a registration and evidence-row example")
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(schema_example(), indent=2))
        return
    if args.registration is None or args.records is None:
        parser.error("registration.json and runs.jsonl are required (see --schema)")
    try:
        raw_registration = args.registration.read_bytes()
        rows, sources = load_observed_records(args.records)
        report = compare_registered_methods(json.loads(raw_registration), rows)
        report["registration_sha256"] = hashlib.sha256(raw_registration).hexdigest()
        report["records_sha256"] = hashlib.sha256(args.records.read_bytes()).hexdigest()
        report["verified_source_hashes"] = sources
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Comparison reports are immutable; existing results remain intact.
            with args.output.open("x") as handle:
                handle.write(rendered)
        print(rendered, end="")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Comparison refused: {error}\n")


if __name__ == "__main__":
    main()
