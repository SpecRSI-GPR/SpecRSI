#!/usr/bin/env python3
"""Register, screen, confirm and audit immutable inference recipes with real costs.

Development measurements select a recipe before the heldout audit. Interrupted attempts remain in
both denominators and accounting; an unresolved request prevents automatic replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from specrsi.benchmarks import load_examples, validate_disjoint_splits
from specrsi.core.models import Candidate, EvaluationScope, Task, digest, thaw
from specrsi.experiments import QwenExecutor
from specrsi.models import LocalModelClient


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_hashes():
    paths = [Path(__file__), *(ROOT / "src/specrsi").rglob("*.py")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def physical_accounting(records):
    measured, reserved, gpu_seconds = 0, 0, 0.
    seen = set()
    for record in records:
        request_id = record.get("request_id")
        if not request_id or request_id in seen:
            raise ValueError("Physical calls require unique request IDs")
        seen.add(request_id)
        status = record.get("metering_status", "server_measured")
        if status == "server_measured":
            counts = [record.get("prompt_tokens"), record.get("completion_tokens")]
            if any(type(value) is not int or value < 0 for value in counts):
                raise ValueError("Physical call has invalid measured token counts")
            cost = sum(counts)
            if record.get("accounted_tokens", cost) != cost:
                raise ValueError("Physical measured and accounted costs differ")
            if "reserved_tokens" in record and cost > record["reserved_tokens"]:
                raise ValueError("Physical call exceeded its enforced reservation")
            elapsed, gpus = record.get("elapsed_seconds", 0), record.get("gpu_count", 1)
            if not math.isfinite(elapsed) or elapsed < 0 or type(gpus) is not int or gpus < 0:
                raise ValueError("Physical call has invalid GPU timing")
            gpu_seconds += elapsed * gpus
            measured += cost
        elif status == "unavailable_reserved_upper_bound":
            cost = record.get("reserved_tokens")
            if type(cost) is not int or cost < 1 or record.get("accounted_tokens") != cost:
                raise ValueError("Unmetered calls require their full enforced reservation")
            reserved += cost
        else:
            raise ValueError("Unknown physical call metering status")
    return {"charged_tokens": measured + reserved, "measured_tokens": measured,
            "reserved_unmetered_tokens": reserved, "metering_complete": reserved == 0,
            "gpu_seconds": gpu_seconds}


def append_jsonl(path, values):
    with path.open("a") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_selected_candidate(output, frozen):
    """Publish a bare frozen Candidate for artifact-only transfer, never a ledger."""
    path = output / "selected_candidate.json"
    manifest = frozen["candidate"]
    if Candidate(**manifest).uid != frozen["candidate_uid"]:
        raise ValueError("Selected candidate identity differs from the frozen choice")
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Selected candidate artifact changed after confirmation")
    else:
        save(path, manifest)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != frozen["selected_candidate_sha256"]:
        raise ValueError("Selected candidate bytes differ from the frozen hash")
    return path


def choose(examples, count, seed):
    if type(count) is not int or not 0 < count <= len(examples):
        raise ValueError(f"Invalid count {count} for {len(examples)} examples")
    return sorted(examples, key=lambda e: digest([seed, e.dataset, e.id, e.content_hash]))[:count]


def exclude_prior_audit(config, all_splits):
    """Freeze metadata-only exclusions before sampling; never inspect old outcomes.

    The list accepts campaign directories or registration.json files, relative to
    the working directory like the other configuration paths. All registered
    audit inputs are protected, including requests that never finished. Input
    hashes use BenchmarkExample's whitespace/casefold normalization, not semantic
    equivalence. IDs are dataset-qualified; input hashes match across datasets.
    """
    if "exclude_prior_campaigns" not in config:
        return all_splits, None
    requested = config["exclude_prior_campaigns"]
    if not isinstance(requested, list) or any(not isinstance(p, str) or not p.strip() for p in requested):
        raise ValueError("exclude_prior_campaigns must be a list of nonempty registration paths")
    protected_ids, protected_hashes, sources, seen = set(), set(), [], set()
    destination = (Path(config["output"]) / "registration.json").resolve()
    for configured_path in requested:
        source = Path(configured_path)
        if source.is_dir():
            source = source / "registration.json"
        source = source.resolve()
        if source == destination or source in seen:
            raise ValueError("Prior audit exclusion source cannot be this campaign or a duplicate path")
        if source.name != "registration.json":
            raise ValueError("Prior audit exclusions require a registration.json file or campaign directory")
        seen.add(source)
        try:
            raw = source.read_bytes()
            previous = json.loads(raw)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read prior audit registration: {source}") from error
        prior_config = previous.get("config") if isinstance(previous, dict) else None
        examples = previous.get("examples") if isinstance(previous, dict) else None
        rows = examples.get("audit") if isinstance(examples, dict) else None
        dataset = prior_config.get("dataset") if isinstance(prior_config, dict) else None
        if (not isinstance(dataset, str) or not dataset or
                prior_config.get("schema") != "inference_performance_v1" or
                not isinstance(rows, list) or not rows):
            raise ValueError(f"Invalid prior audit registration metadata: {source}")
        counts = prior_config.get("counts")
        if (not isinstance(counts, dict) or type(counts.get("audit")) is not int or
                counts["audit"] != len(rows)):
            raise ValueError(f"Prior registered audit count disagrees with its identities: {source}")
        identities, ids = [], set()
        for row in rows:
            example_id = row.get("id") if isinstance(row, dict) else None
            input_hash = row.get("input_hash") if isinstance(row, dict) else None
            if (not isinstance(example_id, str) or not example_id or example_id in ids or
                    not isinstance(input_hash, str) or len(input_hash) != 64 or
                    any(c not in "0123456789abcdef" for c in input_hash)):
                raise ValueError(f"Prior audit requires unique IDs and comparable SHA256 input hashes: {source}")
            ids.add(example_id)
            protected_ids.add((dataset, example_id))
            protected_hashes.add(input_hash)
            identities.append({"id": example_id, "input_hash": input_hash})
        sources.append({"configured_path": configured_path, "registration_path": str(source),
                        "registration_sha256": hashlib.sha256(raw).hexdigest(), "dataset": dataset,
                        "registered_audit_count": len(identities), "audit_examples": identities})

    def matched_by(example):
        return (["dataset_and_id"] if (example.dataset, example.id) in protected_ids else []) + (
            ["input_hash"] if example.content_hash in protected_hashes else [])

    for split in ("development", "selection"):
        if any(matched_by(example) for example in all_splits[split]):
            raise ValueError(f"Prior protected audit inputs overlap {split}; cannot use them for candidate selection")
    keep, excluded = [], []
    for example in all_splits["audit"]:
        reasons = matched_by(example)
        if reasons:
            excluded.append({"id": example.id, "input_hash": example.content_hash, "matched_by": reasons})
        else:
            keep.append(example)
    identities = lambda rows: sorted((e.dataset, e.id, e.content_hash) for e in rows)
    provenance = {"schema": "prior_registered_audit_exclusion_v1", "sources": sources,
        "policy": "all prior registered audit inputs, regardless of execution or score; dataset-qualified ID OR input hash",
        "input_identity_boundary": "BenchmarkExample whitespace/casefold input SHA256; semantic rewrites are not detected",
        "complete_historical_exposure_claimed": False,
        "unique_prior_audit_ids": len(protected_ids), "unique_prior_audit_input_hashes": len(protected_hashes),
        "audit_pool_before_count": len(all_splits["audit"]), "audit_pool_after_count": len(keep),
        "audit_pool_before_identity_sha256": digest(identities(all_splits["audit"])),
        "audit_pool_after_identity_sha256": digest(identities(keep)),
        "excluded_count": len(excluded), "excluded_examples": excluded,
        "development_selection_unchanged": True}
    return {**all_splits, "audit": keep}, provenance


def verify_exclusion_sources(provenance):
    """Reject source replacement/deletion even if the chosen audit sample survives."""
    for source in provenance["sources"] if provenance is not None else ():
        try:
            current = hashlib.sha256(Path(source["registration_path"]).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError("Prior audit exclusion source changed or disappeared") from error
        if current != source["registration_sha256"]:
            raise ValueError("Prior audit exclusion source changed during execution")


def summarize(rows):
    if not rows:
        raise ValueError("An empty evaluation cannot be summarized")
    return {"count": len(rows), "complete": sum(r["status"] == "complete" for r in rows),
            "mean_score": sum(r["score"] for r in rows) / len(rows),
            "charged_tokens": sum(r["charged_tokens"] for r in rows),
            "measured_tokens": sum(r["measured_tokens"] for r in rows),
            "reserved_unmetered_tokens": sum(r["reserved_unmetered_tokens"] for r in rows),
            "gpu_seconds": sum(r["gpu_seconds"] for r in rows),
            "metering_complete": all(r["metering_complete"] for r in rows)}


def paired_comparison(left, right, seed, repetitions=10000):
    if [r["example_id"] for r in left] != [r["example_id"] for r in right] or not left:
        raise ValueError("Paired comparison requires exactly the same ordered examples")
    differences = [a["score"] - b["score"] for a, b in zip(left, right)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                   for _ in range(repetitions))
    return {"mean_difference": sum(differences) / len(differences),
            "paired_bootstrap_95pct": [means[int(.025 * repetitions)], means[int(.975 * repetitions)]],
            "wins": sum(x > 0 for x in differences), "ties": sum(x == 0 for x in differences),
            "losses": sum(x < 0 for x in differences),
            "scope": "these registered examples and decoding seed only; not a SOTA claim"}


def run(config, *, executor=None, loaded=None, stop_after="audit"):
    config = json.loads(json.dumps(config, allow_nan=False))
    if stop_after not in {"register", "development", "selection", "audit"}:
        raise ValueError("Unknown stopping stage")
    if config.get("schema") != "inference_performance_v1":
        raise ValueError("Use the explicit inference_performance_v1 extension schema")
    for key in ("execution_cap", "total_token_budget"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config.get("finalists", 2)) is not int or config.get("finalists", 2) < 1:
        raise ValueError("finalists must be a positive integer")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "pending_request.json"
    if pending.exists():
        raise RuntimeError("Unresolved physical request: reconcile pending_request.json before replay; no free retry")
    lock = json.loads(Path(config["model_lock"]).read_text())
    if executor is None:
        from transformers import AutoTokenizer
        client = LocalModelClient(config["base_url"], config["served_model"])
        health = client.health()
        if health.get("revision") != lock["revision"] or not health.get("server_enforces_total_token_cap"):
            raise ValueError("Serving identity or enforceable token budget does not match registration")
        tokenizer = AutoTokenizer.from_pretrained(lock["local_path"], local_files_only=True,
            trust_remote_code=lock.get("trust_remote_code", False))
        executor = QwenExecutor(client, tokenizer, {})
    else:
        health = {"test_injected_executor": True, "revision": lock["revision"]}
    dataset = config["dataset"]
    all_splits = loaded if loaded is not None else {split: load_examples(dataset, split,
        data_root=config.get("data_root", "data"), verify=True) for split in ("development", "selection", "audit")}
    validate_disjoint_splits(all_splits)
    eligible_splits, audit_exclusions = exclude_prior_audit(config, all_splits)
    splits = {s: choose(eligible_splits[s], config["counts"][s], config["sample_seed"])
              for s in ("development", "selection", "audit")}
    candidates = {}
    for row in config["candidates"]:
        name = row["name"]
        if name in candidates:
            raise ValueError("Duplicate candidate name")
        candidates[name] = Candidate(name, row["artifacts"],
            {"generation": row["generation"], "seed": config["decode_seed"], "model_revision": lock["revision"]},
            semantic_kind="prompt_inference_extension", deterministic=False)
    baseline = config["baseline"]
    controls = config.get("audit_controls", [])
    if baseline not in candidates or any(x not in candidates for x in controls):
        raise ValueError("Baseline and audit controls must be registered candidates")
    scopes = {}
    for split, examples in splits.items():
        tasks = tuple(Task(e.id, {"prompt": e.prompt}, e.reference,
                           {"dataset": dataset, "example_id": e.id}) for e in examples)
        scopes[split] = EvaluationScope(f"performance-{dataset}-{split}", tasks,
            f"frozen-inference-extension:{lock['revision']}", f"native:{dataset}",
            split="dev" if split == "development" else split)
        for candidate in candidates.values():
            executor.validate_prompt_candidate(candidate, tasks, config["execution_cap"])
    sources = source_hashes()
    registration = {"config": config, "candidates": {k: v.manifest() for k, v in candidates.items()},
        "scope_ids": {k: v.uid for k, v in scopes.items()},
        "examples": {s: [{"id": e.id, "input_hash": e.content_hash} for e in rows]
                     for s, rows in splits.items()},
        "source_hashes": sources, "model_lock_sha256": hashlib.sha256(Path(config["model_lock"]).read_bytes()).hexdigest(),
        "extension_beyond_pdf_reference": True, "statistical_superiority_claimed": False,
        "selection_rule": "highest confirmation mean among development finalists; then lower tokens, then name",
        "incomplete_rule": "score zero and retain denominator/cost; no complete-performance claim",
        "external_sota_comparability": False}
    if audit_exclusions is not None:
        registration["audit_exclusions"] = audit_exclusions
    verify_exclusion_sources(audit_exclusions)
    path = output / "registration.json"
    if path.exists():
        if json.loads(path.read_text()) != registration:
            raise ValueError("Frozen registration changed; use a new campaign directory")
    else:
        save(path, registration)
        save(output / "health_at_registration.json", health)
    if stop_after == "register":
        return {"registered": str(path)}
    results_path = output / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    prior = {(r["split"], r["candidate"], r["example_id"]): r for r in rows}
    if len(prior) != len(rows):
        raise ValueError("Duplicate scored records")
    calls_path = output / "physical_calls.jsonl"
    physical_rows = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
    physical_accounting(physical_rows)  # Also reject duplicate physical request IDs across tasks.
    grouped = {}
    for record in physical_rows:
        key = record["split"], record["recipe"], record["example_id"]
        if key not in prior:
            raise ValueError("Orphan physical calls require reconciliation; no automatic replay")
        grouped.setdefault(key, []).append(record)
    for key, row in prior.items():
        split, name, example_id = key
        if split not in scopes or name not in candidates or example_id not in {t.name for t in scopes[split].tasks}:
            raise ValueError("Scored record is outside the frozen registration")
        if row["candidate_uid"] != candidates[name].uid:
            raise ValueError("Scored record candidate identity differs from registration")
        if row["status"] not in {"complete", "interrupted", "error"}:
            raise ValueError("Unknown scored record status")
        if not math.isfinite(row["score"]) or (row["status"] != "complete" and row["score"] != 0):
            raise ValueError("Incomplete records must retain a zero score in the denominator")
        accounting = physical_accounting(grouped.get(key, []))
        if any(row[field] != value for field, value in accounting.items()):
            raise ValueError("Scored record differs from its physical cost accounting")
        if row["status"] == "complete" and (not row.get("outcome") or not grouped.get(key) or not accounting["metering_complete"]):
            raise ValueError("A completed result requires a metered physical outcome")
        if row["charged_tokens"] > config["execution_cap"]:
            raise ValueError("Scored record exceeded the registered execution cap")
    total_charged = sum(r["charged_tokens"] for r in rows)
    if total_charged > config["total_token_budget"]:
        raise ValueError("Recorded costs exceed the registered total token budget")

    def verify_integrity():
        if source_hashes() != sources or hashlib.sha256(Path(config["model_lock"]).read_bytes()).hexdigest() != registration["model_lock_sha256"]:
            raise ValueError("Registered source or model lock changed during execution")
        verify_exclusion_sources(audit_exclusions)

    def evaluate(split, names):
        nonlocal total_charged
        for task in scopes[split].tasks:
            for name in names:
                key = split, name, task.name
                if key in prior:
                    continue
                verify_integrity()
                if total_charged + config["execution_cap"] > config["total_token_budget"]:
                    raise RuntimeError("Registered token budget cannot reserve the next complete execution")
                candidate = candidates[name]
                save(pending, {"split": split, "candidate": name, "candidate_uid": candidate.uid,
                    "example_id": task.name, "reservation_tokens": config["execution_cap"],
                    "started_unix": time.time(), "status": "in_flight_usage_not_yet_known"})
                start = len(executor.physical_records)
                begun = time.monotonic()
                try:
                    result = executor.run(candidate, task, scopes[split], config["execution_cap"])
                    records = executor.physical_records[start:]
                    accounting = physical_accounting(records)
                    charged = result.actual_cost
                    if charged != accounting["charged_tokens"] or charged > config["execution_cap"]:
                        raise ValueError("Executor cost disagrees with physical accounting or exceeds execution cap")
                    if result.status == "complete" and (not records or not accounting["metering_complete"]):
                        raise ValueError("A completed result requires metered physical calls")
                    verify_integrity()
                    row = {"split": split, "candidate": name, "candidate_uid": candidate.uid,
                        "example_id": task.name, "status": result.status,
                        "score": result.outcome.score if result.status == "complete" and result.outcome is not None else 0.,
                        "outcome": thaw(result.outcome.fields) if result.outcome else None,
                        "receipt": thaw(result.receipt), "error": result.error,
                        **accounting,
                        "elapsed_seconds": time.monotonic() - begun}
                except Exception as error:
                    # The pending reservation survives. Re-running cannot erase
                    # an admitted call whose physical receipt was lost.
                    records = executor.physical_records[start:]
                    append_jsonl(calls_path, [{**record, "split": split, "recipe": name, "example_id": task.name}
                                              for record in records])
                    save(output / "request_failure.json", {"type": type(error).__name__, "message": str(error),
                        "reservation_tokens": config["execution_cap"], "metering_complete": False})
                    raise
                append_jsonl(calls_path, [{**record, "split": split, "recipe": name, "example_id": task.name}
                                          for record in records])
                append_jsonl(results_path, [row])
                prior[key] = row
                rows.append(row)
                total_charged += charged
                pending.unlink()
                print(json.dumps({k: row[k] for k in ("split", "candidate", "example_id", "status", "score", "charged_tokens")}), flush=True)
        return {name: summarize([prior[(split, name, t.name)] for t in scopes[split].tasks]) for name in names}

    development = evaluate("development", list(candidates))
    save(output / "development_summary.json", development)
    if stop_after == "development":
        return development
    eligible = [k for k in candidates if development[k]["complete"] == len(scopes["development"].tasks)
                and development[k]["metering_complete"]]
    ranked = sorted(eligible, key=lambda k: (-development[k]["mean_score"], development[k]["charged_tokens"], k))
    finalists = list(dict.fromkeys([baseline, *controls, *ranked[:config.get("finalists", 2)]]))
    confirmation = evaluate("selection", finalists)
    save(output / "confirmation_summary.json", confirmation)
    eligible = [k for k in finalists if confirmation[k]["complete"] == len(scopes["selection"].tasks)
                and confirmation[k]["metering_complete"]]
    if not eligible:
        raise RuntimeError("No fully completed metered candidate can be selected")
    selected = min(eligible, key=lambda k: (-confirmation[k]["mean_score"], confirmation[k]["charged_tokens"], k))
    frozen = {"selected": selected, "candidate": candidates[selected].manifest(),
              "candidate_uid": candidates[selected].uid, "confirmation": confirmation,
              "registration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "audit_recipe_names": list(dict.fromkeys([baseline, *controls, selected]))}
    candidate_bytes = (json.dumps(frozen["candidate"], indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    frozen["selected_candidate_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    frozen["selected_candidate_file"] = "selected_candidate.json"
    frozen_path = output / "frozen_choice.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("Frozen choice changed after confirmation")
    if not frozen_path.exists():
        save(frozen_path, frozen)
    save_selected_candidate(output, frozen)
    if stop_after == "selection":
        return frozen
    audit = evaluate("audit", frozen["audit_recipe_names"])
    selected_rows = [prior[("audit", selected, t.name)] for t in scopes["audit"].tasks]
    comparisons = {k: paired_comparison(selected_rows, [prior[("audit", k, t.name)] for t in scopes["audit"].tasks],
                                       config["sample_seed"])
                   for k in frozen["audit_recipe_names"] if k != selected}
    report = {"selected": selected, "development": development, "confirmation": confirmation,
        "audit": audit, "selected_vs_controls": comparisons, "total_charged_tokens": total_charged,
        "complete": all(r["status"] == "complete" and r["metering_complete"] for r in rows),
        "source_unchanged": source_hashes() == sources,
        "extension_beyond_pdf_reference": True, "statistical_superiority_claimed": False,
        "sota_claimed": False}
    save(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-after", choices=("register", "development", "selection", "audit"), default="audit")
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()), stop_after=args.stop_after), indent=2))
