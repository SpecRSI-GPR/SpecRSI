"""Optional bounded numerical optimization of immutable scientific candidates.

This is a performance extension, not the learned PDF VATS/GVO controller.
The native simulator, its seeds, and its validity checks remain host-owned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from specrsi.core.models import Candidate, digest, thaw
from specrsi.science.executor import ScienceExecutor, make_candidate, make_scope, mutate
from specrsi.science.registry import ROOT, load_config


METHODS = ("differential_evolution", "random_search", "stock_mutation")
# These are the exact bounds enforced by science.irp.simulate, not new limits.
IRP_BOUNDS = (("target_days", 0.0, 4.0), ("safety_stock", 0.0, 30.0),
              ("forecast_blend", 0.0, 1.0))


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (20260928,)
    development_calls: int = 25
    execution_cap_ms: int = 30000
    case_id: int = 0
    domain: str = "irp"

    def __post_init__(self):
        if self.domain != "irp":
            raise ValueError("Only the registered IRP parameter family is supported")
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)):
            raise ValueError("Distinct integer seeds in [0, 2**32) are required")
        if type(self.development_calls) is not int or self.development_calls < 7:
            raise ValueError("At least seven development calls are required")
        if type(self.execution_cap_ms) is not int or self.execution_cap_ms < 1:
            raise ValueError("Positive native execution cap required")
        if type(self.case_id) is not int or self.case_id < 0:
            raise ValueError("Nonnegative case ID required")


def _write(path: Path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)



def implementation_snapshot(root=None):
    """Pin all imported implementation sources and registered configuration.

    The native evaluator identity is untouched. This extra campaign-level guard
    includes benchmarks/science.py, which is outside the native source registry.
    """
    root = Path(root or ROOT)
    paths = (set(root.glob("src/specrsi/**/*.py"))
             | set(root.glob("src/specrsi/**/*.cc"))
             | set(root.glob("configs/science/**/*.json")))
    paths.update(path for path in (root / "pyproject.toml", root / "scripts/run_science_performance.py")
                 if path.is_file())
    snapshot = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(paths)}
    snapshot["running_optimizer_source"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return snapshot


def assert_implementation(snapshot, root=None):
    if implementation_snapshot(root) != snapshot:
        raise RuntimeError("Campaign implementation/configuration changed; results cannot be adopted")


def _journal(path, event):
    # Persist before launching paid work, including before KeyboardInterrupt or
    # a worker exception can leave the result incomplete.
    with Path(path).open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parameters_from_vector(vector):
    values = list(vector)
    if len(values) != len(IRP_BOUNDS):
        raise ValueError("Incorrect parameter dimension")
    parameters = {}
    for (name, lower, upper), value in zip(IRP_BOUNDS, values):
        value = float(value)
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} violates registered bounds")
        parameters[name] = value
    return parameters


def _candidate_from_manifest(manifest):
    return Candidate(**manifest)


class NativeMeter:
    """Every attempted candidate consumes one call and retains its full receipt."""

    def __init__(self, directory, config, scopes, executor=None, *, source_snapshot=None):
        self.directory = Path(directory)
        self.config, self.scopes = config, scopes
        self.executor = executor or ScienceExecutor()
        self.calls = []
        self.phase = "development"
        self.frozen = None
        self.source_snapshot = source_snapshot if source_snapshot is not None else implementation_snapshot()

    def begin_selection(self):
        if self.phase != "development":
            raise ValueError("Selection follows development exactly once")
        self.phase = "selection"

    def freeze(self, artifact):
        if self.phase != "selection":
            raise ValueError("Only selected artifacts can be frozen")
        self.frozen = artifact
        _write(self.directory / "frozen.json", artifact)
        self.frozen_sha256 = hashlib.sha256((self.directory / "frozen.json").read_bytes()).hexdigest()
        self.audit_uids = frozenset([artifact["baseline"]["uid"],
                                     *(row["selected_uid"] for row in artifact["arms"])])
        self.phase = "audit"

    def evaluate(self, candidate, *, phase, arm):
        if phase != self.phase:
            raise ValueError("Cannot tune on audit or evaluate outside the registered phase")
        if phase == "audit":
            if hashlib.sha256((self.directory / "frozen.json").read_bytes()).hexdigest() != self.frozen_sha256:
                raise ValueError("Frozen artifact was changed")
            if candidate.uid not in self.audit_uids:
                raise ValueError("Audit candidate was not frozen before audit")
        scope = self.scopes[phase]
        assert_implementation(self.source_snapshot)
        index = len(self.calls)
        pending = {"event": "pending", "index": index, "phase": phase, "arm": arm,
                   "candidate_uid": candidate.uid, "candidate": candidate.manifest(),
                   "scope_uid": scope.uid, "reserved_cap_ms": self.config.execution_cap_ms,
                   "unix_time_ns": time.time_ns()}
        _journal(self.directory / "execution_journal.jsonl", pending)
        started = time.monotonic()
        # Exactly one complete native suite per call; no partial-scenario surrogate.
        result = None
        try:
            result = self.executor.run(candidate, scope.tasks[0], scope, self.config.execution_cap_ms)
            wall_seconds = time.monotonic() - started
            valid = (result.status == "complete" and result.outcome is not None
                     and result.outcome.admissible and result.outcome.fields.get("valid") is True)
            score = float(result.outcome.fields["native_score"]) if valid else None
            if score is not None and not math.isfinite(score):
                raise ValueError("Nonfinite native result")
            row = {"index": index, "phase": phase, "arm": arm,
                   "candidate_uid": candidate.uid, "candidate": candidate.manifest(),
                   "scope_uid": scope.uid, "status": result.status, "valid": valid,
                   "native_score": score, "executor_cost_ms": result.actual_cost,
                   "physical_wall_seconds": wall_seconds, "error": result.error,
                   "receipt": thaw(result.receipt)}
        except BaseException as exc:
            wall_seconds = time.monotonic() - started
            known_cost = getattr(result, "actual_cost", None)
            if type(known_cost) is not int or known_cost < 0:
                known_cost = None
            row = {**pending, "event": "failed", "status": "unexpected_exception", "valid": False,
                   "native_score": None, "executor_cost_ms": known_cost,
                   "physical_wall_seconds": wall_seconds,
                   "observed_physical_cost_ms": math.ceil(wall_seconds * 1000),
                   "executor_cost_unknown": known_cost is None,
                   "worker_completion_unknown": result is None,
                   "error": f"{type(exc).__name__}: {exc}", "receipt": {}}
            self.calls.append(row)
            _journal(self.directory / "evaluations.jsonl", row)
            _journal(self.directory / "execution_journal.jsonl", row)
            # Unknown worker cost is not silently zero or a fabricated complete
            # expense. Stop before selection/audit and retain the paid attempt.
            raise
        self.calls.append(row)
        _journal(self.directory / "evaluations.jsonl", row)
        _journal(self.directory / "execution_journal.jsonl", {
            "event": "finished", "index": index, "phase": phase, "arm": arm,
            "candidate_uid": candidate.uid, "status": row["status"], "valid": row["valid"],
            "executor_cost_ms": row["executor_cost_ms"], "physical_wall_seconds": wall_seconds,
            "unix_time_ns": time.time_ns()})
        assert_implementation(self.source_snapshot)
        return row


def _rank(row):
    return row["native_score"] if row["valid"] else -math.inf


def optimize(method, seed, baseline, config, meter):
    """Develop one artifact with exactly the declared real-evaluation budget."""
    if method not in METHODS:
        raise ValueError("Unknown numerical optimization method")
    arm = f"{method}-seed-{seed}"
    rows = [meter.evaluate(baseline, phase="development", arm=arm)]
    best = rows[0]
    proposal_seconds = 0.0
    rng = random.Random(seed)

    class CallLimit(Exception):
        pass

    def objective(vector):
        nonlocal best
        if len(rows) >= config.development_calls:
            raise CallLimit()
        candidate = make_candidate(config.domain, parameters_from_vector(vector),
                                   f"{arm}-proposal-{len(rows)}", parent_id=best["candidate_uid"])
        row = meter.evaluate(candidate, phase="development", arm=arm)
        rows.append(row)
        if _rank(row) > _rank(best):
            best = row
        return -_rank(row)

    started = time.monotonic()
    if method == "differential_evolution":
        from scipy.optimize import differential_evolution
        # Six population members; the initial candidate is already paid above.
        # No polish: local minimization must not silently add evaluator calls.
        try:
            differential_evolution(objective, [(lo, hi) for _, lo, hi in IRP_BOUNDS],
                popsize=2, maxiter=config.development_calls, seed=seed, polish=False,
                updating="immediate", workers=1, tol=0.0, atol=0.0)
        except CallLimit:
            pass
    # A solver can terminate early on a flat objective. Fill the SAME fixed
    # budget with prespecified uniform samples, without looking at audit.
    while len(rows) < config.development_calls:
        if method == "stock_mutation":
            parameters = mutate(config.domain, best["candidate"]["artifacts"]["parameters"], rng)
            vector = [parameters[name] for name, _, _ in IRP_BOUNDS]
        else:
            vector = [rng.uniform(lo, hi) for _, lo, hi in IRP_BOUNDS]
        objective(vector)
    proposal_seconds = max(0.0, time.monotonic() - started
                           - sum(row["physical_wall_seconds"] for row in rows[1:]))
    return {"method": method, "seed": seed, "arm": arm,
            "development_calls": len(rows), "development_best": best["native_score"],
            "development_best_manifest": best["candidate"],
            "development_best_uid": best["candidate_uid"],
            "proposal_and_bookkeeping_seconds": proposal_seconds}


def run_experiment(directory, config=ExperimentConfig(), *, executor=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use a fresh output directory; previous artifacts cannot be overwritten")
    scopes = {phase: make_scope(config.domain, phase, (config.case_id,), native_utility=True)
              for phase in ("development", "selection", "audit")}
    baseline = make_candidate(config.domain, load_config(config.domain, "development", config.case_id)["baseline"],
                              "numerical-extension-baseline")
    identity = {phase: scopes[phase].tasks[0].metadata["evaluator_identity"] for phase in scopes}
    if len(set(identity.values())) != 3:
        raise ValueError("Development, selection, and audit identities must differ")
    order = [(method, seed) for seed in config.seeds for method in METHODS]
    random.Random(20260928).shuffle(order)
    source_snapshot = implementation_snapshot()
    protocol = {"kind": "numerical_performance_extension_not_learned_pdf_controller",
                "config": asdict(config), "methods": METHODS, "bounds": IRP_BOUNDS,
                "order": order, "implementation_snapshot": source_snapshot, "scopes": {phase: scope.manifest() for phase, scope in scopes.items()},
                "baseline": baseline.manifest(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "selection_rule": "development winner versus baseline; strict native-score improvement required",
                "audit_rule": "evaluate every frozen arm and its baseline once; no reselection",
                "budget_rule": "equal complete-native-suite call count, measured wall costs reported separately",
                "sota_claimed": False, "original_paper_replication": False}
    _write(directory / "registration.json", protocol)
    started = time.monotonic()
    meter = NativeMeter(directory, config, scopes, executor, source_snapshot=source_snapshot)
    developed = []
    for method, seed in order:
        row = optimize(method, seed, baseline, config, meter)
        developed.append(row)
        _write(directory / "development.json", developed)
        print(json.dumps({"phase": "development", "arm": row["arm"],
                          "score": row["development_best"], "calls": row["development_calls"]}), flush=True)
    meter.begin_selection()
    selected = []
    for row in developed:
        reference = meter.evaluate(baseline, phase="selection", arm=row["arm"])
        candidate = _candidate_from_manifest(row["development_best_manifest"])
        result = meter.evaluate(candidate, phase="selection", arm=row["arm"])
        chosen = result if _rank(result) > _rank(reference) else reference
        if not chosen["valid"]:
            raise RuntimeError("No admissible baseline or optimized candidate on selection")
        selected.append({**row, "selected_manifest": chosen["candidate"],
                         "selected_uid": chosen["candidate_uid"], "selection_score": chosen["native_score"],
                         "selection_baseline": reference["native_score"]})
    frozen = {"registration_sha256": digest(protocol), "baseline": {"uid": baseline.uid, "manifest": baseline.manifest()},
              "arms": selected, "selection_finished_before_audit": True}
    meter.freeze(frozen)
    results = []
    for row in selected:
        reference = meter.evaluate(baseline, phase="audit", arm=row["arm"])
        result = meter.evaluate(_candidate_from_manifest(row["selected_manifest"]), phase="audit", arm=row["arm"])
        arm_calls = [call for call in meter.calls if call["arm"] == row["arm"]]
        gain = result["native_score"] - reference["native_score"] if result["valid"] and reference["valid"] else None
        results.append({**row, "audit_valid": result["valid"], "audit_score": result["native_score"],
                        "audit_baseline": reference["native_score"], "audit_gain": gain,
                        "physical_calls": len(arm_calls), "executor_cost_ms": sum(c["executor_cost_ms"] for c in arm_calls),
                        "physical_wall_seconds": sum(c["physical_wall_seconds"] for c in arm_calls)})
    summary = {"kind": protocol["kind"], "sota_claimed": False, "original_paper_replication": False,
               "scope": f"registered prospective native {config.domain}, case {config.case_id}",
               "native_metric": "mean(1-cost/(12*demand))-0.15*population_std",
               "config": asdict(config), "results": results,
               "physical_calls": len(meter.calls), "elapsed_seconds": time.monotonic()-started,
               "executor_cost_ms": sum(row["executor_cost_ms"] for row in meter.calls),
               "failed_or_invalid_calls": sum(not row["valid"] for row in meter.calls),
               "frozen_sha256": meter.frozen_sha256,
               "evaluation_log_sha256": hashlib.sha256((directory / "evaluations.jsonl").read_bytes()).hexdigest()}
    assert_implementation(source_snapshot)
    summary["execution_journal_sha256"] = hashlib.sha256((directory / "execution_journal.jsonl").read_bytes()).hexdigest()
    _write(directory / "summary.json", summary)
    return summary
