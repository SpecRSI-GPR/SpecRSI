import json
import specrsi.science_optimization as optimization
import pytest

from specrsi.core.execution import ExecutionResult
from specrsi.core.models import Outcome
from specrsi.science.executor import make_candidate, make_scope
from specrsi.science_optimization import (
    ExperimentConfig, NativeMeter, parameters_from_vector, run_experiment,
)


@pytest.fixture(autouse=True)
def stable_unit_test_sources(monkeypatch, request):
    # Other tests isolate budgets/phase gates; the dedicated tests below check
    # real recursive snapshots against a tiny disposable implementation tree.
    if not request.node.name.startswith("test_full_source_guard"):
        monkeypatch.setattr(optimization, "implementation_snapshot", lambda root=None: {"test": "fixed"})


class Objective:
    def __init__(self, audit_sign=1, invalid=False):
        self.calls = []
        self.audit_sign, self.invalid = audit_sign, invalid

    def run(self, candidate, task, scope, cap):
        phase = task.inputs["scenario_set"]
        self.calls.append((phase, candidate.uid))
        p = candidate.artifacts["parameters"]
        score = -(p["target_days"] - 2.1)**2 - (p["safety_stock"] - 5)**2 / 100
        if phase == "audit":
            score *= self.audit_sign
        # Native scores can be negative: invalid must not beat valid negatives.
        invalid = self.invalid and p["target_days"] > 2.5
        return ExecutionResult("complete", 7, Outcome(0.2, {
            "native_score": None if invalid else score, "valid": not invalid,
            "evaluator_identity": task.metadata["evaluator_identity"],
        }, admissible=not invalid))


def test_matched_call_budgets_and_audit_cannot_change_selection(tmp_path):
    config = ExperimentConfig(development_calls=7)
    one, two = Objective(1, True), Objective(-1, True)
    first = run_experiment(tmp_path / "one", config, executor=one)
    second = run_experiment(tmp_path / "two", config, executor=two)
    assert first["physical_calls"] == second["physical_calls"] == 3 * 11
    assert all(row["physical_calls"] == 11 for row in first["results"])
    assert all(row["executor_cost_ms"] == 77 for row in first["results"])
    assert [r["selected_uid"] for r in first["results"]] == [r["selected_uid"] for r in second["results"]]
    assert [r["selection_score"] for r in first["results"]] == [r["selection_score"] for r in second["results"]]
    assert [phase for phase, _ in one.calls] == ["development"]*21 + ["selection"]*6 + ["audit"]*6
    assert all(row["selection_score"] is not None for row in first["results"])
    for row in first["results"]:
        assert row["audit_gain"] == pytest.approx(-next(r for r in second["results"] if r["arm"] == row["arm"])["audit_gain"])


def test_audit_gate_rejects_unfrozen_changes_and_phase_backtracking(tmp_path):
    config = ExperimentConfig(development_calls=7)
    scopes = {phase: make_scope("irp", phase) for phase in ("development", "selection", "audit")}
    meter = NativeMeter(tmp_path, config, scopes, Objective())
    root = make_candidate("irp")
    with pytest.raises(ValueError, match="phase"):
        meter.evaluate(root, phase="audit", arm="arm")
    meter.begin_selection()
    artifact = {"baseline": {"uid": root.uid}, "arms": []}
    meter.freeze(artifact)
    child = make_candidate("irp", {"target_days": 2, "safety_stock": 0, "forecast_blend": 1})
    # Mutating the caller's dict must not alter the sealed admission set.
    artifact["arms"].append({"selected_uid": child.uid})
    with pytest.raises(ValueError, match="not frozen"):
        meter.evaluate(child, phase="audit", arm="arm")
    with pytest.raises(ValueError, match="phase"):
        meter.evaluate(root, phase="development", arm="arm")
    (tmp_path / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        meter.evaluate(root, phase="audit", arm="arm")


@pytest.mark.parametrize("vector", [[1, 2], [float("nan"), 2, 1], [1, 31, 1], [1, 0, -0.1]])
def test_only_declared_finite_native_bounds(vector):
    with pytest.raises(ValueError):
        parameters_from_vector(vector)


def test_recorded_failure_still_costs_a_call(tmp_path):
    class Failed:
        def run(self, candidate, task, scope, cap):
            return ExecutionResult("interrupted", cap, error="native timeout")
    config = ExperimentConfig(development_calls=7)
    scopes = {"development": make_scope("irp", "development")}
    meter = NativeMeter(tmp_path, config, scopes, Failed())
    row = meter.evaluate(make_candidate("irp"), phase="development", arm="arm")
    assert row["native_score"] is None and not row["valid"]
    assert row["executor_cost_ms"] == config.execution_cap_ms
    assert len(meter.calls) == 1
    assert len((tmp_path / "evaluations.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("failure", [RuntimeError("worker failed"), KeyboardInterrupt()])
def test_unexpected_worker_exception_is_journaled_before_propagation(tmp_path, failure):
    class Exploding:
        def run(self, candidate, task, scope, cap):
            events = [json.loads(line) for line in (tmp_path / "execution_journal.jsonl").read_text().splitlines()]
            assert events[-1]["event"] == "pending"
            raise failure
    meter = NativeMeter(tmp_path, ExperimentConfig(development_calls=7),
                        {"development": make_scope("irp", "development")}, Exploding())
    with pytest.raises(type(failure)):
        meter.evaluate(make_candidate("irp"), phase="development", arm="arm")
    events = [json.loads(line) for line in (tmp_path / "execution_journal.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["pending", "failed"]
    assert len(meter.calls) == 1
    assert meter.calls[0]["executor_cost_ms"] is None
    assert meter.calls[0]["executor_cost_unknown"]
    assert meter.calls[0]["physical_wall_seconds"] >= 0
    assert meter.calls[0]["observed_physical_cost_ms"] >= 0
    assert not (tmp_path / "summary.json").exists()


def test_malformed_result_retains_known_paid_cost(tmp_path):
    class Malformed:
        def run(self, candidate, task, scope, cap):
            return ExecutionResult("complete", 19, Outcome(.5, {"valid": True}))
    meter = NativeMeter(tmp_path, ExperimentConfig(development_calls=7),
                        {"development": make_scope("irp", "development")}, Malformed())
    with pytest.raises(KeyError):
        meter.evaluate(make_candidate("irp"), phase="development", arm="arm")
    assert meter.calls[0]["executor_cost_ms"] == 19
    assert not meter.calls[0]["executor_cost_unknown"]


@pytest.mark.parametrize("during_call", [False, True])
def test_full_source_guard_includes_imported_benchmark_before_and_after_call(tmp_path, monkeypatch, during_call):
    source = tmp_path / "implementation" / "src" / "specrsi" / "benchmarks" / "science.py"
    source.parent.mkdir(parents=True)
    source.write_text("native_metric_version = 1\n")
    snapshot = optimization.implementation_snapshot
    monkeypatch.setattr(optimization, "implementation_snapshot", lambda root=None: snapshot(tmp_path / "implementation"))
    calls = []
    class Changing:
        def run(self, candidate, task, scope, cap):
            calls.append(candidate.uid)
            source.write_text("native_metric_version = 2\n")
            return Objective().run(candidate, task, scope, cap)
    output = tmp_path / "output"
    output.mkdir()
    meter = NativeMeter(output, ExperimentConfig(development_calls=7),
                        {"development": make_scope("irp", "development")}, Changing())
    if not during_call:
        source.write_text("native_metric_version = 2\n")
    with pytest.raises(RuntimeError, match="implementation/configuration changed"):
        meter.evaluate(make_candidate("irp"), phase="development", arm="arm")
    assert len(calls) == int(during_call)
    assert len(meter.calls) == int(during_call)
    if during_call:
        assert meter.calls[0]["executor_cost_ms"] == 7
        events = [json.loads(line) for line in (output / "execution_journal.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["pending", "finished"]
