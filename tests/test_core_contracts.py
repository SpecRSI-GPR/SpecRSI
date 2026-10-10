from dataclasses import replace
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from specrsi.core import (AcquisitionCaps, BudgetExceeded, BudgetLedger, Candidate, ConservativeChecker,
                      CostCapViolation, EvaluationScope, EvidenceConflict, ExecutionResult, FrozenMap,
                      Ledger, Outcome, PureIntegerExecutor, ResultAcquirer, Task, accept)


def fixture():
    tasks = (Task("read", {"route": "read", "x": 2}, 2), Task("write", {"route": "write", "x": 2}, 3))
    scope = EvaluationScope("selection", tasks, "pure-int-v1", "integer-equality-v1", ("answer", "primitives"))
    parent = Candidate("parent", {"routes": {"read": ("x",), "write": ("x",)}},
                       {"contract": "pure-int-v1"}, "pure_integer", True)
    child = Candidate("child", {"routes": {"read": ("x",), "write": ("add", ("x",), ("const", 1))}},
                      parent.runtime, "pure_integer", True, parent_id=parent.uid)
    ledger = Ledger(receipt_secret=b"test-only-host-secret-32-bytes!!!")
    budget = BudgetLedger(1000)
    acquirer = ResultAcquirer(ledger, budget, PureIntegerExecutor())
    return tasks, scope, parent, child, ledger, budget, acquirer


def test_candidate_and_task_inputs_are_deeply_frozen():
    mutable = {"routes": {"read": ["x"]}}
    candidate = Candidate("test", mutable)
    original = candidate.uid
    mutable["routes"]["read"].append("effect")
    assert candidate.uid == original
    with pytest.raises(TypeError):
        candidate.artifacts["new"] = 1
    with pytest.raises(TypeError):
        FrozenMap({"nonfinite": float("nan")})


def test_proven_reuse_imports_complete_scored_effects_and_failures():
    tasks, scope, parent, child, ledger, budget, acquirer = fixture()
    for task in tasks:
        acquirer.acquire(parent, task, scope, (), AcquisitionCaps(10))
    inherited = acquirer.acquire(child, tasks[0], scope, (parent,), AcquisitionCaps(10))
    executed = acquirer.acquire(child, tasks[1], scope, (parent,), AcquisitionCaps(10))
    assert inherited.record.route == "P" and inherited.actual_cost == 1
    assert inherited.record.outcome.fields["primitives"] == 1
    assert executed.record.route == "U" and executed.actual_cost == 4
    # An unchanged failing source remains a failure, even on a new child identity.
    same_bad = replace(parent, name="same-bad", parent_id=parent.uid)
    failure = acquirer.acquire(same_bad, tasks[1], scope, (parent,), AcquisitionCaps(10))
    assert failure.record.route == "P" and failure.record.outcome.score == 0
    assert ledger.audit_provenance() == 5


def test_equality_without_source_supplies_no_evidence():
    tasks, scope, parent, child, ledger, _, _ = fixture()
    certificate = ledger.checker.check(parent, child, tasks[0], scope, 1).certificate
    with pytest.raises(ValueError, match="without valid source"):
        ledger.inherit(parent, child, tasks[0], scope, certificate)
    assert ledger.missing(child, scope) == tasks


def test_scope_runtime_scorer_and_scored_effect_changes_are_u():
    tasks, scope, parent, child, ledger, _, _ = fixture()
    checker = ledger.checker
    variants = [replace(child, runtime={"contract": "pure-int-v2"}),
                replace(child, deterministic=False),
                replace(child, artifacts={**dict(child.artifacts), "prompt": "global"}),
                replace(child, artifacts={**dict(child.artifacts), "global_config": "changed"})]
    for candidate in variants:
        assert checker.check(parent, candidate, tasks[0], scope, 1).status == "U"
    assert checker.check(parent, child, tasks[0], replace(scope, scorer_id="different"), 1).status == "U"
    assert checker.check(parent, child, tasks[0], replace(scope, scored_fields=("answer", "wall_latency")), 1).status == "U"
    checked = checker.check(parent, child, tasks[0], scope, 1).certificate
    assert not checker.verify(checked, parent, child, tasks[0], replace(scope, replay={"epoch": 2}))


def test_opaque_identical_artifacts_never_infer_determinism():
    task = Task("llm", {"prompt": "2+2"}, "4")
    scope = EvaluationScope("llm", (task,), "hosted-or-llm", "answer-v1")
    parent = Candidate("a", {"prompt": "2+2"}, {"seed": 42}, deterministic=True)
    child = replace(parent, name="b", parent_id=parent.uid)
    assert ConservativeChecker().check(parent, child, task, scope, 1).status == "U"


def test_registered_exact_contract_requires_full_artifact_and_runtime_identity():
    task = Task("fixed", {}, 1)
    scope = EvaluationScope("fixed", (task,), "trusted-immutable-v1", "fixed-v1")
    checker = ConservativeChecker(exact_contracts={"trusted-immutable-v1": lambda c, t, s: c.semantic_kind == "fixed"})
    parent = Candidate("a", {"answer": 1}, {"contract": "trusted-immutable-v1"}, "fixed", True)
    child = replace(parent, name="b", parent_id=parent.uid)
    assert checker.check(parent, child, task, scope, 1).status == "P"
    assert checker.check(parent, replace(child, artifacts={"answer": 1, "other": 0}), task, scope, 1).status == "U"


def test_interrupted_check_has_no_binary_label_and_execution_can_finish_u():
    tasks, scope, parent, child, ledger, _, acquirer = fixture()
    acquirer.acquire(parent, tasks[0], scope, (), AcquisitionCaps(10))
    result = acquirer.acquire(child, tasks[0], scope, (parent,), AcquisitionCaps(10, check=0))
    assert result.checks[0].status == "interrupted"
    assert result.record.route == "U"


def test_interrupted_execution_is_paid_missing_and_can_retry():
    tasks, scope, _, child, ledger, budget, acquirer = fixture()
    partial = acquirer.acquire(child, tasks[1], scope, (), AcquisitionCaps(2))
    assert partial.status == "interrupted" and partial.actual_cost == 2 and partial.record is None
    assert ledger.missing(child, scope) == tasks
    completed = acquirer.acquire(child, tasks[1], scope, (), AcquisitionCaps(3))
    assert completed.status == "complete" and budget.spent == 5


def test_existing_records_are_not_reexecuted_even_concurrently():
    tasks, scope, parent, _, ledger, budget, acquirer = fixture()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: acquirer.acquire(parent, tasks[0], scope, (), AcquisitionCaps(10, lookup=1)), range(8)))
    assert sum(not r.from_cache for r in results) == 1
    assert budget.spent == 9 and len(ledger.records) == 1


def test_forecasts_and_partial_results_have_no_complete_score():
    tasks, scope, parent, child, ledger, _, acquirer = fixture()
    acquirer.acquire(parent, tasks[0], scope, (), AcquisitionCaps(10))
    with pytest.raises(ValueError, match="partial candidates"):
        ledger.score(parent, scope)
    assert ledger.missing(child, scope) == tasks
    with pytest.raises(ValueError):
        Outcome(1., {"answer": 2}).validate(scope)


def test_native_nonadditive_aggregation_requires_all_outcomes():
    tasks, scope, parent, _, ledger, _, acquirer = fixture()
    native = replace(scope, aggregation_id="all_success_v1")
    acquirer.acquire(parent, tasks[0], native, (), AcquisitionCaps(10))
    with pytest.raises(ValueError):
        ledger.score(parent, native)
    acquirer.acquire(parent, tasks[1], native, (), AcquisitionCaps(10))
    assert ledger.score(parent, native) == 0.
    assert not ledger.complete(parent, scope)  # no cross-scope leakage


def test_conflicts_quarantine_descendant_coverage():
    tasks, scope, parent, child, ledger, _, acquirer = fixture()
    acquirer.acquire(parent, tasks[0], scope, (), AcquisitionCaps(10))
    acquirer.acquire(child, tasks[0], scope, (parent,), AcquisitionCaps(10))
    with pytest.raises(EvidenceConflict):
        ledger.admit_direct(parent, tasks[0], scope, Outcome(0., {"answer": 0, "primitives": 1}), {})
    assert ledger.get(parent, tasks[0], scope) is None
    assert ledger.get(child, tasks[0], scope) is None


def test_snapshot_roundtrip_and_tampered_direct_outcome_rejected():
    tasks, scope, parent, child, ledger, _, acquirer = fixture()
    for task in tasks:
        acquirer.acquire(parent, task, scope, (), AcquisitionCaps(10))
        acquirer.acquire(child, task, scope, (parent,), AcquisitionCaps(10))
    secret = b"test-only-host-secret-32-bytes!!!"
    snapshot = ledger.to_json()
    restored = Ledger.from_json(snapshot, receipt_secret=secret)
    assert restored.audit_provenance() == 4 and restored.score(child, scope) == 1.
    forged = json.loads(snapshot)
    forged["records"][0]["outcome"]["score"] = 0.
    with pytest.raises(ValueError, match="invalid provenance"):
        Ledger.from_json(json.dumps(forged), receipt_secret=secret)
    with pytest.raises(ValueError):
        Ledger.from_json(snapshot, receipt_secret=b"different-host-receipt-secret")


def test_acceptance_strict_margin_hard_constraints_and_audit_isolation():
    tasks, scope, parent, child, ledger, _, acquirer = fixture()
    for task in tasks:
        acquirer.acquire(parent, task, scope, (), AcquisitionCaps(10))
        acquirer.acquire(child, task, scope, (), AcquisitionCaps(10))
    assert accept(parent, [child], ledger, scope, margin=.5).selected == parent
    assert accept(parent, [child], ledger, scope, margin=.49).selected == child
    assert accept(parent, [child], ledger, scope, hard_constraint=lambda c: c == parent).selected == parent
    with pytest.raises(ValueError, match="audit"):
        accept(parent, [child], ledger, replace(scope, split="audit"))


def test_budget_atomicity_refunds_and_exactly_once_transfers():
    budget = BudgetLedger(20)
    budget.reserve("finalist-a", 8)
    budget.reserve("finalist-b", 8)
    with pytest.raises(BudgetExceeded):
        budget.reserve_work({"setup": (2, "setup")}, {"finalist-a": 11})
    assert budget.reserved == {"finalist-a": 8, "finalist-b": 8}
    budget.reserve_work({"setup": (1, "setup")}, {"finalist-a": 10})
    budget.settle("setup", 1)
    budget.settle("finalist-a", 3)
    assert budget.available == 8 and budget.pending == 8 and budget.spent == 4
    with pytest.raises(ValueError):
        budget.settle("finalist-a", 3)
    budget.settle("finalist-b", 8)
    assert budget.invariant() and budget.spent == 12


def test_backend_cap_breach_is_visible_and_halts_budget():
    class BrokenExecutor:
        def run(self, candidate, task, scope, cap, resume_token=None):
            return ExecutionResult("complete", cap + 1, Outcome(1., {"answer": 2, "primitives": 1}))
    tasks, scope, parent, _, ledger, budget, _ = fixture()
    acquirer = ResultAcquirer(ledger, budget, BrokenExecutor())
    with pytest.raises(CostCapViolation):
        acquirer.acquire(parent, tasks[0], scope, (), AcquisitionCaps(2))
    assert budget.spent == 3 and budget.breached and budget.available == 0
    assert ledger.get(parent, tasks[0], scope) is None


def test_unmetered_exception_is_bound_not_reported_actual_cost():
    class CrashedExecutor:
        def run(self, *args, **kwargs):
            raise RuntimeError("worker vanished")
    tasks, scope, parent, _, ledger, budget, _ = fixture()
    acquirer = ResultAcquirer(ledger, budget, CrashedExecutor())
    with pytest.raises(RuntimeError, match="worker vanished"):
        acquirer.acquire(parent, tasks[0], scope, (), AcquisitionCaps(10))
    assert not budget.cost_valid and budget.charges[-1].cost_kind == "conservative_bound"
    assert not budget.invariant() and budget.available == 0
