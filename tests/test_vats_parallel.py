"""Real overlap, synchronous admission and exact billing for Appendix F.4."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import pytest

from specrsi.core import CostCapViolation, PureIntegerExecutor
from specrsi.core.execution import ExecutionResult
from specrsi.core.jobs import ExecutedJobOutcome, JobExecutionResult, SetupExecutionResult
from specrsi.core.vats import VATSConfig
from specrsi.planning.completion import CompletionProblem, VerificationJob
from test_core_vats import make_vats


def parallel_fixture(executor=None, backend=None, *, workers=2):
    engine = make_vats(config=VATSConfig(workers=workers, execution_cap=10, max_queries=0,
        preparation_cap=5, solver_backend="exhaustive"), verification_backend=backend)
    child = replace(engine.state.incumbent, name="parallel-child", parent_id=engine.state.incumbent.uid,
                    artifacts={"routes": {"read": ("x",), "write": ("add", ("x",), ("const", 1))}})
    engine.state.candidates[child.uid] = child
    engine.state.ledger.register(child, engine.state.scope)
    if executor is not None:
        engine.executor = engine.acquirer.executor = executor
    problem = engine._catalog(engine.state)
    plan = engine._verify_plan(tuple(problem.by_id), problem)
    return engine, child, problem, plan


def test_parallel_direct_calls_really_overlap_and_host_admission_waits_for_whole_wave(monkeypatch):
    barrier = threading.Barrier(2, timeout=5)
    owner = threading.get_ident()

    class Executor(PureIntegerExecutor):
        def run(self, *args, **kwargs):
            assert len(engine.state.ledger.records) == 2
            assert engine.state.budget.pending == 20
            barrier.wait()
            time.sleep(.03)
            assert len(engine.state.ledger.records) == 2
            return super().run(*args, **kwargs)

    engine, child, problem, plan = parallel_fixture(Executor())
    admit = engine.state.ledger.admit_direct

    def host_admit(*args, **kwargs):
        assert threading.get_ident() == owner
        return admit(*args, **kwargs)

    monkeypatch.setattr(engine.state.ledger, "admit_direct", host_admit)
    events = engine._execute_plan(plan, problem)
    calls = [e for e in events if e["event"] == "acquisition"]
    assert max(e["started_seconds"] for e in calls) < min(e["completed_seconds"] for e in calls)
    report = events[-1]
    assert report["event"] == "verification_schedule"
    assert report["peak_active_workers"] == report["workers"] == 2
    assert report["worker_seconds"] == pytest.approx(sum(e["worker_seconds"] for e in calls))
    assert report["worker_time_unit"] == "host-call-seconds"
    assert engine.state.ledger.complete(child, engine.state.scope)
    assert engine.state.budget.spent == 4
    assert engine.state.remaining_attempts == engine.config.max_attempts - 2
    assert engine.state.budget.pending == 0


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_requires_explicit_adapter_contract_and_single_worker_is_unchanged(workers):
    class UnsafeExecutor:
        calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            return PureIntegerExecutor().run(*args, **kwargs)

    executor = UnsafeExecutor()
    engine, child, problem, plan = parallel_fixture(executor, workers=workers)
    if workers == 2:
        with pytest.raises(ValueError, match="parallel_safe"):
            engine._execute_plan(plan, problem)
        assert executor.calls == engine.state.budget.spent == engine.state.budget.pending == 0
    else:
        events = engine._execute_plan(plan, problem)
        assert executor.calls == 2
        assert engine.state.ledger.complete(child, engine.state.scope)
        assert not any(e["event"] == "verification_schedule" for e in events)


def test_parallel_timeout_is_billed_without_admitting_missing_outcome_and_can_resume():
    class Executor(PureIntegerExecutor):
        def run(self, candidate, task, scope, cap, resume_token=None):
            if task.inputs["route"] == "write" and resume_token is None:
                return ExecutionResult("interrupted", 5, resume_token="bounded-resume", error="timeout")
            return super().run(candidate, task, scope, cap, resume_token)

    engine, child, problem, plan = parallel_fixture(Executor())
    engine._execute_plan(plan, problem)
    assert not engine.state.ledger.complete(child, engine.state.scope)
    assert engine.state.budget.spent == 6
    assert len(engine.state.ledger.records) == 3
    assert engine.state.remaining_attempts == engine.config.max_attempts - 2
    assert engine.state.budget.pending == 0
    problem = engine._catalog(engine.state)
    engine._execute_plan(engine._verify_plan(tuple(problem.by_id), problem), problem)
    assert engine.state.ledger.complete(child, engine.state.scope)
    assert engine.state.budget.spent == 9
    assert engine.state.remaining_attempts == engine.config.max_attempts - 3
    assert not engine._resume_tokens


@pytest.mark.parametrize("failure", ["crash", "overrun", "interrupt"])
def test_parallel_direct_failure_drains_and_bills_successful_sibling_before_raising(failure):
    barrier = threading.Barrier(2, timeout=5)
    completed = threading.Event()

    class Executor(PureIntegerExecutor):
        def run(self, candidate, task, scope, cap, resume_token=None):
            barrier.wait()
            if task.inputs["route"] == "read":
                if failure == "crash":
                    raise RuntimeError("unmetered evaluator crash")
                if failure == "interrupt":
                    raise KeyboardInterrupt("cancelled evaluator")
                return ExecutionResult("error", cap + 1, error="broken enforcement")
            time.sleep(.02)
            completed.set()
            return super().run(candidate, task, scope, cap, resume_token)

    engine, child, problem, plan = parallel_fixture(Executor())
    with pytest.raises({"crash": RuntimeError, "overrun": CostCapViolation, "interrupt": KeyboardInterrupt}[failure]):
        engine._execute_plan(plan, problem)
    assert completed.is_set()
    assert len(engine.state.ledger.records) == 2
    assert engine.state.budget.spent == (14 if failure == "overrun" else 13)
    assert engine.state.budget.pending == 0
    assert engine.state.remaining_attempts == engine.config.max_attempts - 2
    if failure != "overrun":
        assert not engine.state.budget.cost_valid
        assert sum(c.actual for c in engine.state.budget.charges if c.cost_kind == "actual") == 3
    else:
        assert engine.state.budget.breached
        assert all(c.cost_kind == "actual" for c in engine.state.budget.charges)


@pytest.mark.parametrize("fail_after_enqueue", [False, True])
def test_failed_pool_submission_still_joins_and_charges_already_started_work(monkeypatch, fail_after_enqueue):
    class FailingPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            if getattr(self, "submitted", False):
                if fail_after_enqueue:
                    super().submit(*args, **kwargs)
                raise RuntimeError("worker dispatch failed")
            self.submitted = True
            return super().submit(*args, **kwargs)

    monkeypatch.setattr("specrsi.core.vats.ThreadPoolExecutor", FailingPool)
    engine, child, problem, plan = parallel_fixture()
    first = problem.by_id[problem.order(plan.job_ids)[0]]
    task = next(t for t in engine.state.scope.tasks if t.uid == first.keys[0][1])
    expected = PureIntegerExecutor().run(child, task, engine.state.scope, first.cap).actual_cost
    with pytest.raises(RuntimeError, match="dispatch failed"):
        engine._execute_plan(plan, problem)
    assert engine.state.budget.spent == (4 if fail_after_enqueue else expected)
    assert engine.state.remaining_attempts == engine.config.max_attempts - (2 if fail_after_enqueue else 1)
    assert engine.state.budget.pending == 0
    assert len(engine.state.ledger.records) == 2


def test_overlapping_direct_keys_are_serialized_then_read_from_identical_cache():
    class Executor(PureIntegerExecutor):
        calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            return super().run(*args, **kwargs)

    executor = Executor()
    engine, child, problem, plan = parallel_fixture(executor)
    first = plan.jobs[0]
    duplicate = replace(first, uid="duplicate-request")
    problem = replace(problem, jobs=(first, duplicate))
    plan = engine._verify_plan(tuple(problem.by_id), problem)
    events = engine._execute_plan(plan, problem)
    calls = [e for e in events if e["event"] == "acquisition"]
    assert executor.calls == 1
    assert sorted(e["from_cache"] for e in calls) == [False, True]
    assert engine.state.remaining_attempts == engine.config.max_attempts - 1
    assert events[-1]["peak_active_workers"] == 1


class ParallelBackend:
    version = "parallel-integer-v1"
    parallel_safe = True

    def __init__(self, *, failure=None):
        self.failure = failure
        self.setup_calls = []
        self.calls = []
        self.barrier = threading.Barrier(2, timeout=5)

    def catalog(self, context, *, budget, attempts):
        child = next(c for c in context.candidates.values() if c.parent_id)
        jobs = (VerificationJob("prepare", (), 1, kind="prerequisite"), *(
            VerificationJob(task.name, ((child.uid, task.uid),), 10) for task in context.scope.tasks))
        return CompletionProblem(jobs, {"execution": 3}, budget, attempts,
                                 dependencies=tuple((task.name, "prepare") for task in context.scope.tasks))

    def setup(self, group, context, *, cap):
        self.setup_calls.append(group)
        if self.failure == "setup":
            return SetupExecutionResult("error", 1)
        return SetupExecutionResult("complete", 2, {"isolated": True},
            {"group": group, "execution_id": context.scope.execution_id, "run_id": "setup"})

    def run(self, job, context, *, setup, prerequisites, resume_token=None):
        self.calls.append(job.uid)
        assert setup["isolated"]
        if job.kind == "prerequisite":
            assert not prerequisites
            return JobExecutionResult("error" if self.failure == "prerequisite" else "complete", 1, 1,
                                      artifacts={"ready": True})
        assert prerequisites["prepare"].artifacts["ready"]
        assert len(context.records) == 2
        self.barrier.wait()
        candidate_id, task_id = job.keys[0]
        candidate = context.candidates[candidate_id]
        task = next(t for t in context.scope.tasks if t.uid == task_id)
        if self.failure == "crash" and job.uid == "read":
            raise RuntimeError("compound crash")
        actual = PureIntegerExecutor().run(candidate, task, context.scope, job.cap)
        receipt = {"job_id": job.uid, "candidate_id": candidate_id, "task_id": task_id,
                   "scope_id": context.scope.uid, "execution_id": context.scope.execution_id, "run_id": job.uid}
        if self.failure == "receipt" and job.uid == "read":
            receipt["scope_id"] = "wrong-scope"
        outcome = ExecutedJobOutcome(candidate_id, task_id, actual.outcome, receipt)
        return JobExecutionResult("complete", actual.actual_cost,
            2 if self.failure == "attempts" and job.uid == "read" else 1, (outcome,))


def test_parallel_compound_dependencies_and_shared_setup_preserve_physical_billing():
    backend = ParallelBackend()
    engine, child, problem, plan = parallel_fixture(backend=backend)
    events = engine._execute_plan(plan, problem)
    assert backend.setup_calls == ["execution"]
    assert backend.calls[0] == "prepare" and set(backend.calls[1:]) == {"read", "write"}
    acquisitions = {e["job_id"]: e for e in events if e["event"] == "compound_acquisition"}
    assert acquisitions["prepare"]["completed_seconds"] <= min(acquisitions[k]["started_seconds"] for k in ("read", "write"))
    assert max(acquisitions[k]["started_seconds"] for k in ("read", "write")) < min(acquisitions[k]["completed_seconds"] for k in ("read", "write"))
    assert engine.state.ledger.complete(child, engine.state.scope)
    assert engine.state.budget.spent == 7
    assert engine.state.remaining_attempts == engine.config.max_attempts - 3
    assert events[-1]["peak_active_workers"] == 2


@pytest.mark.parametrize("failure", ["setup", "prerequisite"])
def test_failed_parallel_prerequisites_block_descendants_without_attempt_or_fee(failure):
    backend = ParallelBackend(failure=failure)
    engine, child, problem, plan = parallel_fixture(backend=backend)
    events = engine._execute_plan(plan, problem)
    assert backend.calls == ([] if failure == "setup" else ["prepare"])
    assert engine.state.budget.spent == (1 if failure == "setup" else 3)
    assert engine.state.remaining_attempts == engine.config.max_attempts - int(failure == "prerequisite")
    assert engine.state.budget.pending == 0
    assert len(engine.state.ledger.records) == 2
    blocked = [e for e in events if e["event"] == "blocked_job"]
    assert all(e["attempts"] == e["cost"] == 0 for e in blocked)
    assert len(blocked) == (3 if failure == "setup" else 2)


@pytest.mark.parametrize("failure", ["receipt", "attempts", "crash"])
def test_parallel_compound_invalid_wave_is_charged_in_full_without_any_evidence(failure):
    backend = ParallelBackend(failure=failure)
    engine, child, problem, plan = parallel_fixture(backend=backend)
    error = {"receipt": ValueError, "attempts": CostCapViolation, "crash": RuntimeError}[failure]
    with pytest.raises(error):
        engine._execute_plan(plan, problem)
    assert set(backend.calls) == {"prepare", "read", "write"}
    assert engine.state.budget.spent == (16 if failure == "crash" else 7)
    assert engine.state.remaining_attempts == engine.config.max_attempts - (4 if failure == "attempts" else 3)
    assert engine.state.budget.pending == 0
    assert len(engine.state.ledger.records) == 2
    if failure == "attempts":
        assert engine.state.budget.breached
    if failure == "crash":
        assert not engine.state.budget.cost_valid


def test_parallel_block_reserves_full_expense_and_attempt_cap_before_dispatch():
    backend = ParallelBackend()
    engine, child, problem, plan = parallel_fixture(backend=backend)
    engine.state.remaining_attempts = plan.attempt_cap - 1
    with pytest.raises(ValueError, match="feasible"):
        engine._execute_plan(plan, problem)
    assert not backend.calls and not backend.setup_calls
    assert engine.state.budget.spent == engine.state.budget.pending == 0
