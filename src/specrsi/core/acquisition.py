"""Bounded acquisition of contracted outcomes with explicit expense accounting.

Eq. (2): f_real(c)=(J(c),v(c)), and
    c_star_n in argmax_{(c,J,v) in H_n : v=1} J.
This interface obtains one required item at a time; complete evidence across I
is aggregated by Ledger to recover J_Gamma(c) and the validity predicate. One
successful acquisition is therefore not a complete-candidate evaluation unless
it supplies the final missing requirement.

The completeness mask is m_ci=1 iff a valid outcome for candidate c, task i,
and the fixed scope Gamma exists. Acquire_Gamma either reads that identical
record, obtains a checked equality to a valid source, or executes the real
check. Direct execution supplies route U; checked equality supplies route P.
The direct VATS configuration sets max_sources=0, removing cross-version reuse.
An identity includes the immutable candidate/task manifests and the scope's
executor, scorer, required fields, replay semantics and environment state.

For newly acquired records Delta_E, verification makes the state transition
    T^+=T,
    E^+=E union {e in Delta_E : Valid_Gamma(e)=1}.
Valid_Gamma means identity, scope, fields and provenance are valid; it does not
mean the task passed. A complete hard failure can set m_ci=1 while leaving the
candidate inadmissible. A timeout with missing fields leaves m_ci=0.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Sequence
import uuid

from .budget import BudgetLedger, CostCapViolation, cost_int
from .checker import CheckResult
from .execution import Executor
from .ledger import Ledger, Record
from .models import Candidate, EvaluationScope, Task


@dataclass(frozen=True)
class AcquisitionCaps:
    execution: int
    lookup: int = 0
    check: int = 1
    max_sources: int = 1

    def __post_init__(self) -> None:
        for x in (self.execution, self.lookup, self.check, self.max_sources):
            cost_int(x)

    def total(self, number_of_sources: int = 0) -> int:
        # Enforceable acquisition cap, in the same integer expense units as B:
        # c_bar = c_lookup + c_execution +
        #         min(number_of_sources, max_sources)*c_check.
        # Every paid equality check may fail, so reserve the entire direct
        # execution fallback as well. Expected savings cannot fund this cap.
        return self.lookup + self.execution + self.check * min(number_of_sources, self.max_sources)


@dataclass(frozen=True)
class AcquisitionResult:
    record: Record | None
    actual_cost: int
    status: str
    checks: tuple[CheckResult, ...] = ()
    from_cache: bool = False
    resume_token: str | None = None
    error: str | None = None


class ResultAcquirer:
    def __init__(self, ledger: Ledger, budget: BudgetLedger, executor: Executor):
        self.ledger, self.budget, self.executor = ledger, budget, executor
        self._locks: dict[tuple, Lock] = {}
        self._lock_guard = Lock()

    def acquire(self, candidate: Candidate, task: Task, scope: EvaluationScope,
                sources: Sequence[Candidate], caps: AcquisitionCaps,
                *, reservation: str | None = None, resume_token: str | None = None) -> AcquisitionResult:
        key = candidate.uid, task.uid, scope.uid
        with self._lock_guard:
            lock = self._locks.setdefault(key, Lock())
        with lock:
            return self._acquire(candidate, task, scope, sources, caps, reservation, resume_token)

    def _acquire(self, candidate: Candidate, task: Task, scope: EvaluationScope,
                 sources: Sequence[Candidate], caps: AcquisitionCaps, reservation: str | None,
                 resume_token: str | None) -> AcquisitionResult:
        if not scope.contains(task):
            raise ValueError("acquisition task outside fixed evaluation scope")
        # Source order is part of the bounded checker protocol and is logged.
        seen: set[str] = set()
        unique_sources = []
        for source in sources:
            if source.uid != candidate.uid and source.uid not in seen:
                seen.add(source.uid)
                unique_sources.append(source)
        sources = tuple(unique_sources[:caps.max_sources])
        existing = self.ledger.get(candidate, task, scope)
        required_cap = caps.lookup if existing is not None else caps.total(len(sources))
        job = reservation or f"acquire:{uuid.uuid4().hex}"
        if reservation is None:
            self.budget.reserve(job, required_cap, "acquisition")
        elif job not in self.budget.reserved or self.budget.reserved[job] < required_cap:
            raise ValueError("provided reservation does not cover full checker-plus-fallback cap")
        actual = caps.lookup
        checks: list[CheckResult] = []
        if existing is not None:
            # An identical admitted outcome already establishes m_ci=1. Reading
            # it spends the contracted lookup expense but launches no new task
            # execution; the caller's primitive-attempt accounting uses from_cache.
            self.budget.settle(job, actual)
            return AcquisitionResult(existing, actual, "complete", from_cache=True)
        try:
            for source in sources:
                # Source validity is necessary before a proof can import an outcome.
                if self.ledger.get(source, task, scope) is None:
                    continue
                check = self.ledger.checker.check(source, candidate, task, scope, caps.check)
                cost_int(check.actual_cost)
                actual += check.actual_cost
                checks.append(check)
                if check.actual_cost > caps.check:
                    raise CostCapViolation("checker exceeded its enforced cap")
                if check.status == "P":
                    if check.certificate is None:
                        raise ValueError("P result lacks a certificate")
                    record = self.ledger.inherit(source, candidate, task, scope, check.certificate)
                    self.budget.settle(job, actual)
                    return AcquisitionResult(record, actual, "complete", tuple(checks))
                if check.status not in ("U", "interrupted"):
                    raise ValueError("unknown checker result")
            result = self.executor.run(candidate, task, scope, caps.execution, resume_token=resume_token)
            actual += result.actual_cost
            if result.actual_cost > caps.execution:
                raise CostCapViolation("executor exceeded its enforced cap")
            record = None
            if result.status == "complete":
                # Eq. (3) requires valid COMPLETE fields before m_ci can become 1.
                # ``admit_direct`` validates the outcome even when score=0 or the
                # hard-constraint flag is false; those are observed failures,
                # distinct from interrupted/partial outcomes with no record.
                record = self.ledger.admit_direct(candidate, task, scope, result.outcome, result.receipt)
            self.budget.settle(job, actual)
            return AcquisitionResult(record, actual, result.status, tuple(checks), resume_token=result.resume_token, error=result.error)
        except CostCapViolation:
            # Account measured expenditure, even when it reveals a broken backend.
            if job in self.budget.reserved:
                try:
                    self.budget.settle(job, actual)
                finally:
                    self.budget.breached = True
            raise
        except Exception:
            # An unmetered exception cannot be treated as free. Conservatively
            # charge the admitted cap and propagate so the caller can diagnose it.
            if job in self.budget.reserved:
                self.budget.settle_unknown(job)
            raise
