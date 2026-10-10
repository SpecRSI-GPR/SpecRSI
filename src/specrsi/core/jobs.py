"""Trusted compound verification backend contracts for PDF Appendix F.4.

A job requests possible coverage. Only individual, complete outcomes accompanied
by execution receipts can enter the host ledger. Setup and prerequisite outputs
are operational artifacts, not task evidence. Backends must enforce the expense
and attempt caps they receive and isolate candidate execution environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from .budget import cost_int
from .ledger import Record
from .models import Candidate, EvaluationScope, FrozenMap, Outcome
from specrsi.planning.completion import CompletionProblem, VerificationJob


@dataclass(frozen=True)
class JobContext:
    scope: EvaluationScope
    candidates: Mapping[str, Candidate]
    records: Mapping[tuple[str, str, str], Record]
    history: tuple[FrozenMap, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))
        object.__setattr__(self, "history", tuple(FrozenMap(row) for row in self.history))


@dataclass(frozen=True)
class ExecutedJobOutcome:
    candidate_id: str
    task_id: str
    outcome: Outcome
    receipt: FrozenMap

    def __post_init__(self):
        object.__setattr__(self, "receipt", FrozenMap(self.receipt))
        if not self.candidate_id or not self.task_id or not isinstance(self.outcome, Outcome):
            raise ValueError("job outcome requires an immutable candidate, task and complete outcome")

    def validate(self, job: VerificationJob, context: JobContext) -> None:
        if (self.candidate_id, self.task_id) not in job.keys:
            raise ValueError("job outcome falls outside its requested coverage")
        expected = {"job_id": job.uid, "candidate_id": self.candidate_id, "task_id": self.task_id,
                    "scope_id": context.scope.uid, "execution_id": context.scope.execution_id}
        if any(self.receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("compound execution receipt does not bind the job, candidate, task and scope")
        if not isinstance(self.receipt.get("run_id"), str) or not self.receipt["run_id"]:
            raise ValueError("compound execution receipt requires a physical run identity")
        self.outcome.validate(context.scope)


@dataclass(frozen=True)
class JobExecutionResult:
    status: str
    actual_cost: int
    actual_attempts: int
    outcomes: tuple[ExecutedJobOutcome, ...] = ()
    artifacts: FrozenMap = field(default_factory=FrozenMap)
    resume_token: str | None = None
    error: str | None = None

    def __post_init__(self):
        cost_int(self.actual_cost)
        cost_int(self.actual_attempts)
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "artifacts", FrozenMap(self.artifacts))
        if self.status not in {"complete", "interrupted", "error"}:
            raise ValueError("invalid compound job status")
        if any(not isinstance(outcome, ExecutedJobOutcome) for outcome in self.outcomes):
            raise TypeError("compound outcomes must have explicit execution receipts")


@dataclass(frozen=True)
class SetupExecutionResult:
    status: str
    actual_cost: int
    artifacts: FrozenMap = field(default_factory=FrozenMap)
    receipt: FrozenMap = field(default_factory=FrozenMap)
    error: str | None = None

    def __post_init__(self):
        cost_int(self.actual_cost)
        object.__setattr__(self, "artifacts", FrozenMap(self.artifacts))
        object.__setattr__(self, "receipt", FrozenMap(self.receipt))
        if self.status not in {"complete", "interrupted", "error"}:
            raise ValueError("invalid shared setup status")


class VerificationBackend(Protocol):
    """A fixed evaluator extension, never a drafter-editable verification rule.

    ``catalog`` only reads existing evidence. ``setup`` is called once per actual
    setup group in a selected block. ``run`` receives completed prerequisite
    artifacts and exact caps in ``job``; failed prerequisites block descendants.
    A complete operation may have less realized coverage than its planned keys.
    An interrupted operation may still return some complete, receipted outcomes.
    ``parallel_safe=True`` explicitly permits concurrent ``run`` calls, including
    immutable shared-setup artifacts. Relevant mutable environment state must be
    isolated by the backend. Without this declaration only one worker is legal.
    """
    version: str
    parallel_safe: bool

    def catalog(self, context: JobContext, *, budget: int, attempts: int) -> CompletionProblem: ...

    def setup(self, group: str, context: JobContext, *, cap: int) -> SetupExecutionResult: ...

    def run(self, job: VerificationJob, context: JobContext, *, setup: FrozenMap,
            prerequisites: Mapping[str, JobExecutionResult], resume_token: str | None = None) -> JobExecutionResult: ...
