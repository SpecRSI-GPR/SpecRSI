"""Trusted executor protocol; opaque task backends always require real execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .budget import cost_int
from .checker import selected_expression
from .models import Candidate, EvaluationScope, FrozenMap, Outcome, Task, digest


@dataclass(frozen=True)
class ExecutionResult:
    status: str  # complete, interrupted, error; scored timeout is complete
    actual_cost: int
    outcome: Outcome | None = None
    receipt: FrozenMap = field(default_factory=FrozenMap)
    resume_token: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        cost_int(self.actual_cost)
        object.__setattr__(self, "receipt", FrozenMap(self.receipt))
        if self.status not in ("complete", "interrupted", "error"):
            raise ValueError("unknown execution status")
        if (self.status == "complete") != (self.outcome is not None):
            raise ValueError("only complete executions supply scored outcomes")


class Executor(Protocol):
    # Optional at runtime: required only for concurrent VERIFY. True promises
    # thread-safe calls and isolation of relevant candidate/environment state.
    parallel_safe: bool

    def run(self, candidate: Candidate, task: Task, scope: EvaluationScope, cap: int,
            resume_token: str | None = None) -> ExecutionResult: ...


class PureIntegerExecutor:
    """Primitive-count reference backend, not a wall-clock or GPU-cost model."""
    parallel_safe = True

    def run(self, candidate: Candidate, task: Task, scope: EvaluationScope, cap: int,
            resume_token: str | None = None) -> ExecutionResult:
        cost_int(cap)
        expr = selected_expression(candidate, task, scope)
        used = 0

        def evaluate(node: tuple) -> int:
            nonlocal used
            if used == cap:
                raise TimeoutError("primitive-count cap exhausted")
            used += 1
            if node[0] == "x":
                return task.inputs["x"]
            if node[0] == "const":
                return node[1]
            a, b = evaluate(node[1]), evaluate(node[2])
            if node[0] == "add":
                return a + b
            if node[0] == "sub":
                return a - b
            return a * b

        try:
            answer = evaluate(expr)
        except TimeoutError:
            return ExecutionResult("interrupted", used, resume_token=None, error="cap exhausted; restart possible")
        outcome = Outcome(float(answer == task.expected), {"answer": answer, "primitives": used})
        return ExecutionResult("complete", used, outcome, FrozenMap({"executor": "pure-int-v1", "expression": digest(expr)}))
