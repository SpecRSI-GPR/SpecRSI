"""Conservative equality proofs for explicitly supported immutable semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .models import Candidate, EvaluationScope, FrozenMap, Task, digest


def validate_expr(expr: tuple, depth: int = 0) -> int:
    if not isinstance(expr, tuple) or not expr or depth > 128:
        raise ValueError("malformed or excessively deep pure expression")
    if expr == ("x",):
        return 1
    if expr[0] == "const" and len(expr) == 2 and type(expr[1]) is int:
        return 1
    if expr[0] in ("add", "sub", "mul") and len(expr) == 3:
        return 1 + validate_expr(expr[1], depth + 1) + validate_expr(expr[2], depth + 1)
    raise ValueError("unsupported pure expression")


def selected_expression(candidate: Candidate, task: Task, scope: EvaluationScope) -> tuple:
    if not candidate.deterministic or candidate.semantic_kind != "pure_integer":
        raise ValueError("candidate is not supported deterministic pure integer code")
    if candidate.runtime != FrozenMap({"contract": "pure-int-v1"}):
        raise ValueError("unknown runtime effects")
    if scope.execution_id != "pure-int-v1" or scope.scorer_id != "integer-equality-v1":
        raise ValueError("unknown execution/scorer semantics")
    if set(scope.scored_fields) - {"answer", "primitives"}:
        raise ValueError("scored effects are outside the pure expression proof")
    if set(candidate.artifacts) - {"routes", "global_config"} or "routes" not in candidate.artifacts:
        raise ValueError("unmodeled global candidate effects")
    if set(task.inputs) != {"route", "x"} or type(task.inputs["x"]) is not int or type(task.expected) is not int:
        raise ValueError("unsupported task inputs or metric")
    if not scope.contains(task):
        raise ValueError("task outside evaluation scope")
    expression = candidate.artifacts["routes"][task.inputs["route"]]
    validate_expr(expression)
    return expression


@dataclass(frozen=True)
class ReuseCertificate:
    checker_id: str
    rule: str
    source_id: str
    candidate_id: str
    task_id: str
    scope_id: str
    evidence_digest: str


@dataclass(frozen=True)
class CheckResult:
    status: str  # P, U, interrupted
    actual_cost: int
    certificate: ReuseCertificate | None = None
    reason: str = ""


class ConservativeChecker:
    """Never treat sampled agreement, edit locality, hashes alone, or seeds as proof.

    Exact-identity rules are opt-in *host contracts*: each validator must establish
    that complete immutable artifacts/runtime/replay determine every scored effect.
    No exact-identity rule is enabled for arbitrary Python, LLMs, or hosted APIs.
    """
    version = "specrsi-conservative-checker-v1"

    def __init__(self, cost_per_check: int = 1,
                 exact_contracts: Mapping[str, Callable[[Candidate, Task, EvaluationScope], bool]] | None = None):
        from .budget import cost_int
        self.cost_per_check = cost_int(cost_per_check)
        self._exact_contracts = dict(exact_contracts or {})

    def _proof(self, source: Candidate, child: Candidate, task: Task, scope: EvaluationScope) -> ReuseCertificate | None:
        if not (source.deterministic and child.deterministic and scope.contains(task)):
            return None
        if source.runtime != child.runtime:
            return None
        rule = None
        evidence = None
        try:
            a, b = selected_expression(source, task, scope), selected_expression(child, task, scope)
            if a == b and source.artifacts.get("global_config") == child.artifacts.get("global_config"):
                rule, evidence = "selected-pure-integer-expression-v1", (a, source.runtime, scope.uid, task.uid)
        except (KeyError, ValueError, TypeError, RecursionError):
            pass
        validator = self._exact_contracts.get(scope.execution_id)
        if rule is None and validator is not None:
            if (source.artifacts == child.artifacts and source.semantic_kind == child.semantic_kind
                    and validator(source, task, scope) and validator(child, task, scope)):
                rule = "registered-deterministic-full-identity-v1"
                evidence = (source.artifacts, source.runtime, scope.uid, task.uid, scope.execution_id)
        if rule is None:
            return None
        return ReuseCertificate(self.version, rule, source.uid, child.uid, task.uid, scope.uid, digest(evidence))

    def check(self, source: Candidate, child: Candidate, task: Task, scope: EvaluationScope, cap: int) -> CheckResult:
        from .budget import cost_int
        cost_int(cap)
        if cap < self.cost_per_check:
            return CheckResult("interrupted", cap, reason="checker cap exhausted; no P/U label")
        certificate = self._proof(source, child, task, scope)
        return CheckResult("P" if certificate else "U", self.cost_per_check, certificate,
                           "checked equality" if certificate else "unsupported or unequal complete effects")

    def verify(self, certificate: ReuseCertificate, source: Candidate, child: Candidate,
               task: Task, scope: EvaluationScope) -> bool:
        return certificate == self._proof(source, child, task, scope)
