"""Immutable evaluation identities and complete scored outcomes.

Evaluation and acceptance contracts are host-owned. A model prediction is deliberately
not an Outcome and cannot create evidence coverage.
"""
from __future__ import annotations

from collections.abc import Mapping, Iterator
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Any


def freeze(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(x) for x in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"manifest values must be finite JSON data, got {type(value).__name__}")


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(x) for x in value]
    return value


def digest(value: Any) -> str:
    return sha256(json.dumps(thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, init=False)
class FrozenMap(Mapping):
    _items: tuple[tuple[str, Any], ...]

    def __init__(self, value: Mapping | None = None):
        value = {} if value is None else value
        if any(type(k) is not str for k in value):
            raise TypeError("manifest keys must be strings")
        object.__setattr__(self, "_items", tuple(sorted((k, freeze(v)) for k, v in value.items())))

    def __getitem__(self, key: str) -> Any:
        for k, value in self._items:
            if k == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (k for k, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class Candidate:
    name: str
    artifacts: FrozenMap = field(default_factory=FrozenMap)
    runtime: FrozenMap = field(default_factory=FrozenMap)
    semantic_kind: str = "opaque"
    deterministic: bool = False
    operating_penalty: float = 0.0
    parent_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", FrozenMap(self.artifacts))
        object.__setattr__(self, "runtime", FrozenMap(self.runtime))
        if not self.name or not math.isfinite(self.operating_penalty) or self.operating_penalty < 0:
            raise ValueError("invalid candidate name or operating penalty")
        if type(self.deterministic) is not bool:
            raise TypeError("deterministic must be a boolean")

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "artifacts": thaw(self.artifacts), "runtime": thaw(self.runtime),
                "semantic_kind": self.semantic_kind, "deterministic": self.deterministic,
                "operating_penalty": self.operating_penalty, "parent_id": self.parent_id}

    @property
    def uid(self) -> str:
        return digest(self.manifest())


@dataclass(frozen=True)
class Task:
    name: str
    inputs: FrozenMap = field(default_factory=FrozenMap)
    expected: Any = None
    metadata: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", FrozenMap(self.inputs))
        object.__setattr__(self, "metadata", FrozenMap(self.metadata))
        object.__setattr__(self, "expected", freeze(self.expected))
        if not self.name:
            raise ValueError("task name must not be empty")

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "inputs": thaw(self.inputs), "expected": thaw(self.expected), "metadata": thaw(self.metadata)}

    @property
    def uid(self) -> str:
        return digest(self.manifest())


@dataclass(frozen=True)
class EvaluationScope:
    """Complete selection registry, scorer, replay state and scored effects.

    A native aggregator is supplied to Ledger by this scope's immutable aggregator_id.
    Audit scopes must be separate and are never accepted as selection evidence.
    """
    name: str
    tasks: tuple[Task, ...]
    execution_id: str
    scorer_id: str
    scored_fields: tuple[str, ...] = ("answer",)
    aggregation_id: str = "weighted_mean_v1"
    weights: tuple[float, ...] = ()
    replay: FrozenMap = field(default_factory=FrozenMap)
    constraints_id: str = "all_outcomes_admissible_v1"
    split: str = "selection"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "scored_fields", tuple(self.scored_fields))
        object.__setattr__(self, "replay", FrozenMap(self.replay))
        if not self.tasks or len({t.uid for t in self.tasks}) != len(self.tasks):
            raise ValueError("required task registry must be nonempty and duplicate-free")
        if not self.name or not self.execution_id or not self.scorer_id or not self.aggregation_id:
            raise ValueError("scope requires explicit versioned contract identities")
        if not self.scored_fields or len(set(self.scored_fields)) != len(self.scored_fields):
            raise ValueError("scored_fields must be nonempty and unique")
        weights = tuple(self.weights) or (1.0 / len(self.tasks),) * len(self.tasks)
        if len(weights) != len(self.tasks) or any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-12):
            raise ValueError("weights must be nonnegative and sum to one")
        object.__setattr__(self, "weights", weights)
        if self.split not in ("selection", "audit", "train", "dev"):
            raise ValueError("invalid evaluation split")

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "tasks": [t.manifest() for t in self.tasks], "execution_id": self.execution_id,
                "scorer_id": self.scorer_id, "scored_fields": list(self.scored_fields), "aggregation_id": self.aggregation_id,
                "weights": list(self.weights), "replay": thaw(self.replay), "constraints_id": self.constraints_id, "split": self.split}

    @property
    def uid(self) -> str:
        return digest(self.manifest())

    def contains(self, task: Task) -> bool:
        return task.uid in {t.uid for t in self.tasks}


@dataclass(frozen=True)
class Outcome:
    score: float
    fields: FrozenMap
    admissible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", FrozenMap(self.fields))
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("per-item score must be finite and in [0, 1]")
        if type(self.admissible) is not bool:
            raise TypeError("admissibility must be boolean")

    def manifest(self) -> dict[str, Any]:
        return {"score": self.score, "fields": thaw(self.fields), "admissible": self.admissible}

    def validate(self, scope: EvaluationScope) -> None:
        if not set(scope.scored_fields).issubset(self.fields):
            raise ValueError("outcome omits required scored effects")


RecordKey = tuple[str, str, str]  # candidate, task, complete evaluation scope
