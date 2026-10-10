from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


class InvalidCandidate(ValueError):
    """A candidate violates the frozen API or hard scenario constraints."""


def exact_keys(candidate: Mapping, keys: Sequence[str]) -> None:
    if not isinstance(candidate, Mapping) or set(candidate) != set(keys):
        raise InvalidCandidate(f"Candidate must have exactly these fields: {list(keys)}")


def scalar(value, lower: float, upper: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCandidate(f"{name} must be a finite number")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise InvalidCandidate(f"{name} outside [{lower}, {upper}]")
    return float(value)


def vector(value, count: int, lower: float, upper: float, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise InvalidCandidate(f"{name} must contain exactly {count} values")
    return [scalar(x, lower, upper, name) for x in value]
