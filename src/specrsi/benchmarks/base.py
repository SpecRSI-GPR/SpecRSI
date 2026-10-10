from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping


class MissingArtifactError(RuntimeError):
    """A native task cannot run without its explicitly identified artifact."""


@dataclasses.dataclass(frozen=True)
class BenchmarkExample:
    id: str
    dataset: str
    split: str
    prompt: str
    reference: Any
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        # Use input identity, independent of answers, IDs, and assigned splits.
        text = " ".join(self.prompt.split()).casefold()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> "BenchmarkExample":
        return cls(**{f.name: value[f.name] for f in dataclasses.fields(cls) if f.name in value})


@dataclasses.dataclass(frozen=True)
class ScoreResult:
    score: float
    metrics: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.score):
            raise ValueError("Non-finite native score")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
