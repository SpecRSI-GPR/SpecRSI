"""Versioned public benchmark data and task-native scoring adapters."""

from .base import BenchmarkExample, MissingArtifactError, ScoreResult
from .data import load_examples, validate_disjoint_splits, verify_manifest
from .scorers import aime_score, game24_score, hotpotqa_score, hover_score, ifbench_score, pupa_score

__all__ = [
    "BenchmarkExample", "MissingArtifactError", "ScoreResult", "load_examples",
    "validate_disjoint_splits", "verify_manifest", "aime_score", "game24_score",
    "hotpotqa_score", "hover_score", "ifbench_score", "pupa_score",
]
