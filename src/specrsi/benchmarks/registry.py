from __future__ import annotations

from .base import BenchmarkExample, MissingArtifactError, ScoreResult
from .external import mbpp_bubblewrap_score, swebench_score
from .livebench import livebench_math_score
from .planning import planbench_example_score
from .scorers import aime_score, game24_score, hotpotqa_score, hover_retrieval_score, ifbench_score, pupa_score


def score_example(example: BenchmarkExample, prediction, **options) -> ScoreResult:
    """Dispatch by the registered dataset; no fallback approximate scorer."""
    dataset = example.dataset
    if dataset == "game24": return game24_score(prediction, example.reference)
    if dataset == "aime25": return aime_score(prediction, example.reference)
    if dataset == "hotpotqa": return hotpotqa_score(prediction, example.reference, **options)
    if dataset == "ifbench": return ifbench_score(prediction, example.reference, **options)
    if dataset == "hover": return hover_retrieval_score(prediction["retrieved_docs"], example.reference)
    if dataset == "pupa": return pupa_score(prediction, example.reference, **options)
    if dataset == "livebench_math": return livebench_math_score(prediction, example.reference, **options)
    if dataset == "mbpp": return mbpp_bubblewrap_score(prediction, example.reference, **options)
    if dataset == "planbench": return planbench_example_score(example, prediction, **options)
    if dataset in {"swe_verified", "swe_lite"}: return swebench_score(prediction, example.id)
    harness = options.get("harness")
    if harness is None:
        raise MissingArtifactError(f"{dataset} requires its registered native harness and scenario artifacts")
    return harness.score(example, prediction)
