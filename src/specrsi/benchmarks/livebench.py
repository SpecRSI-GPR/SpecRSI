"""Call official, task-specific LiveBench math scoring functions."""
from __future__ import annotations

import importlib
from pathlib import Path
import re
import sys
from types import FunctionType, SimpleNamespace
from typing import Callable, Mapping

from .base import MissingArtifactError, ScoreResult
from .data import verify_manifest


LOCAL_FALLBACK_METRIC = "official_math_with_prospective_local_AMPS_fallback"


def livebench_math_score(prediction: str, reference: Mapping, *, root: str | Path = ".", verify: bool = True,
                        equivalence_judge: Callable[[str, str], bool] | None = None,
                        judge_identity: str | None = None) -> ScoreResult:
    """Preserve native extraction/symbolic scoring; optionally supply a local judge.

    The callback replaces only the official AMPS LLM fallback when requested by
    the pinned harness. It is a separately named prospective metric. Neither
    process credentials nor the imported official module are ever modified.
    """
    if equivalence_judge is not None and (not callable(equivalence_judge)
            or not isinstance(judge_identity, str) or not judge_identity.strip()):
        raise ValueError("A prospective local judge requires a callable and frozen identity")
    root = Path(root).resolve()
    if verify:
        verify_manifest(root / "data/manifests/livebench_harness.json", root=root)
    sources = list((root / "data/raw/livebench_harness/source").glob("LiveBench-*"))
    if len(sources) != 1:
        raise MissingArtifactError("Prepare the pinned official LiveBench harness")
    if str(sources[0]) not in sys.path:
        sys.path.insert(0, str(sources[0]))
    task = str(reference.get("subtask") or reference["task"])
    answer = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
    truth = str(reference["ground_truth"])
    fallback_calls = 0
    try:
        if task.startswith("aime"):
            module = importlib.import_module("livebench.process_results.math.math_competitions.utils")
            score = module.aime_process_results(truth, answer)
        elif task.startswith(("amc", "smc")) or "_amc" in task:
            module = importlib.import_module("livebench.process_results.math.math_competitions.utils")
            score = module.mathcontest_process_results(truth, answer, reference["turns"][0])
        elif task.startswith(("imo", "usamo")):
            module = importlib.import_module("livebench.process_results.math.olympiad.utils")
            score = module.proof_rearrangement_process_results(truth, answer, edit_distance=True)
        elif "amps_hard" in task:
            module = importlib.import_module("livebench.process_results.math.AMPS_Hard.utils")
            def registered_judge(expected, parsed_answer):
                nonlocal fallback_calls
                if equivalence_judge is None:
                    raise MissingArtifactError("Official LiveBench AMPS fallback requires an explicitly registered judge; choose the named prospective local fallback metric or register the official gpt-5-mini judge")
                fallback_calls += 1
                decision = equivalence_judge(expected, parsed_answer)
                if type(decision) is not bool:
                    raise ValueError("AMPS equivalence judge must return a boolean")
                return decision

            # The official function gates its LLM fallback on key presence. A
            # private function-global view satisfies only that branch guard and
            # replaces its callback; no hosted client can be reached. Its other
            # extraction, normalization and symbolic functions remain unchanged.
            original = module.amps_hard_process_results
            private_globals = dict(original.__globals__)
            private_globals.update(is_equiv_llm=registered_judge,
                os=SimpleNamespace(environ={"OPENAI_API_KEY": "local-callback-no-hosted-key"}))
            local_function = FunctionType(original.__code__, private_globals, original.__name__,
                                          original.__defaults__, original.__closure__)
            score = local_function(truth, answer)
        else:
            raise MissingArtifactError(f"Native scoring for LiveBench subtask {task} is not configured")
    except ImportError as exc:
        raise MissingArtifactError(f"Missing official LiveBench math dependency: {exc}") from exc
    # Enforce that import resolution actually selected the pinned, hashed source.
    if not Path(module.__file__).resolve().is_relative_to(sources[0].resolve()):
        raise ValueError("A different LiveBench installation shadowed the pinned harness")
    return ScoreResult(float(score), {"native_score": float(score), "task": task, "evaluator": "official_livebench",
        "automatic_remote_judge": False,
        "metric_variant": LOCAL_FALLBACK_METRIC if equivalence_judge is not None else "official_native_math",
        "local_fallback_calls": fallback_calls, "judge_identity": judge_identity if equivalence_judge is not None else None})
