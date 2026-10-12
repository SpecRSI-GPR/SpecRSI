from dataclasses import FrozenInstanceError, replace

import pytest

from specrsi.benchmarks.base import BenchmarkExample
from specrsi.core.models import thaw
from specrsi.performance_candidates import (
    build_performance_candidates, example_identity_keys, recipe_definitions,
)


def _runtime():
    return {"model_revision": "test-model", "seed": 123,
            "generation": {"max_tokens": 32, "enable_thinking": False, "temperature": 0.0}}


def _example(index, *, answer=None, question=None, metadata=None):
    question = question or f"What is the birthplace of Person {index}?"
    answer = answer or f"Place {index}"
    return BenchmarkExample(
        f"dev-{index}", "hotpotqa", "development",
        f"Person {index}:\nPerson {index} was born in Place {index}.\n\n"
        f"Distractor {index}:\nThis unrelated paragraph is not useful.\n\n"
        f"Place {index}:\nPlace {index} is located in Region {index}.\n\n"
        f"Question: {question}\nAnswer:",
        {"answer": answer, "supporting_facts": [[f"Person {index}", 0], [f"Place {index}", 0]]},
        {"question": question, **(metadata or {})},
    )


def _pool():
    return [_example(1), _example(2), _example(3, answer="yes", question="Do Alpha and Beta share a location?"),
            _example(4, question="Which person was born earlier, Gamma or Delta?"), _example(5)]


def test_seeded_candidate_pool_is_order_independent_and_immutable():
    runtime = _runtime()
    first = build_performance_candidates(_pool(), runtime=runtime, seed=7)
    second = build_performance_candidates(reversed(_pool()), runtime=_runtime(), seed=7)
    assert [item.manifest() for item in first] == [item.manifest() for item in second]
    assert len(first) == 6
    assert len({item.candidate.uid for item in first}) == 6
    runtime["generation"]["max_tokens"] = 99
    assert first[0].candidate.runtime["generation"]["max_tokens"] == 32
    with pytest.raises((FrozenInstanceError, AttributeError)):
        first[0].recipe_id = "changed"
    with pytest.raises(TypeError):
        first[0].provenance["seed"] = 99
    exported = first[-1].manifest()
    exported["provenance"]["demonstrations"].clear()
    assert len(first[-1].provenance["demonstrations"]) == 4


def test_demonstrations_preserve_support_and_record_explicit_supervision():
    result = build_performance_candidates(_pool(), runtime=_runtime(), recipe_ids=["balanced_4shot"])[0]
    prompt = result.candidate.artifacts["system_prompt"]
    demos = result.provenance["demonstrations"]
    assert result.provenance["supervision"] == "explicit_supervised_development"
    assert {item["kind"] for item in demos} == {"bridge", "comparison", "yes_no"}
    assert all(item["source_mode"] == "supervised_development" and item["split"] == "development" for item in demos)
    assert all(item["reference_sha256"] and item["demonstration_sha256"] for item in demos)
    assert "Distractor" not in prompt
    for demo in demos:
        index = int(demo["example_id"].split("-")[-1])
        assert f"Person {index} was born in Place {index}." in prompt
        assert f"Place {index} is located in Region {index}." in prompt


@pytest.mark.parametrize("match", ["id", "input", "question", "lineage", "content_id"])
def test_protected_overlap_is_removed_without_reading_protected_labels(match):
    class ForbiddenReference:
        def __str__(self):
            raise AssertionError("Protected labels were read")

    source = _example(99, metadata={"lineage_id": "root-99", "content_id": "content-99"})
    protected = replace(_example(100), split="audit", reference=ForbiddenReference())
    if match == "id":
        protected = replace(protected, id=source.id)
    elif match == "input":
        protected = replace(protected, prompt=source.prompt.upper())
    elif match == "question":
        protected = replace(protected, metadata={"question": source.metadata["question"]})
    elif match == "lineage":
        protected = replace(protected, metadata={"root_lineage_id": "root-99"})
    else:
        protected = replace(protected, metadata={"content_id": "content-99"})
    result = build_performance_candidates(_pool() + [source], runtime=_runtime(),
                                          protected_examples=[protected], recipe_ids=["balanced_4shot"])[0]
    assert source.id not in {demo["example_id"] for demo in result.provenance["demonstrations"]}
    assert {"id": source.id, "reason": "protected_overlap"} in thaw(result.provenance["excluded"])


def test_development_labels_cannot_be_taken_from_other_splits():
    with pytest.raises(ValueError, match="development"):
        build_performance_candidates([replace(_example(1), split="selection")], runtime=_runtime())


def test_duplicate_descendants_cannot_fill_few_shot_pool():
    one = _example(1)
    duplicate = replace(one, id="another-id")
    assert example_identity_keys(one) & example_identity_keys(duplicate)
    result = build_performance_candidates([one, duplicate], runtime=_runtime(), recipe_ids=["evidence_2shot"])
    assert result == ()


def test_oversize_few_shot_is_omitted_without_truncation_or_output_cap_changes():
    seen = []

    def token_counter(messages, *, enable_thinking):
        seen.append((messages, enable_thinking))
        return sum(len(message["content"]) for message in messages)

    zero_ids = [recipe["id"] for recipe in recipe_definitions() if not recipe["demonstration_count"]]
    cap = max(len(recipe["instruction"]) for recipe in recipe_definitions()) + 64
    result = build_performance_candidates(_pool(), runtime=_runtime(), token_counter=token_counter,
                                          execution_cap=cap, task_prompts=["short", "somewhat longer"])
    assert [item.recipe_id for item in result] == zero_ids
    assert len(seen) == 12  # Every candidate, including omitted ones, checks both task inputs.
    for item in result:
        assert item.candidate.runtime["generation"]["max_tokens"] == 32
        assert item.provenance["cap_validation"]["max_required_tokens"] <= cap
        assert item.provenance["cap_validation"]["task_count"] == 2
    with pytest.raises(ValueError, match="zero-shot"):
        build_performance_candidates([], runtime=_runtime(), token_counter=token_counter,
                                     execution_cap=10, task_prompts=["input"], recipe_ids=["direct"])


def test_token_cap_guarantee_requires_counter_and_inputs():
    with pytest.raises(ValueError, match="all task prompts"):
        build_performance_candidates([], runtime=_runtime(), execution_cap=4096)


@pytest.mark.parametrize("dataset", ["aime25", "math", "livebench_math"])
def test_math_bank_has_distinct_reasoning_methods_and_inherits_exact_runtime(dataset):
    runtime = _runtime()
    runtime["generation"].update(enable_thinking=True, max_tokens=2048)
    source = BenchmarkExample("math-dev", dataset, "development", "Solve x + 1 = 2.", object())
    result = build_performance_candidates([source], runtime=runtime)
    assert [item.recipe_id for item in result] == ["direct", "careful_derivation", "verify", "independent_alternative"]
    assert len({item.candidate.artifacts["system_prompt"] for item in result}) == 4
    assert all(thaw(item.candidate.runtime) == runtime for item in result)
    assert all(item.provenance["supervision"] == "none" for item in result)
    assert all(not item.provenance["demonstrations"] for item in result)
    assert all(item.provenance["dataset"] == dataset for item in result)
    if dataset.startswith("aime"):
        assert "0 to 999" in result[0].candidate.artifacts["system_prompt"]
    else:
        assert r"\boxed" in result[0].candidate.artifacts["system_prompt"]


def test_no_data_math_bank_and_definition_exports_are_detached():
    result = build_performance_candidates([], runtime=_runtime(), dataset="aime25")
    assert len(result) == 4
    definitions = recipe_definitions("aime25")
    definitions[0]["instruction"] = "Changed"
    assert recipe_definitions("aime25")[0]["instruction"] != "Changed"
    with pytest.raises(ValueError, match="one dataset"):
        build_performance_candidates([_example(1), BenchmarkExample("a", "aime25", "development", "p", "1")],
                                     runtime=_runtime())
    with pytest.raises(ValueError, match="registered"):
        build_performance_candidates([], runtime=_runtime(), dataset="aime25", recipe_ids=["evidence_2shot"])
