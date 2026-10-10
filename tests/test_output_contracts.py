"""Task input changes stay explicit, shared, and fully metered."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from specrsi.core.models import Candidate, EvaluationScope, Task, thaw
from specrsi.experiments import QwenExecutor
from specrsi.models import ModelReply
from specrsi.output_contracts import HOTPOT_SHORT_ANSWER_V1, generation_messages


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages).encode())


def fixture(contract=True, dataset="hotpotqa"):
    generation = {"max_tokens": 8, "temperature": 0., "enable_thinking": False}
    if contract:
        generation["output_contract"] = "hotpot_short_answer_v1"
    candidate = Candidate("root", {"system_prompt": "Mention all relevant entities."},
        {"generation": generation, "model_revision": "frozen", "seed": 3})
    task = Task("q", {"prompt": "PUBLIC_QUESTION"}, "Paris",
                {"dataset": dataset, "private": "PRIVATE_METADATA"})
    scope = EvaluationScope("test", (task,), "execution-v1", "score-v1")
    tokenizer, calls = Tokenizer(), []

    def generate(messages, **kwargs):
        calls.append((messages, kwargs))
        return ModelReply("Paris", len(tokenizer.apply_chat_template(messages)), 2,
                          .1, "stop", "frozen", "measured-request")

    executor = QwenExecutor(SimpleNamespace(generate=generate), tokenizer, generation)
    return executor, candidate, task, scope, calls


def test_contract_is_an_immutable_input_and_does_not_postprocess_native_answer():
    executor, candidate, task, scope, calls = fixture()
    messages = executor.generation_messages(candidate, task)
    count = executor.prompt_tokens(messages)
    executor.validate_prompt_candidate(candidate, [task], count + 8)
    result = executor.run(candidate, task, scope, count + 8)
    assert result.status == "complete" and result.outcome.score == 1.
    assert result.outcome.fields["answer"] == "Paris"
    assert result.actual_cost == count + 2
    assert result.receipt["prompt_tokens"] == count
    assert result.receipt["extension_beyond_pdf_reference"] is True
    assert calls[0][0] == messages
    assert calls[0][1]["total_token_cap"] == count + 8
    assert "output_contract" not in calls[0][1]
    assert messages[0]["content"].endswith(HOTPOT_SHORT_ANSWER_V1)
    assert "Paris" not in json.dumps(messages) and "PRIVATE_METADATA" not in json.dumps(messages)
    legacy = thaw(candidate.runtime)
    del legacy["generation"]["output_contract"]
    assert candidate.uid != replace(candidate, runtime=legacy).uid


def test_contract_token_cost_is_enforced_before_dispatch():
    executor, candidate, task, scope, calls = fixture()
    count = executor.prompt_tokens(executor.generation_messages(candidate, task))
    with pytest.raises(ValueError, match="require"):
        executor.validate_prompt_candidate(candidate, [task], count + 7)
    result = executor.run(candidate, task, scope, count + 7)
    assert result.status == "interrupted" and result.actual_cost == 0
    assert not calls


def test_legacy_messages_and_receipts_stay_unchanged():
    executor, candidate, task, scope, calls = fixture(False)
    assert executor.generation_messages(candidate, task) == [
        {"role": "system", "content": "Mention all relevant entities."},
        {"role": "user", "content": "PUBLIC_QUESTION"}]
    result = executor.run(candidate, task, scope, 1000)
    assert "output_contract" not in result.receipt
    assert "extension_beyond_pdf_reference" not in result.receipt


@pytest.mark.parametrize("contract", [None, "unknown", True, {}])
def test_unknown_contracts_are_rejected(contract):
    with pytest.raises(ValueError, match="Unknown output_contract"):
        generation_messages("system", "public", {"output_contract": contract}, dataset="hotpotqa")


@pytest.mark.parametrize("artifact", ["workflow", "answer_strategy"])
def test_unimplemented_recipe_combinations_fail_before_dispatch(artifact):
    executor, candidate, task, scope, calls = fixture()
    candidate = replace(candidate, artifacts={**thaw(candidate.artifacts), artifact: {}})
    with pytest.raises(ValueError, match="single-prompt"):
        executor.validate_prompt_candidate(candidate, [task], 1000)
    with pytest.raises(ValueError, match="single-prompt"):
        executor.run(candidate, task, scope, 1000)
    assert not calls


def test_contract_cannot_silently_change_a_different_task_interface():
    executor, candidate, task, scope, calls = fixture(dataset="aime25")
    with pytest.raises(ValueError, match="HotpotQA"):
        executor.run(candidate, task, scope, 1000)
    assert not calls
