"""Large trees must not hide the editable parent's actual failure feedback."""
import json
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from specrsi.core import Candidate, Outcome, Task
from specrsi.learning.bridge import HFPromptDrafter
from specrsi.learning.paper_bridge import VATSSearchBridge
from specrsi.learning.records import DecisionRecord
from specrsi.learning.sampler import SampledEdit


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("USER\n" + messages[0]["content"] + "\nASSISTANT\n")


def fixture(artifacts=None):
    model = SimpleNamespace(backend="hf", device="cpu", tokenizer=CharacterTokenizer(),
        config=SimpleNamespace(max_length=4096, feature_dim=8, option_dim=12))
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    parent = Candidate("parent", artifacts or {"system_prompt": "Current instruction " * 500})
    failed = Task("failed", {"prompt": "VISIBLE_INPUT_START " + "long supplied document " * 1000 + " QUESTION_AT_END"},
        "SECRET_REFERENCE_MUST_NOT_APPEAR")
    completed = Task("completed", {"prompt": "Other visible question"}, "OTHER_SECRET_REFERENCE")
    outcomes = {failed.uid: SimpleNamespace(outcome=Outcome(0., {"answer": "ACTUAL_PARENT_FAILURE", "finish_reason": "stop"})),
        completed.uid: SimpleNamespace(outcome=Outcome(1., {"answer": "OBSERVED_SUCCESS"}))}
    ledger = SimpleNamespace(get=lambda candidate, task, scope: outcomes.get(task.uid), missing=lambda *args: ())
    state = SimpleNamespace(scope=SimpleNamespace(tasks=(completed, failed)), ledger=ledger, version=1,
        # This large unrelated tree was previously serialized before feedback.
        candidates={f"other-{i}": "unrelated metadata " * 100 for i in range(100)})
    return model, bridge, parent, state


def recorded_sample(model, input_ids, text):
    prefix = tuple(input_ids)
    decisions = (DecisionRecord("edit_token", ("7", "8"), 0, math.log(.5), input_ids=prefix,
        behavior_probabilities=(.5, .5)),)
    return SampledEdit(prefix, (7,), decisions, text, len(prefix) + 1, "eos")


def test_bounded_parent_feedback_retains_task_tail_and_observed_failure_without_references():
    model, bridge, parent, state = fixture()
    text = bridge.drafter_context_text(state, parent, token_limit=1600)
    context = json.loads(text)
    assert len(model.tokenizer.encode(text)) <= 1600
    assert context["examples"][0]["parent_observed_outcome"]["score"] == 0.
    for value in ("VISIBLE_INPUT_START", "QUESTION_AT_END", "ACTUAL_PARENT_FAILURE"):
        assert value in text
    assert "SECRET_REFERENCE" not in text and "unrelated metadata" not in text


def test_final_draft_prompt_preserves_parent_feedback_under_full_model_cap(monkeypatch):
    model, bridge, parent, state = fixture()
    prompts = []

    def sample(model, input_ids, **kwargs):
        prompts.append(model.tokenizer.decode(input_ids))
        assert len(input_ids) + kwargs["max_new_tokens"] <= model.config.max_length
        return recorded_sample(model, input_ids, "A genuinely changed task instruction")

    monkeypatch.setattr("specrsi.learning.sampler.sample_edit_tokens", sample)
    result = HFPromptDrafter(bridge, max_new_tokens=64).draft(parent, state, 4096, random.Random(0))
    assert result.candidate is not None
    for value in ("VISIBLE_INPUT_START", "QUESTION_AT_END", "ACTUAL_PARENT_FAILURE"):
        assert value in prompts[0]
    assert "SECRET_REFERENCE" not in prompts[0]
    assert "Preserve the required output format and interface constraints" in prompts[0]


def test_opt_in_output_contract_is_visible_to_drafter_with_generalization_guidance(monkeypatch):
    from specrsi.output_contracts import HOTPOT_SHORT_ANSWER_V1
    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    parent = replace(parent, runtime={"generation": {"output_contract": "hotpot_short_answer_v1"}})
    prompts = []

    def sample(model, input_ids, **kwargs):
        prompts.append(model.tokenizer.decode(input_ids))
        assert len(input_ids) + kwargs["max_new_tokens"] <= 4096
        return recorded_sample(model, input_ids, "Check the requested relation before answering briefly.")

    monkeypatch.setattr("specrsi.learning.sampler.sample_edit_tokens", sample)
    result = HFPromptDrafter(bridge, max_new_tokens=64).draft(parent, state, 4096, random.Random(0))
    assert result.status == "complete" and result.candidate.runtime == parent.runtime
    assert HOTPOT_SHORT_ANSWER_V1 in prompts[0]
    assert "Do not hard-code" in prompts[0]
    assert "SECRET_REFERENCE" not in prompts[0]
    assert "ACTUAL_PARENT_FAILURE" in prompts[0]
    assert result.actual_cost == len(model.tokenizer.encode(prompts[0])) + 1


def test_feedback_has_no_fixed_256_token_ceiling_when_complete_evidence_fits():
    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    evidence = "START " + "important middle evidence " * 40 + " END"
    task = Task("long-but-affordable", {"prompt": evidence}, "PRIVATE_ANSWER")
    state.scope = SimpleNamespace(tasks=(task,))
    state.ledger = SimpleNamespace(get=lambda *args: SimpleNamespace(
        outcome=Outcome(0., {"answer": "Observed failure", "finish_reason": "stop"})), missing=lambda *args: ())
    text = bridge.drafter_context_text(state, parent, token_limit=1800)
    context = json.loads(text)
    assert json.loads(context["examples"][0]["visible_input"])["prompt"] == evidence
    assert len(model.tokenizer.encode(text)) <= 1800
    assert "PRIVATE_ANSWER" not in text


def test_short_fields_release_budget_and_larger_budget_reveals_more_actual_input():
    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    smaller = json.loads(bridge.drafter_context_text(state, parent, token_limit=1200))
    larger_text = bridge.drafter_context_text(state, parent, token_limit=3000)
    larger = json.loads(larger_text)
    assert len(larger["examples"]) == len(smaller["examples"]) == 2
    assert len(larger["examples"][0]["visible_input"]) > 3 * len(smaller["examples"][0]["visible_input"])
    assert "QUESTION_AT_END" in larger["examples"][0]["visible_input"]
    assert json.loads(larger["examples"][1]["visible_input"])["prompt"] == "Other visible question"
    assert len(model.tokenizer.encode(larger_text)) <= 3000
    assert "SECRET_REFERENCE" not in larger_text


def test_feedback_serialization_counts_escaping_and_never_cuts_json():
    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    task = Task("escaped", {"prompt": '\\"\n' * 2000}, "SECRET")
    state.scope = SimpleNamespace(tasks=(task,))
    state.ledger = SimpleNamespace(get=lambda *args: SimpleNamespace(
        outcome=Outcome(0., {"answer": '\\"\n' * 1000})), missing=lambda *args: ())
    text = bridge.drafter_context_text(state, parent, token_limit=1600)
    assert json.loads(text)["examples"][0]["parent_observed_outcome"]["score"] == 0.
    assert len(model.tokenizer.encode(text)) <= 1600
    assert "SECRET" not in text


def test_draft_uses_actual_work_cap_with_a_larger_model_context(monkeypatch):
    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    model.config.max_length = 8192
    prompts = []

    def sample(model, input_ids, **kwargs):
        assert len(input_ids) + kwargs["max_new_tokens"] <= 2048
        prompts.append(model.tokenizer.decode(input_ids))
        return recorded_sample(model, input_ids, "Use the relevant evidence and answer briefly.")

    monkeypatch.setattr("specrsi.learning.sampler.sample_edit_tokens", sample)
    result = HFPromptDrafter(bridge, max_new_tokens=64).draft(parent, state, 2048, random.Random(0))
    assert result.candidate is not None and len(prompts) == 1
    assert "ACTUAL_PARENT_FAILURE" in prompts[0] and "QUESTION_AT_END" in prompts[0]
    assert result.actual_cost <= 2048


def test_too_small_draft_cap_has_no_model_call_or_charge(monkeypatch):
    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    monkeypatch.setattr("specrsi.learning.sampler.sample_edit_tokens",
        lambda *args, **kwargs: pytest.fail("Insufficient cap must fail before generation"))
    result = HFPromptDrafter(bridge, max_new_tokens=64).draft(parent, state, 65, random.Random(0))
    assert result.candidate is None and result.status == "interrupted"
    assert result.actual_cost == 0 and not bridge.physical_inferences


def test_max_new_tokens_is_a_ceiling_and_allows_a_shorter_affordable_draft(monkeypatch):
    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    calls = []

    def sample(model, input_ids, **kwargs):
        calls.append(kwargs["max_new_tokens"])
        assert 0 < kwargs["max_new_tokens"] < 2048
        assert len(input_ids) + kwargs["max_new_tokens"] <= 1024
        return recorded_sample(model, input_ids, "Answer briefly using the supplied evidence.")

    monkeypatch.setattr("specrsi.learning.sampler.sample_edit_tokens", sample)
    result = HFPromptDrafter(bridge, max_new_tokens=2048).draft(parent, state, 1024, random.Random(0))
    assert result.candidate is not None and len(calls) == 1
    assert result.actual_cost <= 1024 and len(bridge.physical_inferences) == 1


@pytest.mark.parametrize("cap", [512, 768, 1024])
def test_feedback_clips_byte_tokens_on_original_unicode_boundaries(cap):
    class ByteTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            return list(text.encode("utf-8"))

        def decode(self, tokens, **kwargs):
            return bytes(tokens).decode("utf-8", errors="replace")

    model, bridge, parent, state = fixture({"system_prompt": "Answer briefly."})
    model.tokenizer = ByteTokenizer()
    task = Task("unicode", {"prompt": "👩🏽‍🔬" * 500}, "PRIVATE")
    state.scope = SimpleNamespace(tasks=(task,))
    state.ledger = SimpleNamespace(get=lambda *args: SimpleNamespace(
        outcome=Outcome(0., {"answer": "中文👩🏽‍🔬" * 100})), missing=lambda *args: ())
    text = bridge.drafter_context_text(state, parent, token_limit=cap)
    assert json.loads(text)["examples"]
    assert "\ufffd" not in text
    assert len(model.tokenizer.encode(text)) <= cap


@pytest.mark.parametrize("workflow", [False, True])
def test_unchanged_edit_is_rejected_but_keeps_paid_generation_and_actor_trace(monkeypatch, workflow):
    artifacts = {"workflow": {"nodes": [{"id": "solve", "instruction": "Answer the question", "dependencies": []}],
        "output_node": "solve"}} if workflow else {"system_prompt": "Same instruction"}
    model, bridge, parent, state = fixture(artifacts)
    text = json.dumps(artifacts) if workflow else artifacts["system_prompt"]
    samples = []

    def sample(model, input_ids, **kwargs):
        samples.append(recorded_sample(model, input_ids, text))
        return samples[-1]

    monkeypatch.setattr("specrsi.learning.sampler.sample_edit_tokens", sample)
    result = HFPromptDrafter(bridge, max_new_tokens=64).draft(parent, state, 4096, random.Random(0))
    assert result.candidate is None and result.status == "error"
    assert "did not change" in result.error
    assert result.actual_cost == samples[0].token_cost > 0
    assert bridge.token_decisions == list(samples[0].decisions)
    assert bridge.physical_inferences[0]["completion_tokens"] == 1
