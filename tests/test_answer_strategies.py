from dataclasses import FrozenInstanceError, replace
import json
import random
from types import SimpleNamespace

import pytest

from specrsi.answer_strategies import AnswerStrategy, execute_answer_strategy
from specrsi.core.models import Candidate, EvaluationScope, Task, thaw
from specrsi.experiments import QwenDrafter, QwenExecutor, carried_initial_artifacts
from specrsi.models import ModelReply


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [0] * 10


def fixture(outputs, strategy=None, *, reference="Paris", dataset="hotpotqa", tokenizer=None, enable_thinking=True):
    strategy = strategy or AnswerStrategy(reasoning_tokens=8, answer_tokens=4, arbiter_tokens=4)
    calls = []
    pending = iter(outputs)

    def generate(messages, **kwargs):
        calls.append((messages, kwargs))
        output = next(pending)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, ModelReply):
            return output
        return ModelReply(output, 10, 2, .1, "stop", "frozen", f"request-{len(calls)}")

    generation = {"max_tokens": 9, "temperature": 1.9, "top_p": .7, "enable_thinking": enable_thinking}
    executor = QwenExecutor(SimpleNamespace(generate=generate), tokenizer or Tokenizer(), generation)
    candidate = Candidate("fixed", {"system_prompt": "Solve accurately.", "answer_strategy": strategy.to_dict()},
                          {"generation": generation, "model_revision": "frozen", "seed": 3})
    task = Task("one", {"prompt": "Public question and supplied context"}, reference,
                {"dataset": dataset, "private": "PRIVATE_METADATA_SENTINEL"})
    scope = EvaluationScope("strategy", (task,), "answer-strategy-v1", "native-scorer-v1")
    return executor, candidate, task, scope, calls


def test_evidence_extract_is_metered_and_cannot_see_reference_or_private_metadata():
    executor, candidate, task, scope, calls = fixture(["Decisive evidence", "Paris"], reference="REFERENCE_SENTINEL")
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "complete" and result.actual_cost == 24
    assert result.outcome.fields["answer"] == "Paris"
    serialized = json.dumps(calls)
    assert "REFERENCE_SENTINEL" not in serialized and "PRIVATE_METADATA_SENTINEL" not in serialized
    assert "Decisive evidence" in calls[1][0][1]["content"]
    assert [kwargs["max_tokens"] for _, kwargs in calls] == [8, 4]
    assert [kwargs["total_token_cap"] for _, kwargs in calls] == [18, 14]
    assert [kwargs["temperature"] for _, kwargs in calls] == [.3, 0.]
    assert all(kwargs["enable_thinking"] is False for _, kwargs in calls)
    assert all(kwargs["top_p"] == .7 for _, kwargs in calls)
    assert result.receipt["metering_complete"] is True
    assert result.receipt["extension_beyond_pdf_reference"] is True
    assert len(executor.physical_records) == 2


def test_generic_math_extractor_enforces_boxed_integer_and_uses_native_score():
    strategy = AnswerStrategy(version="reasoning_extract_v1", mode="reasoning_extract",
                              answer_format="boxed_integer", reasoning_tokens=8, answer_tokens=4)
    executor, candidate, task, scope, calls = fixture(["Compute 40 + 2 = 42.", r"\boxed{42}"],
                                                     strategy, reference=42, dataset="aime25")
    executor.validate_prompt_candidate(candidate, [task], 40)
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "complete" and result.outcome.score == 1.
    assert "exact arithmetic" in calls[0][0][0]["content"]
    assert r"\boxed{...}" in calls[1][0][0]["content"]
    assert result.outcome.fields["strategy_decision"] == "single_reasoning_extraction"


def test_unanimous_independent_samples_skip_arbiter_without_sharing_notes():
    strategy = AnswerStrategy(mode="independent_consensus", samples=2, reasoning_tokens=8, answer_tokens=4)
    executor, candidate, task, scope, calls = fixture(["FIRST_NOTES", "Paris", "SECOND_NOTES", "paris."], strategy)
    result = executor.run(candidate, task, scope, 80)
    assert result.status == "complete" and result.actual_cost == 48
    assert result.outcome.fields["strategy_decision"] == "unanimous_agreement"
    assert result.outcome.score == 1.
    assert len(calls) == 4
    assert calls[0][0] == calls[2][0]
    assert "FIRST_NOTES" not in json.dumps(calls[2][0])
    assert [kwargs["seed"] for _, kwargs in calls] == [3, 4, 5, 6]


def test_disagreement_invokes_and_charges_frozen_arbiter():
    strategy = AnswerStrategy(mode="independent_consensus", samples=2, reasoning_tokens=8, answer_tokens=4, arbiter_tokens=6)
    executor, candidate, task, scope, calls = fixture(["e1", "Paris", "e2", "London", "Paris"], strategy)
    result = executor.run(candidate, task, scope, 80)
    assert result.status == "complete" and result.actual_cost == 60
    assert result.outcome.fields["strategy_decision"] == "evidence_arbiter"
    assert calls[-1][1]["max_tokens"] == 6 and calls[-1][1]["total_token_cap"] == 16
    assert "Paris" in calls[-1][0][1]["content"] and "London" in calls[-1][0][1]["content"]
    assert len(result.receipt["requests"]) == 5


def test_majority_must_be_strict_and_does_not_choose_a_tie_for_free():
    strategy = AnswerStrategy(mode="independent_consensus", samples=3, agreement="majority", reasoning_tokens=8, answer_tokens=4)
    executor, candidate, task, scope, calls = fixture(["e1", "Paris", "e2", "London", "e3", "Paris"], strategy)
    result = executor.run(candidate, task, scope, 100)
    assert result.status == "complete" and result.outcome.fields["strategy_decision"] == "majority_agreement"
    assert len(calls) == 6
    strategy = replace(strategy, samples=2)
    executor, candidate, task, scope, calls = fixture(["e1", "Paris", "e2", "London", "Paris"], strategy)
    result = executor.run(candidate, task, scope, 200)
    assert result.status == "complete" and len(calls) == 5


def test_insufficient_remaining_budget_preserves_paid_prefix_and_never_shrinks_output():
    executor, candidate, task, scope, calls = fixture(["evidence"])
    result = executor.run(candidate, task, scope, 25)
    assert result.status == "interrupted" and result.actual_cost == 12
    assert result.outcome is None and len(calls) == 1
    assert len(executor.physical_records) == 1
    executor, candidate, task, scope, calls = fixture([])
    result = executor.run(candidate, task, scope, 17)
    assert result.status == "interrupted" and result.actual_cost == 0 and not calls


@pytest.mark.parametrize("reply,status,error", [
    (ModelReply("unfinished", 10, 2, .5, "timeout", "frozen", "failed"), "interrupted", "timed out"),
    (ModelReply("bad count", 11, 2, .5, "stop", "frozen", "failed"), "error", "tokenizer"),
    (ModelReply("wrong model", 10, 2, .5, "stop", "other", "failed"), "error", "revision"),
    (ModelReply("over budget", 10, 20, .5, "stop", "frozen", "failed"), "error", "violated"),
])
def test_failed_replies_always_preserve_measured_cost_and_never_score(reply, status, error):
    executor, candidate, task, scope, calls = fixture([reply])
    result = executor.run(candidate, task, scope, 40)
    assert result.status == status and error in result.error and result.outcome is None
    assert result.actual_cost == reply.prompt_tokens + reply.completion_tokens
    assert len(calls) == len(executor.physical_records) == 1
    assert result.receipt["metering_complete"] is True


def test_transport_failure_charges_reserved_bound_and_is_explicitly_unmeasured():
    executor, candidate, task, scope, calls = fixture(["paid evidence", TimeoutError("no usage receipt")])
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and result.actual_cost == 12 + 14
    assert result.outcome is None and len(calls) == 2
    assert result.receipt["metering_complete"] is False
    failed = executor.physical_records[-1]
    assert failed["metering_status"] == "unavailable_reserved_upper_bound"
    assert failed["accounted_tokens"] == failed["reserved_tokens"] == 14
    assert "completion_tokens" not in failed and "elapsed_seconds" not in failed
    assert sum(record["accounted_tokens"] for record in executor.physical_records) == result.actual_cost


def test_native_scorer_failure_preserves_all_paid_generation_costs(monkeypatch):
    executor, candidate, task, scope, calls = fixture(["evidence", "Paris"])
    def failing_scorer(*args, **kwargs):
        raise RuntimeError("native scorer unavailable")
    monkeypatch.setattr("specrsi.experiments.score_answer", failing_scorer)
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and result.actual_cost == 24
    assert result.outcome is None and "Native scorer failed" in result.error
    assert len(result.receipt["requests"]) == len(calls) == len(executor.physical_records) == 2


def test_strategy_identity_is_immutable_and_changes_with_recipe():
    strategy = AnswerStrategy()
    with pytest.raises(FrozenInstanceError):
        strategy.samples = 4
    _, candidate, _, _, _ = fixture([], strategy)
    changed = replace(candidate, artifacts={"system_prompt": "Solve accurately.",
                                            "answer_strategy": replace(strategy, reasoning_tokens=512).to_dict()})
    assert candidate.uid != changed.uid
    with pytest.raises(TypeError):
        candidate.artifacts["answer_strategy"]["reasoning_tokens"] = 999


@pytest.mark.parametrize("change", [
    {"version": "unregistered"}, {"samples": True}, {"samples": 2},
    {"mode": "independent_consensus"}, {"reasoning_tokens": 0}, {"answer_tokens": 1.5},
    {"temperature": float("nan")}, {"unknown": 1}, {"answer_format": "boxed_integer"},
])
def test_invalid_recipe_fails_before_inference(change):
    raw = AnswerStrategy().to_dict()
    raw.update(change)
    with pytest.raises(ValueError):
        AnswerStrategy.from_dict(raw)


def test_initial_and_carried_manifest_preserve_frozen_strategy(tmp_path):
    strategy = AnswerStrategy().to_dict()
    artifacts, _ = carried_initial_artifacts({}, {"initial_prompt": "Solve accurately.", "answer_strategy": strategy})
    assert artifacts["answer_strategy"] == strategy
    _, candidate, _, _, _ = fixture([], AnswerStrategy())
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(candidate.manifest()))
    artifacts, receipt = carried_initial_artifacts({"initial_candidate_manifest": str(path)}, {})
    assert artifacts == thaw(candidate.artifacts)
    assert receipt["source_candidate_id"] == candidate.uid


def test_qwen_drafter_edits_prompt_without_silently_dropping_strategy():
    executor, candidate, task, scope, calls = fixture([json.dumps({"system_prompt": "Revised instruction"})])
    state = SimpleNamespace(scope=scope, candidates={candidate.uid: candidate}, ledger=SimpleNamespace(get=lambda *args: None))
    drafted = QwenDrafter(executor, execution_cap=40).draft(candidate, state, 1000, random.Random(0))
    assert drafted.status == "complete"
    assert drafted.candidate.artifacts["system_prompt"] == "Revised instruction"
    assert drafted.candidate.artifacts["answer_strategy"] == candidate.artifacts["answer_strategy"]


def test_legacy_candidate_keeps_single_call_and_original_generation_contract():
    executor, candidate, task, scope, calls = fixture(["Paris"])
    generation = thaw(candidate.runtime["generation"])
    generation["enable_thinking"] = False
    candidate = replace(candidate, artifacts={"system_prompt": "Solve accurately."},
                        runtime={**thaw(candidate.runtime), "generation": generation})
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "complete" and len(calls) == 1
    assert calls[0][1]["max_tokens"] == 9 and calls[0][1]["temperature"] == 1.9


class ThinkingTokenizer:
    """Different template sizes expose mismatches hidden by constant tokenizers."""
    def __init__(self):
        self.flags = []

    def apply_chat_template(self, messages, **kwargs):
        enabled = kwargs["enable_thinking"]
        assert type(enabled) is bool and kwargs["thinking"] == enabled
        self.flags.append(enabled)
        return [0] * (13 if enabled else 10)


def v2_strategy(**kwargs):
    return AnswerStrategy(version="reasoning_extract_v2", mode="reasoning_extract",
                          reasoning_tokens=8, answer_tokens=4, arbiter_tokens=4,
                          answer_format="boxed_integer", **kwargs)


def thinking_reply(text="", *, notes="Partial derivation: 40 + 2 = 42", finish="length", **kwargs):
    return ModelReply(text, 13, 8, .5, finish, "frozen", "thinking-stage",
                      reasoning_content=notes, **kwargs)


def test_v2_length_uses_real_reasoning_as_fallible_notes_and_charges_both_stages():
    tokenizer = ThinkingTokenizer()
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply(), r"\boxed{42}"], v2_strategy(), reference="REFERENCE_SENTINEL",
        tokenizer=tokenizer)
    executor.validate_prompt_candidate(candidate, [task], 40)
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "complete" and result.actual_cost == 33
    assert result.outcome.fields["answer"] == r"\boxed{42}"
    assert tokenizer.flags == [True, True, False]  # preflight, reasoning, extraction
    assert [kwargs["enable_thinking"] for _, kwargs in calls] == [True, False]
    assert [kwargs["total_token_cap"] for _, kwargs in calls] == [21, 14]
    assert [kwargs["max_tokens"] for _, kwargs in calls] == [8, 4]
    assert "exact arithmetic" in calls[0][0][0]["content"]
    assert "Fallible solution notes:\nPartial derivation" in calls[1][0][1]["content"]
    assert "Partial derivation" not in calls[1][0][0]["content"]
    assert "REFERENCE_SENTINEL" not in json.dumps(calls)
    assert "PRIVATE_METADATA_SENTINEL" not in json.dumps(calls)
    reasoning = executor.physical_records[0]
    assert reasoning["text"] == "" and reasoning["content_source"] == "reasoning_content"
    assert reasoning["finish_reason"] == "length" and reasoning["accounted_tokens"] == 21
    assert result.receipt["metering_complete"] is True


@pytest.mark.parametrize("notes", [None, "", "   "])
def test_v2_empty_final_and_notes_fail_with_exact_cost(notes):
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply(notes=notes)], v2_strategy(), tokenizer=ThinkingTokenizer())
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and "empty result" in result.error
    assert result.actual_cost == 21 and result.outcome is None and len(calls) == 1


def test_v2_stop_final_has_priority_over_separate_notes():
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply("FINAL_NOTES", notes="IGNORED_NOTES", finish="stop"), r"\boxed{42}"],
        v2_strategy(), tokenizer=ThinkingTokenizer())
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "complete" and result.actual_cost == 33
    assert "FINAL_NOTES" in calls[1][0][1]["content"]
    assert "IGNORED_NOTES" not in json.dumps(calls[1][0])
    assert executor.physical_records[0]["content_source"] == "final"


@pytest.mark.parametrize("text", ["", "even a final cannot rescue a timeout"])
def test_v2_timeout_with_reasoning_is_never_extracted_or_scored(text):
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply(text, finish="timeout")], v2_strategy(), tokenizer=ThinkingTokenizer())
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "interrupted" and "timed out" in result.error
    assert result.outcome is None and result.actual_cost == 21 and len(calls) == 1
    assert result.receipt["metering_complete"] is True


@pytest.mark.parametrize("error", ["invalid_reasoning_type", "conflicting_reasoning_fields"])
def test_v2_rejected_reasoning_preserves_measured_usage(error):
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply(notes=None, reasoning_error=error)], v2_strategy(), tokenizer=ThinkingTokenizer())
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and error in result.error
    assert result.actual_cost == 21 and result.outcome is None and len(calls) == 1
    assert result.receipt["metering_complete"] is True


def test_v2_extraction_does_not_consume_its_own_reasoning_as_an_answer():
    extraction = ModelReply("", 10, 4, .1, "length", "frozen", "extract",
                            reasoning_content=r"\boxed{42}")
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply(), extraction], v2_strategy(), tokenizer=ThinkingTokenizer())
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and "extract_0 generated an empty result" in result.error
    assert result.actual_cost == 35 and result.outcome is None and len(calls) == 2
    assert calls[1][1]["enable_thinking"] is False


@pytest.mark.parametrize("version,mode", [
    ("reasoning_extract_v1", "reasoning_extract"), ("hotpot_evidence_v1", "evidence_extract"),
])
def test_old_versions_ignore_native_thinking_and_never_recover_separate_notes(version, mode):
    strategy = AnswerStrategy(version=version, mode=mode, reasoning_tokens=8, answer_tokens=4)
    old_reply = ModelReply("", 10, 8, .1, "length", "frozen", "old", reasoning_content="usable notes")
    executor, candidate, task, scope, calls = fixture([old_reply], strategy)
    executor.validate_prompt_candidate(candidate, [task], 18)
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and "empty result" in result.error
    assert result.actual_cost == 18 and result.outcome is None and len(calls) == 1
    assert calls[0][1]["enable_thinking"] is False


def test_v2_consensus_reasoning_is_independent_and_arbiter_is_nonthinking():
    strategy = replace(v2_strategy(), mode="independent_consensus", samples=2)
    tokenizer = ThinkingTokenizer()
    second = replace(thinking_reply(notes="SECOND_NOTES"), request_id="second")
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply(notes="FIRST_NOTES"), r"\boxed{41}", second, r"\boxed{42}", r"\boxed{42}"],
        strategy, reference=42, dataset="aime25", tokenizer=tokenizer)
    result = executor.run(candidate, task, scope, 100)
    assert result.status == "complete" and result.outcome.score == 1. and result.actual_cost == 78
    assert tokenizer.flags == [True, False, True, False, False]
    assert [kwargs["enable_thinking"] for _, kwargs in calls] == tokenizer.flags
    assert calls[0][0] == calls[2][0] and "FIRST_NOTES" not in json.dumps(calls[2][0])
    assert "FIRST_NOTES" in calls[-1][0][1]["content"] and "SECOND_NOTES" in calls[-1][0][1]["content"]
    assert result.outcome.fields["strategy_decision"] == "reasoning_arbiter"


def test_v2_budget_preflight_counts_thinking_template_and_does_not_shorten_later_stage():
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply()], v2_strategy(), tokenizer=ThinkingTokenizer())
    with pytest.raises(ValueError, match="requires 21 tokens"):
        executor.validate_prompt_candidate(candidate, [task], 20)
    executor.validate_prompt_candidate(candidate, [task], 21)
    result = executor.run(candidate, task, scope, 20)
    assert result.status == "interrupted" and result.actual_cost == 0 and calls == []
    result = executor.run(candidate, task, scope, 34)  # 21 paid, fewer than 14 left
    assert result.status == "interrupted" and result.actual_cost == 21 and len(calls) == 1
    assert result.outcome is None and calls[0][1]["max_tokens"] == 8


def test_v2_respects_disabled_thinking_and_rejects_nonboolean_before_request():
    executor, candidate, task, scope, calls = fixture(
        ["ordinary derivation", r"\boxed{42}"], v2_strategy(), enable_thinking=False)
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "complete" and all(not kwargs["enable_thinking"] for _, kwargs in calls)
    executor, candidate, task, scope, calls = fixture([], v2_strategy(), enable_thinking="false")
    with pytest.raises(ValueError, match="boolean enable_thinking"):
        executor.validate_prompt_candidate(candidate, [task], 40)
    with pytest.raises(ValueError, match="boolean enable_thinking"):
        executor.run(candidate, task, scope, 40)
    assert calls == []


@pytest.mark.parametrize("change,error", [
    ({"model_revision": "changed"}, "revision changed"),
    ({"prompt_tokens": 12}, "tokenizer counts"),
    ({"completion_tokens": 9}, "violated enforced token cap"),
])
def test_v2_reasoning_cannot_rescue_invalid_meter_or_model_identity(change, error):
    reply = replace(thinking_reply(), **change)
    executor, candidate, task, scope, calls = fixture([reply], v2_strategy(), tokenizer=ThinkingTokenizer())
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and error in result.error
    assert result.actual_cost == reply.prompt_tokens + reply.completion_tokens
    assert result.outcome is None and len(calls) == 1


def test_v2_unknown_extraction_failure_still_charges_full_reserved_bound():
    executor, candidate, task, scope, calls = fixture(
        [thinking_reply(), TimeoutError("receipt unavailable")], v2_strategy(), tokenizer=ThinkingTokenizer())
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "error" and result.actual_cost == 35 and result.outcome is None
    assert result.receipt["metering_complete"] is False and len(calls) == 2
    assert executor.physical_records[-1]["reserved_tokens"] == 14


def test_v2_expansion_is_versioned_and_only_expands_reasoning_budget():
    old = AnswerStrategy(version="reasoning_extract_v1", mode="reasoning_extract")
    with pytest.raises(ValueError, match="4096"):
        replace(old, reasoning_tokens=6144)
    new = replace(old, version="reasoning_extract_v2", reasoning_tokens=6144)
    assert AnswerStrategy.from_dict(new.to_dict()) == new
    assert replace(new, reasoning_tokens=8192).reasoning_tokens == 8192
    for change in ({"reasoning_tokens": 8193}, {"answer_tokens": 4097}, {"arbiter_tokens": 4097}):
        with pytest.raises(ValueError):
            replace(new, **change)
    _, old_candidate, _, _, _ = fixture([], old)
    _, new_candidate, _, _, _ = fixture([], replace(old, version="reasoning_extract_v2"))
    assert old_candidate.uid != new_candidate.uid


def test_plain_executor_never_scores_separate_reasoning_in_place_of_final():
    reply = thinking_reply(notes=r"\boxed{42}")
    executor, candidate, task, scope, calls = fixture(
        [reply], v2_strategy(), reference=42, dataset="aime25", tokenizer=ThinkingTokenizer())
    candidate = replace(candidate, artifacts={"system_prompt": "Solve accurately."})
    result = executor.run(candidate, task, scope, 40)
    assert result.status == "complete" and result.actual_cost == 21
    assert result.outcome.score == 0. and result.outcome.fields["answer"] == ""
    assert len(calls) == 1 and executor.physical_records[0]["reasoning_content"] == r"\boxed{42}"
