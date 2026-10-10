from dataclasses import replace
from types import SimpleNamespace

import pytest

from specrsi.answer_strategies import AnswerStrategy, execute_answer_strategy, evidence_messages
from specrsi.models import ModelReply


def recipe(**kwargs):
    return AnswerStrategy(version="native_completion_v3", mode="reasoning_extract",
                          answer_format="boxed_integer", reasoning_tokens=20,
                          answer_tokens=12, **kwargs)


def execute(replies, *, cap=80, strategy=None, sampling=None):
    calls = []
    pending = iter(replies)
    def generate(messages, **kwargs):
        calls.append((messages, kwargs))
        reply = next(pending)
        if isinstance(reply, Exception):
            raise reply
        return reply
    executor = SimpleNamespace(client=SimpleNamespace(generate=generate),
                               prompt_tokens=lambda messages, **kwargs: 10)
    result = execute_answer_strategy(strategy or recipe(), "PUBLIC_INPUT", "REGISTERED_INSTRUCTION",
        executor, generation={"enable_thinking": True, "top_p": .95, **(sampling or {})},
        model_revision="frozen", cap=cap, seed=7)
    return result, calls


def reply(text, *, finish="stop", notes=None, prompt=10, completion=5, revision="frozen", error=None):
    return ModelReply(text, prompt, completion, .2, finish, revision, "request-"+str(completion),
                      reasoning_content=notes, reasoning_error=error)


def test_native_request_matches_control_and_preserves_even_wrong_complete_answer():
    native = r"The calculation gives 42. Therefore \boxed{42}."
    result, calls = execute([reply(native)])
    assert result.status == "complete" and result.answer == native
    assert result.tokens == 15 and result.decision == "native_final_preserved"
    assert len(calls) == 1
    assert calls[0][0] == [{"role": "system", "content": "REGISTERED_INSTRUCTION"},
                           {"role": "user", "content": "PUBLIC_INPUT"}]
    assert calls[0][1]["enable_thinking"] is True
    # The strategy has no reference argument and cannot preserve only correct answers.
    assert "reference" not in str(calls)


@pytest.mark.parametrize("final,finish", [
    ("", "length"), (r"\boxed{42}", "length"), ("42", "stop"),
    (r"\boxed{42} or \boxed{43}", "stop"), (r"\boxed{1000}", "stop"),
    (r"\boxed{42} then \boxed{", "stop"), (r"\boxed{-1}", "stop"),
])
def test_incomplete_or_ambiguous_native_answer_gets_one_metered_completion(final, finish):
    result, calls = execute([reply(final, finish=finish, notes="PRIOR_WORK"),
                             reply(r"\boxed{17}", completion=6)])
    assert result.status == "complete" and result.answer == r"\boxed{17}"
    assert result.decision == "native_budget_completion" and result.tokens == 31
    assert len(calls) == 2 and calls[1][1]["enable_thinking"] is False
    assert [kwargs["max_tokens"] for _, kwargs in calls] == [20, 12]
    assert [kwargs["seed"] for _, kwargs in calls] == [7, 8]
    assert "PRIOR_WORK" in calls[1][0][1]["content"]
    if final:
        assert final in calls[1][0][1]["content"]
    assert "PUBLIC_INPUT" in calls[1][0][1]["content"]


def test_completed_integer_with_zero_padding_is_preserved_without_canonicalizing_text():
    result, calls = execute([reply(r"\boxed{007}")])
    assert result.answer == r"\boxed{007}" and len(calls) == 1


@pytest.mark.parametrize("first,status", [
    (reply("", finish="timeout", notes="paid partial"), "interrupted"),
    (reply("partial", revision="changed"), "error"),
    (reply("partial", prompt=11), "error"),
    (reply("", notes=None, error="invalid_reasoning_type"), "error"),
    (reply("partial", notes=None, error="conflicting_reasoning_fields"), "error"),
    (reply("", notes=None), "error"),
])
def test_failed_native_stage_cannot_be_recovered_for_free(first, status):
    result, calls = execute([first])
    assert result.status == status and result.answer is None and len(calls) == 1
    assert result.tokens == first.prompt_tokens + first.completion_tokens


def test_recovery_never_shrinks_output_to_fit_remaining_case_budget():
    result, calls = execute([reply("", finish="length", notes="work")], cap=35)
    assert result.status == "interrupted" and result.answer is None
    assert result.tokens == 15 and len(calls) == 1


def test_lost_recovery_receipt_retains_known_cost_and_full_unknown_reservation():
    result, calls = execute([reply("", finish="length", notes="work"), TimeoutError()])
    assert result.status == "error" and result.tokens == 15 + 22
    assert result.records[-1]["metering_status"] == "unavailable_reserved_upper_bound"
    assert len(calls) == 2


def test_sampling_parameters_reach_both_stages_and_default_payload_is_unchanged():
    params = {"top_k": 20, "min_p": 0., "presence_penalty": 1.5, "repetition_penalty": 1.}
    _, calls = execute([reply("", finish="length", notes="work"), reply(r"\boxed{17}")], sampling=params)
    assert all(all(kwargs[k] == v for k, v in params.items()) for _, kwargs in calls)
    _, calls = execute([reply(r"\boxed{17}")])
    assert not set(params).intersection(calls[0][1])


def test_long_native_caps_are_explicitly_versioned():
    long = replace(recipe(), reasoning_tokens=24576, answer_tokens=2048)
    assert long.reasoning_tokens == 24576
    with pytest.raises(ValueError):
        replace(long, reasoning_tokens=24577)
    with pytest.raises(ValueError):
        replace(long, version="reasoning_extract_v2")
    with pytest.raises(ValueError):
        replace(long, mode="independent_consensus", samples=2)
    with pytest.raises(ValueError):
        replace(long, answer_format="short_answer")


def test_native_completion_requires_native_thinking_before_any_request():
    from specrsi.answer_strategies import reasoning_enable_thinking
    for disabled in (False, None, "true"):
        with pytest.raises(ValueError):
            reasoning_enable_thinking(recipe(), {"enable_thinking": disabled})
    assert evidence_messages("system", "task", recipe())[0]["content"] == "system"
