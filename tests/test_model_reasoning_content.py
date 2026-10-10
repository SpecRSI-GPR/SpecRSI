"""Separate backend notes must never silently replace the scored final answer."""
from dataclasses import asdict

import httpx
import pytest

from specrsi.models import LocalModelClient, ModelReply


def server_reply(monkeypatch, fields, *, final="", finish_reason="length"):
    # Actual vLLM message shape, with synthetic notes rather than benchmark data.
    message = {"role": "assistant", "content": final, "refusal": None,
               "annotations": None, "audio": None, "function_call": None,
               "tool_calls": [], **fields}
    response = {"id": "metered", "choices": [{"message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 13, "completion_tokens": 8},
                "specrsi_meter": {"elapsed_seconds": .5, "revision": "frozen", "gpu_count": 2}}
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json=response, request=httpx.Request("POST", url))

    monkeypatch.setattr("specrsi.models.client.httpx.post", post)
    reply = LocalModelClient("http://reasoning-unit-test", "synthetic-model").generate(
        [{"role": "user", "content": "Synthetic public question"}],
        max_tokens=8, total_token_cap=21, enable_thinking=True)
    assert len(calls) == 1 and calls[0]["json"]["enable_thinking"] is True
    return reply


@pytest.mark.parametrize("fields,expected", [
    ({"reasoning": "Partial derivation"}, "Partial derivation"),
    ({"reasoning_content": "Partial derivation"}, "Partial derivation"),
    ({"reasoning": "same", "reasoning_content": "same"}, "same"),
    ({"reasoning": "notes", "reasoning_content": None}, "notes"),
    ({"reasoning": None, "reasoning_content": "notes"}, "notes"),
    ({"reasoning": ""}, ""),
    ({"reasoning": None, "reasoning_content": None}, None),
    ({}, None),
])
def test_backend_reasoning_aliases_are_preserved_separately(monkeypatch, fields, expected):
    reply = server_reply(monkeypatch, fields)
    assert reply.text == "" and reply.reasoning_content == expected
    assert reply.reasoning_error is None and reply.finish_reason == "length"
    assert reply.prompt_tokens + reply.completion_tokens == 21
    assert reply.gpu_count == 2 and asdict(reply)["reasoning_content"] == expected


@pytest.mark.parametrize("fields,error", [
    ({"reasoning": ["not a string"]}, "invalid_reasoning_type"),
    ({"reasoning_content": {"text": "not a string"}}, "invalid_reasoning_type"),
    ({"reasoning": False}, "invalid_reasoning_type"),
    ({"reasoning_content": 42}, "invalid_reasoning_type"),
    ({"reasoning": "valid", "reasoning_content": []}, "invalid_reasoning_type"),
    ({"reasoning": "one", "reasoning_content": "two"}, "conflicting_reasoning_fields"),
    ({"reasoning": "", "reasoning_content": "nonempty"}, "conflicting_reasoning_fields"),
])
def test_invalid_or_conflicting_notes_keep_usage_without_constructing_notes(monkeypatch, fields, error):
    reply = server_reply(monkeypatch, fields, final="final answer", finish_reason="stop")
    assert reply.text == "final answer"
    assert reply.reasoning_content is None and reply.reasoning_error == error
    assert (reply.prompt_tokens, reply.completion_tokens, reply.elapsed_seconds) == (13, 8, .5)


def test_timeout_and_original_final_are_preserved_with_notes(monkeypatch):
    reply = server_reply(monkeypatch, {"reasoning": "Unfinished work"}, final="partial", finish_reason="timeout")
    assert reply.text == "partial" and reply.finish_reason == "timeout"
    assert reply.reasoning_content == "Unfinished work"


def test_model_reply_keeps_old_positional_gpu_argument_and_rejects_inconsistent_notes():
    old = ModelReply("answer", 1, 2, .3, "stop", "revision", "request", 7)
    assert old.gpu_count == 7 and old.reasoning_content is None and old.reasoning_error is None
    with pytest.raises(ValueError, match="invalid reasoning"):
        ModelReply("", 1, 2, .3, "length", "revision", "request", reasoning_content=[])
    with pytest.raises(ValueError, match="also be usable"):
        ModelReply("", 1, 2, .3, "length", "revision", "request",
                   reasoning_content="notes", reasoning_error="conflicting_reasoning_fields")
