import math

import httpx
import pytest

from specrsi.models.client import EXTENDED_SAMPLING_PARAMETERS, LocalModelClient, sampling_options


def fake_endpoint(monkeypatch, declaration):
    seen = {"health": 0, "payloads": []}

    def get(url, **kwargs):
        seen["health"] += 1
        return httpx.Response(200, json=declaration, request=httpx.Request("GET", url))

    def post(url, *, json, **kwargs):
        seen["payloads"].append(json)
        return httpx.Response(200, json={"id": "physical-" + str(len(seen["payloads"])),
            "choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            "specrsi_meter": {"elapsed_seconds": .1, "revision": "pinned"}},
            request=httpx.Request("POST", url))

    monkeypatch.setattr("specrsi.models.client.httpx.get", get)
    monkeypatch.setattr("specrsi.models.client.httpx.post", post)
    return seen


@pytest.mark.parametrize("optional", [{}, dict.fromkeys(EXTENDED_SAMPLING_PARAMETERS)])
def test_omitted_and_none_sampling_preserve_legacy_payload_and_no_health(monkeypatch, optional):
    seen = fake_endpoint(monkeypatch, {})
    messages = [{"role": "user", "content": "2+2?"}]
    reply = LocalModelClient("http://sampling-unit-test", "model").generate(messages, **optional)
    assert reply.text == "4"
    assert seen == {"health": 0, "payloads": [{"model": "model", "messages": messages,
        "max_tokens": 512, "temperature": .7, "top_p": .8, "seed": 0,
        "enable_thinking": False, "max_seconds": 120}]}


def test_explicit_sampling_is_forwarded_after_capability_check_and_cached(monkeypatch):
    seen = fake_endpoint(monkeypatch, {"supported_sampling_parameters": list(EXTENDED_SAMPLING_PARAMETERS)})
    options = {"top_k": 20, "min_p": 0., "presence_penalty": 1.5, "repetition_penalty": 1.}
    client = LocalModelClient("http://sampling-unit-test", "model")
    client.ensure_sampling_support(options)
    for _ in range(2):
        client.generate([{"role": "user", "content": "2+2?"}], total_token_cap=32, **options)
    assert seen["health"] == 1
    assert len(seen["payloads"]) == 2
    assert all({name: payload[name] for name in options} == options for payload in seen["payloads"])


@pytest.mark.parametrize("declaration", [{}, {"supported_sampling_parameters": ["temperature"]},
    {"supported_sampling_parameters": "presence_penalty"}, {"supported_sampling_parameters": [True]}])
def test_missing_or_invalid_endpoint_capability_refuses_before_generation(monkeypatch, declaration):
    seen = fake_endpoint(monkeypatch, declaration)
    with pytest.raises(ValueError, match="sampling"):
        LocalModelClient("http://sampling-unit-test").generate([], presence_penalty=1.5)
    assert seen["health"] == 1
    assert seen["payloads"] == []


INVALID = [("top_k", value) for value in (-2, 1.5, True, "20", math.nan, math.inf)] + [
    (name, value) for name in ("min_p", "presence_penalty", "repetition_penalty")
    for value in (True, "0.5", math.nan, math.inf, -math.inf)] + [
    ("min_p", -.1), ("min_p", 1.1), ("presence_penalty", -2.1),
    ("presence_penalty", 2.1), ("repetition_penalty", 0), ("repetition_penalty", -1)]


@pytest.mark.parametrize("name,value", INVALID)
def test_invalid_sampling_fails_helper_and_client_before_any_http(monkeypatch, name, value):
    seen = fake_endpoint(monkeypatch, {"supported_sampling_parameters": list(EXTENDED_SAMPLING_PARAMETERS)})
    with pytest.raises(ValueError, match=name):
        sampling_options({name: value})
    with pytest.raises(ValueError, match=name):
        LocalModelClient("http://sampling-unit-test").generate([], **{name: value})
    assert seen == {"health": 0, "payloads": []}


@pytest.mark.parametrize("options", [
    {"top_k": -1, "min_p": 0, "presence_penalty": -2, "repetition_penalty": .1},
    {"top_k": 0, "min_p": 1, "presence_penalty": 2, "repetition_penalty": 2},
    {"top_k": 20, "min_p": .1, "presence_penalty": 1.5, "repetition_penalty": 1.},
])
def test_sampler_boundaries_and_generation_filter(options):
    assert sampling_options({"max_tokens": 128, "temperature": 1., **options}) == options
    assert sampling_options({"max_tokens": 128, **dict.fromkeys(EXTENDED_SAMPLING_PARAMETERS)}) == {}
    with pytest.raises(ValueError, match="mapping"):
        sampling_options(None)
