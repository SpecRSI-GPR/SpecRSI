import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("fastapi")
import httpx
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location("serve_budget_gateway", Path(__file__).parents[1] / "scripts/serve_budget_gateway.py")
gateway = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gateway
spec.loader.exec_module(gateway)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is True
        return [1] * 10


def make_app(tmp_path, *, prompt_tokens=10, completion_tokens=5, final_content="4", engine_version="vllm-test"):
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"download_manifest": str(manifest), "revision": "pinned", "repo_id": "repo/model",
        "served_model_name": "model", "context_length": 32768, "gpu_ids": [1], "format": "publisher_fp8",
        "engine_version": engine_version, "local_path": "unused", "chat_template_thinking_key": "enable_thinking"}))
    manifest.write_text(json.dumps({"status": "publisher_verified_complete", "revision": "pinned",
        "repo_id": "repo/model", "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}))
    requests = []

    def backend(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"id": "physical-request-1", "choices": [{"finish_reason": "stop",
            "message": {"role": "assistant", "content": final_content, "reasoning": "Compute two plus two."}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens,
                      "completion_tokens_details": {"reasoning_tokens": 4}}})

    log = tmp_path / "physical.jsonl"
    app = gateway.create_app(str(lock), "http://local", str(log), tokenizer=Tokenizer(), transport=httpx.MockTransport(backend))
    return app, requests, log


def payload(**changes):
    return {"model": "model", "messages": [{"role": "user", "content": "2+2?"}],
            "max_tokens": 8, "enable_thinking": True, "total_token_cap": 18, **changes}


@pytest.mark.parametrize("mapping_output", [False, True])
def test_gateway_preserves_reasoning_and_full_backend_usage(tmp_path, monkeypatch, mapping_output):
    if mapping_output:
        monkeypatch.setattr(Tokenizer, "apply_chat_template", lambda *a, **kw:
            {"input_ids": [[1] * 10], "attention_mask": [[1] * 10]})
    app, requests, log = make_app(tmp_path)
    with TestClient(app) as client:
        result = client.post("/v1/chat/completions", json=payload())
        assert result.status_code == 200
        value = result.json()
        assert value["choices"][0]["message"]["reasoning"] == "Compute two plus two."
        assert value["choices"][0]["message"]["content"] == "4"
        assert value["usage"]["completion_tokens"] == 5
        assert value["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4
        assert requests[0]["chat_template_kwargs"] == {"enable_thinking": True}
        assert json.loads(log.read_text())["response"] == value
        assert client.get("/health").json()["server_enforces_total_token_cap"] is True


def test_gateway_rejects_oversize_before_inference_and_keeps_actual_violation_receipts(tmp_path):
    app, requests, log = make_app(tmp_path, completion_tokens=9)
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=payload(total_token_cap=17)).status_code == 400
        assert not requests
        result = client.post("/v1/chat/completions", json=payload())
        assert result.status_code == 200
        assert result.json()["choices"][0]["finish_reason"] == "timeout"
        assert "completion_token_cap_exceeded" in result.json()["specrsi_meter"]["contract_violation"]["reasons"]
        assert len(requests) == 1
        assert json.loads(log.read_text())["response"]["usage"]["completion_tokens"] == 9


def test_absent_final_content_never_promotes_partial_reasoning_into_answer(tmp_path):
    app, _, log = make_app(tmp_path, final_content=None)
    with TestClient(app) as client:
        value = client.post("/v1/chat/completions", json=payload()).json()
    assert value["choices"][0]["message"]["content"] == ""
    assert value["choices"][0]["message"]["reasoning"] == "Compute two plus two."
    assert value["specrsi_meter"]["backend_message_content_was_null"]
    assert json.loads(log.read_text())["response"]["usage"]["completion_tokens"] == 5


def test_gateway_marks_tokenizer_disagreement_incomplete_without_invented_usage(tmp_path):
    app, _, log = make_app(tmp_path, prompt_tokens=11)
    with TestClient(app) as client:
        value = client.post("/v1/chat/completions", json=payload()).json()
        assert value["choices"][0]["finish_reason"] == "timeout"
        assert "tokenizer_mismatch" in value["specrsi_meter"]["contract_violation"]["reasons"]
    assert json.loads(log.read_text())["response"]["usage"]["prompt_tokens"] == 11


def test_late_response_retains_usage_and_backend_status_but_is_not_scored_complete(tmp_path):
    app, _, log = make_app(tmp_path)
    with TestClient(app) as client:
        value = client.post("/v1/chat/completions", json=payload(max_seconds=1e-12)).json()
    assert value["choices"][0]["finish_reason"] == "timeout"
    assert value["specrsi_meter"]["backend_finish_reason"] == "stop"
    assert value["usage"]["completion_tokens"] == 5
    assert json.loads(log.read_text())["response"] == value


def test_modified_lock_cannot_claim_publisher_verified_identity(tmp_path):
    make_app(tmp_path)
    lock = tmp_path / "lock.json"
    changed = json.loads(lock.read_text())
    changed["revision"] = "different-revision"
    lock.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="matching publisher-verified"):
        gateway.create_app(str(lock), "http://local", str(tmp_path / "other.jsonl"), tokenizer=Tokenizer())


def test_real_local_model_client_receives_actual_overcap_usage(tmp_path, monkeypatch):
    from specrsi.models.client import LocalModelClient

    app, _, _ = make_app(tmp_path, completion_tokens=9)
    with TestClient(app) as gateway_client:
        def post(url, *, json, **kwargs):
            return gateway_client.post("/v1/chat/completions", json=json)

        monkeypatch.setattr("specrsi.models.client.httpx.post", post)
        reply = LocalModelClient("http://test-pinned-gateway", "model").generate(
            [{"role": "user", "content": "2+2?"}], max_tokens=8,
            enable_thinking=True, total_token_cap=18)
    assert reply.prompt_tokens == 10
    assert reply.completion_tokens == 9  # Actual usage, exceeding the requested cap of 8.
    assert reply.finish_reason == "timeout"
    assert reply.model_revision == "pinned"


def test_sampling_is_forwarded_and_metered_end_to_end(tmp_path, monkeypatch):
    from specrsi.models.client import LocalModelClient

    app, requests, log = make_app(tmp_path)
    options = {"top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
    with TestClient(app) as gateway_client:
        monkeypatch.setattr("specrsi.models.client.httpx.get", lambda url, **kw: gateway_client.get("/health"))
        monkeypatch.setattr("specrsi.models.client.httpx.post", lambda url, *, json, **kw:
                            gateway_client.post("/v1/chat/completions", json=json))
        reply = LocalModelClient("http://test-sampling-gateway", "model").generate(
            [{"role": "user", "content": "2+2?"}], max_tokens=8, enable_thinking=True,
            total_token_cap=18, temperature=1.0, top_p=.95, seed=42, **options)
    assert reply.prompt_tokens + reply.completion_tokens == 15
    assert reply.text == "4"
    assert reply.reasoning_content == "Compute two plus two."
    assert {name: requests[0][name] for name in options} == options
    receipt = json.loads(log.read_text())
    assert {name: receipt["request"][name] for name in options} == options
    assert receipt["response"]["specrsi_meter"]["sampling_parameters"] == {
        "temperature": 1.0, "top_p": .95, "seed": 42, **options}
    assert receipt["response"]["specrsi_meter"]["gateway_source_sha256"] == gateway.GATEWAY_SOURCE_SHA256


@pytest.mark.parametrize("options", [{}, dict.fromkeys(gateway.EXTENDED_SAMPLING_PARAMETERS)])
def test_omitted_or_null_sampling_keeps_legacy_backend_payload(tmp_path, options):
    app, requests, log = make_app(tmp_path)
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=payload(**options)).status_code == 200
    assert requests == [{"model": "model", "messages": [{"role": "user", "content": "2+2?"}],
        "max_tokens": 8, "temperature": .7, "top_p": .8, "seed": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": True}}]
    receipt = json.loads(log.read_text())
    assert all(name not in receipt["request"] for name in gateway.EXTENDED_SAMPLING_PARAMETERS)
    assert receipt["response"]["specrsi_meter"]["sampling_parameters"] == {
        "temperature": .7, "top_p": .8, "seed": 0}


def test_health_capabilities_and_source_identity_are_fixed_at_startup(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path)
    expected = hashlib.sha256(Path(gateway.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(gateway, "__file__", str(tmp_path / "file_no_longer_on_disk.py"))
    with TestClient(app) as client:
        health = client.get("/health").json()
    assert health["gateway_source_sha256"] == expected
    assert set(gateway.EXTENDED_SAMPLING_PARAMETERS) <= set(health["supported_sampling_parameters"])


def test_unknown_backend_never_claims_or_silently_accepts_optional_sampling(tmp_path):
    app, requests, _ = make_app(tmp_path, engine_version="other-server")
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert not set(gateway.EXTENDED_SAMPLING_PARAMETERS) & set(health["supported_sampling_parameters"])
        assert client.post("/v1/chat/completions", json=payload(top_k=20)).status_code == 400
    assert not requests


INVALID_SAMPLING = [
    ("top_k", value) for value in (-2, 1.5, True, "20", float("nan"), float("inf"))
] + [
    (name, value) for name in ("min_p", "presence_penalty", "repetition_penalty")
    for value in (True, "0.5", float("nan"), float("inf"), -float("inf"))
] + [("min_p", -.1), ("min_p", 1.1), ("presence_penalty", -2.1),
     ("presence_penalty", 2.1), ("repetition_penalty", 0), ("repetition_penalty", -1)]


@pytest.mark.parametrize("name,value", INVALID_SAMPLING)
def test_bad_sampling_is_rejected_before_inference(tmp_path, name, value):
    app, requests, _ = make_app(tmp_path)
    with TestClient(app) as client:
        # Include nonstandard raw JSON nonfinite literals: they must still fail
        # validation without a backend request or an error serializing a 422.
        result = client.post("/v1/chat/completions", content=json.dumps(payload(**{name: value})),
                             headers={"content-type": "application/json"})
        assert result.status_code == 422
    assert not requests


@pytest.mark.parametrize("options", [
    {"top_k": -1, "min_p": 0, "presence_penalty": -2, "repetition_penalty": .01},
    {"top_k": 0, "min_p": 1, "presence_penalty": 2, "repetition_penalty": 10},
    {"top_k": 1, "min_p": .5, "presence_penalty": 0, "repetition_penalty": 1},
])
def test_sampling_valid_boundaries_are_preserved(tmp_path, options):
    app, requests, _ = make_app(tmp_path)
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=payload(**options)).status_code == 200
    assert {name: requests[0][name] for name in options} == options
