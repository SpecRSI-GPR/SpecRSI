"""Known backend usage must survive gateway errors into the campaign journal."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("fastapi")
import httpx
from fastapi.testclient import TestClient

from specrsi.answer_strategies import AnswerStrategy
from specrsi.benchmarks import BenchmarkExample
from specrsi.experiments import QwenExecutor
from specrsi.models import LocalModelClient


def import_script(name):
    spec = importlib.util.spec_from_file_location("accounting_review_" + name,
        Path(__file__).parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gateway = import_script("serve_budget_gateway")
campaign = import_script("run_performance_campaign")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [0] * 10


def setup_gateway(tmp_path, completion_tokens):
    manifest_path, lock_path = tmp_path / "download.json", tmp_path / "model.json"
    lock_path.write_text(json.dumps({"download_manifest": str(manifest_path), "revision": "pinned",
        "repo_id": "repo/model", "served_model_name": "model", "context_length": 32768,
        "gpu_ids": [1], "format": "fp8", "engine_version": "test", "local_path": "unused"}))
    manifest_path.write_text(json.dumps({"status": "publisher_verified_complete", "revision": "pinned",
        "repo_id": "repo/model", "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()}))
    backend_calls = []
    def backend(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        backend_calls.append(json.loads(request.content))
        return httpx.Response(200, json={"id": f"actual-request-{len(backend_calls)}",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "4"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens,
                      "total_tokens": 10 + completion_tokens}})
    gateway_log = tmp_path / "gateway.jsonl"
    app = gateway.create_app(str(lock_path), "http://local-worker", str(gateway_log),
        tokenizer=Tokenizer(), transport=httpx.MockTransport(backend))
    return app, lock_path, gateway_log, backend_calls


def route_local_client(monkeypatch, transport_client):
    def post(url, *, json, **kwargs):
        return transport_client.post("/v1/chat/completions", json=json)
    monkeypatch.setattr("specrsi.models.client.httpx.post", post)
    return LocalModelClient("http://gateway-accounting-unit-test", "model")


def campaign_fixture(tmp_path, lock_path, *, max_seconds=120):
    strategy = AnswerStrategy(reasoning_tokens=8, answer_tokens=4).to_dict()
    config = {"schema": "inference_performance_v1", "output": str(tmp_path / "campaign"),
        "model_lock": str(lock_path), "dataset": "hotpotqa", "sample_seed": 4, "decode_seed": 7,
        "counts": {"development": 1, "selection": 1, "audit": 1}, "execution_cap": 40,
        "total_token_budget": 400, "baseline": "strategy", "finalists": 1,
        "candidates": [{"name": "strategy", "artifacts": {"system_prompt": "Solve correctly.",
            "answer_strategy": strategy}, "generation": {"max_tokens": 8, "max_seconds": max_seconds}}]}
    examples = {split: [BenchmarkExample(split, "hotpotqa", split, "Public question " + split, "4")]
                for split in config["counts"]}
    return config, examples, Path(config["output"])


def test_gateway_cap_violation_reaches_local_client_with_exact_usage(tmp_path, monkeypatch):
    app, _, log, calls = setup_gateway(tmp_path, completion_tokens=9)
    with TestClient(app) as transport_client:
        client = route_local_client(monkeypatch, transport_client)
        reply = client.generate([{"role": "user", "content": "2+2?"}], max_tokens=8, total_token_cap=18)
    assert reply.finish_reason == "timeout"
    assert reply.prompt_tokens == 10 and reply.completion_tokens == 9
    assert len(calls) == 1
    raw = json.loads(log.read_text())["response"]
    assert raw["specrsi_meter"]["contract_violation"]
    assert raw["specrsi_meter"]["backend_finish_reason"] == "stop"


def test_strategy_cap_violation_preserves_exact_cost_and_blocks_campaign_retry(tmp_path, monkeypatch):
    app, lock_path, log, calls = setup_gateway(tmp_path, completion_tokens=9)
    config, examples, output = campaign_fixture(tmp_path, lock_path)
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"test-source": "frozen"})
    with TestClient(app) as transport_client:
        client = route_local_client(monkeypatch, transport_client)
        executor = QwenExecutor(client, Tokenizer(), {})
        with pytest.raises(ValueError, match="exceeded its enforced reservation"):
            campaign.run(config, executor=executor, loaded=examples, stop_after="development")
        physical = [json.loads(line) for line in (output / "physical_calls.jsonl").read_text().splitlines()]
        assert len(physical) == 1
        assert physical[0]["accounted_tokens"] == 19
        assert physical[0]["reserved_tokens"] == 18
        assert physical[0]["metering_status"] == "server_measured"
        assert (output / "pending_request.json").exists()
        assert not (output / "results.jsonl").exists()
        with pytest.raises(RuntimeError, match="no free retry"):
            campaign.run(config, executor=executor, loaded=examples, stop_after="development")
    assert len(calls) == 1
    assert json.loads(log.read_text())["response"]["usage"]["total_tokens"] == 19


def test_real_timeout_with_valid_usage_is_charged_once_and_retains_zero_denominator_score(tmp_path, monkeypatch):
    app, lock_path, _, calls = setup_gateway(tmp_path, completion_tokens=5)
    config, examples, output = campaign_fixture(tmp_path, lock_path, max_seconds=1e-12)
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"test-source": "frozen"})
    with TestClient(app) as transport_client:
        client = route_local_client(monkeypatch, transport_client)
        executor = QwenExecutor(client, Tokenizer(), {})
        summary = campaign.run(config, executor=executor, loaded=examples, stop_after="development")
        assert summary["strategy"]["count"] == 1 and summary["strategy"]["complete"] == 0
        assert summary["strategy"]["charged_tokens"] == summary["strategy"]["measured_tokens"] == 15
        assert summary["strategy"]["reserved_unmetered_tokens"] == 0
        assert summary["strategy"]["mean_score"] == 0
        assert not (output / "pending_request.json").exists()
        replay = campaign.run(config, executor=executor, loaded=examples, stop_after="development")
        assert replay == summary
    assert len(calls) == 1
