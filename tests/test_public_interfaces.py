"""Caller-owned configuration and portable local interfaces."""
import pytest

from specrsi.benchmarks.data import assign_split
from specrsi.cli import doctor
from specrsi.models import LocalModelClient, ModelReply


def test_doctor_does_not_contact_a_default_service(monkeypatch, tmp_path):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("A model service must be explicitly selected")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LocalModelClient, "health", unexpected_call)
    report = doctor()
    assert {"python", "packages", "model", "models", "datasets", "reproduction", "ready_for"} <= report.keys()
    assert report["model"] == {"status": "not_requested"}
    assert report["models"] == {}
    assert report["ready_for"]["local_qwen_inference"] is False


def test_doctor_uses_reference_model_for_an_explicit_endpoint(monkeypatch, tmp_path):
    calls = []

    def health(client):
        calls.append((client.base_url, client.model))
        return {"status": "ok"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LocalModelClient, "health", health)
    report = doctor(base_url="http://example.invalid")
    assert calls == [("http://example.invalid", "Qwen3.5-35B-A3B")]
    assert report["models"]["Qwen3.5-35B-A3B"] == report["model"] == {"status": "ok"}
    assert report["ready_for"]["local_qwen_inference"] is True


@pytest.mark.parametrize("content_hash, expected", [
    ("fixture", "development"),
    ("fixture-3", "selection"),
    ("fixture-1", "audit"),
])
def test_split_assignment_uses_the_registered_default_rule(content_hash, expected):
    assert assign_split(content_hash) == expected
    assert assign_split(content_hash, seed=20260916) == expected


def test_split_assignment_supports_an_explicit_seed():
    assert assign_split("fixture-1", seed=1) == "selection"
    assert assign_split("fixture-1") == "audit"


def test_model_reply_preserves_positive_gpu_allocation():
    reply = ModelReply("answer", 1, 1, 0.1, "stop", "fixture-revision", "request", gpu_count=2)
    assert reply.gpu_count == 2


@pytest.mark.parametrize("gpu_count", [0, -1])
def test_model_reply_rejects_nonpositive_gpu_allocation(gpu_count):
    with pytest.raises(ValueError, match="GPU"):
        ModelReply("answer", 1, 1, 0.1, "stop", "fixture-revision", "request", gpu_count=gpu_count)
