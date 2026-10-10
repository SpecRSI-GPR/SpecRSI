"""Caller-owned configuration and portable local interfaces."""
import pytest

from specrsi.benchmarks.data import assign_split
from specrsi.cli import doctor
from specrsi.models import LocalModelClient, ModelReply
from specrsi.paper_profile import algorithm_metadata


def test_doctor_does_not_contact_a_default_service(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("A model service must be explicitly selected")

    monkeypatch.setattr(LocalModelClient, "health", unexpected_call)
    assert set(doctor()) == {"python", "packages"}
    with pytest.raises(ValueError, match="both"):
        doctor(base_url="http://example.invalid")


def test_algorithm_metadata_requires_no_external_document():
    assert algorithm_metadata({"search": {"algorithm": "vats_pdf"}}) == {"algorithm": "vats_pdf"}
    with pytest.raises(ValueError, match="Unknown"):
        algorithm_metadata({"search": {"algorithm": "unknown"}})


def test_split_fractions_are_caller_controlled():
    assert assign_split("fixture", 1, development_fraction=1, selection_fraction=0) == "development"
    assert assign_split("fixture", 1, development_fraction=0, selection_fraction=1) == "selection"
    assert assign_split("fixture", 1, development_fraction=0, selection_fraction=0) == "audit"
    with pytest.raises(ValueError, match="fractions"):
        assign_split("fixture", 1, development_fraction=1, selection_fraction=1)


def test_cpu_reply_preserves_zero_gpu_allocation():
    reply = ModelReply("answer", 1, 1, 0.1, "stop", "fixture-revision", "request", gpu_count=0)
    assert reply.gpu_count == 0
    with pytest.raises(ValueError, match="GPU"):
        ModelReply("answer", 1, 1, 0.1, "stop", "fixture-revision", "request", gpu_count=-1)
