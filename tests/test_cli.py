"""Environment inspection does not implicitly invoke local model services."""
from specrsi.cli import doctor


def test_doctor_checks_services_only_for_explicit_endpoints(tmp_path, monkeypatch):
    calls = []

    class Client:
        def __init__(self, endpoint, model):
            calls.append((endpoint, model))

        def health(self):
            return {"status": "ok"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("specrsi.models.LocalModelClient", Client)
    result = doctor()
    assert not calls
    assert result["model"]["status"] == "not_requested"
    assert not any(result["ready_for"].values())
    assert all(not item["present"] for item in result["reproduction"].values())

    result = doctor(base_url="http://reference.invalid", kimi_base_url="http://kimi.invalid")
    assert calls == [("http://reference.invalid", "Qwen3.5-35B-A3B"),
                     ("http://kimi.invalid", "Kimi-K2.6")]
    assert result["ready_for"]["local_qwen_inference"]
    assert result["ready_for"]["local_kimi_inference"]


def test_doctor_retains_measured_service_failure(tmp_path, monkeypatch):
    class Client:
        def __init__(self, endpoint, model):
            pass

        def health(self):
            raise RuntimeError("endpoint unavailable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("specrsi.models.LocalModelClient", Client)
    result = doctor(base_url="http://reference.invalid")
    assert result["model"] == {"status": "unavailable", "reason": "endpoint unavailable"}
    assert not result["ready_for"]["local_qwen_inference"]
