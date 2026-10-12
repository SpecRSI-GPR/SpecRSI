import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/gepa/src"))
pytestmark = pytest.mark.skipif(importlib.util.find_spec("gepa") is None, reason="pinned GEPA checkout unavailable")

from specrsi.baselines.gepa import LocalGEPAAdapter, MeteredLocalModel, run_gepa
from specrsi.benchmarks.base import BenchmarkExample
from specrsi.models.client import ModelReply


class FakeModel:
    def generate(self, messages, **kwargs):
        if len(messages) == 1:
            answer = "```\nUse exact arithmetic and return the valid expression.\n```"
        else:
            answer = "(1+3)*(2+4)" if "exact arithmetic" in messages[0]["content"] else "0"
        return ModelReply(answer, 10, 6, .02, "stop", "test-revision", "test-request")


def example(split):
    return BenchmarkExample(split, "game24", split, f"Use 1 2 3 4, question {split}", [1, 2, 3, 4])


def test_gepa_rejects_audit_in_optimizer_and_selection_in_reflection():
    model = MeteredLocalModel(FakeModel())
    adapter = LocalGEPAAdapter(model)
    with pytest.raises(ValueError, match="Audit"):
        adapter.evaluate([example("audit")], {"system_prompt": "seed"})
    batch = adapter.evaluate([example("selection")], {"system_prompt": "seed"}, capture_traces=True)
    with pytest.raises(ValueError, match="development"):
        adapter.make_reflective_dataset({"system_prompt": "seed"}, batch, ["system_prompt"])


def test_real_gepa_loop_fake_model_improves_and_audits_frozen_version(tmp_path):
    model = MeteredLocalModel(FakeModel())
    result = run_gepa(development=[example("development")], selection=[example("selection")],
                      audit=[example("audit")], model=model, output_dir=tmp_path / "run",
                      seed_prompt="seed", max_metric_calls=4)
    assert result["status"] == "complete"
    assert result["candidate_count"] >= 2
    assert result["selection_score"] == result["audit_score"] == 1
    assert model.records[-1]["role"] == "audit"
    assert (tmp_path / "run/selected_candidate.json").is_file()
    assert all(m["prompt_tokens"] == m["calls"] * 10 for m in result["meter"].values())


def test_model_failure_keeps_attempt_record(tmp_path):
    class FailingModel:
        def generate(self, *args, **kwargs):
            raise RuntimeError("service unavailable")
    model = MeteredLocalModel(FailingModel(), journal=tmp_path / "attempts.jsonl")
    with pytest.raises(RuntimeError):
        model.call("input", role="reflection")
    assert model.summary()["reflection"]["unmetered_failed_calls"] == 1
    assert model.records[0]["status"] == "error"
