"""A crashed real collector retains measured role calls and unknown debits."""
import json

import pytest

from test_learning_continuations import prompt_config
from test_paper_integration import FakeRoles


@pytest.mark.parametrize("unknown_state", ["pending", "conservative_bound"])
def test_failed_search_retains_each_known_role_and_executor_receipt_once(tmp_path, monkeypatch, unknown_state):
    from specrsi.core.vats import VATSEngine
    from specrsi.learning.bridge import collect_prompt_specrsi
    config, seen = prompt_config(tmp_path, monkeypatch)
    config["collection"]["forced_repetitions"] = 0
    engines = []

    def fail_after_measured_work(engine):
        engines.append(engine)
        # Distinct physical invocations may have identical content/usage: retain
        # both rather than deduplicating by body or accidentally merging twice.
        bridge = engine.drafter.bridge
        bridge.physical_inferences.extend([
            {"kind": "learned_control", "role": "policy", "prompt_tokens": 7, "completion_tokens": 0},
            {"kind": "learned_control", "role": "policy", "prompt_tokens": 7, "completion_tokens": 0}])
        engine.executor.physical_records.append({"kind": "task", "request_id": "known-task", "tokens": 6})
        engine.state.budget.charge("completed-role-1", 7, "policy")
        engine.state.budget.charge("completed-role-2", 7, "policy")
        engine.state.budget.reserve("interrupted-role", 19, "draft")
        if unknown_state == "conservative_bound":
            engine.state.budget.settle_unknown("interrupted-role")
        raise RuntimeError("synthetic role OOM after two measured calls")

    monkeypatch.setattr(VATSEngine, "run", fail_after_measured_work)
    with pytest.raises(RuntimeError, match="role OOM"):
        collect_prompt_specrsi(FakeRoles(), 0, config)
    report = json.loads((tmp_path / "collection/collection_progress_round_0.json").read_text())
    assert not report["valid_training_batch"] and report["status"] == "failed"
    records = report["physical_executions"]
    assert len(records) == 4  # One acquired root, one task receipt, two role receipts.
    assert len([r for r in records if r.get("role") == "policy"]) == 2
    assert len([r for r in records if r.get("request_id") == "known-task"]) == 1
    assert report["failed_search_physical_accounting"]["role_receipts"] == 2
    assert not report["failed_search_physical_accounting"]["complete"]
    budget = report["failed_search_budget"]
    if unknown_state == "pending":
        assert budget["pending_reservations"] == {"interrupted-role": 19}
        assert report["failed_search_settled_tokens"] == 14
    else:
        assert not budget["pending_reservations"] and not budget["cost_valid"]
        assert budget["charges"][-1]["cost_kind"] == "conservative_bound"
        assert report["failed_search_settled_tokens"] == 33
    assert len(engines) == 1 and len(seen) == 1  # No second group member or implicit retry.
    contexts = report['development_contexts']
    assert len(contexts) == 1 and contexts[0]['examples'][0]['example_id'] == 'development'
    assert set(contexts[0]['examples'][0]) == {'example_id', 'content_hash'}
