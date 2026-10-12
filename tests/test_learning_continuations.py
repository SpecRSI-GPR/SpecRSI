"""Real collector/engine paths with explicit fake model I/O, never benchmark claims."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from specrsi.core import Candidate, EvaluationScope, Outcome, Task
from specrsi.core.execution import ExecutionResult, PureIntegerExecutor
from specrsi.core.search import DraftResult
from specrsi.learning.continuations import ForcedCollectionConfig, FrozenRoleGuard, collect_paid_continuations
from test_paper_integration import FakeRoles, fixture

torch.set_num_threads(2)


def prompt_config(tmp_path, monkeypatch):
    config, seen = fixture(tmp_path, monkeypatch, learned=True)
    executor = {**config["model"], "base_url": "http://127.0.0.1:18080"}
    collection = {"algorithm": "vats_pdf", "dataset": "hotpotqa", "contexts": 1, "tasks_per_context": 1,
        "group_size": 2, "max_actions": 4, "max_candidates": 2, "budget_tokens": 4000,
        "execution_cap": 16, "draft_cap": 16, "initial_prompt": "initial", "exploration": .1,
        "executor": executor, "vats": config["search"]["vats"], "forced_repetitions": 2,
        "forced_state_indices": [0, 1]}
    return {"collection": collection, "output": str(tmp_path / "collection"),
            "training": {"seed": 17, "cost_coefficient": .001}}, seen


def test_prompt_collector_real_paid_forks_have_complete_charges_and_critic_only_labels(tmp_path, monkeypatch):
    from specrsi.learning.bridge import collect_prompt_specrsi
    config, seen = prompt_config(tmp_path, monkeypatch)
    result = collect_prompt_specrsi(FakeRoles(), 1, config)
    assert result["pairs"] and len(result["forced_trajectories"]) >= 4
    assert all(t.forced for t in result["forced_trajectories"])
    assert all(not t.forced for group in result["groups"] for t in group)
    for pair in result["pairs"]:
        assert pair.left_count == pair.right_count == 2
        assert len(set(pair.left_seeds + pair.right_seeds)) == 4
        assert pair.left.context_text == pair.right.context_text
    meta = result["metadata"]
    assert meta["forced_search_cost"] == sum(t.total_cost for t in result["forced_trajectories"])
    assert meta["total_acquisition_tokens"] == meta["initial_acquisition_tokens"] + meta["search_cost"] + meta["forced_search_cost"]
    journal = [json.loads(line) for line in (tmp_path / "collection/forced_acquisition_round_1.jsonl").read_text().splitlines()]
    completed = [event for event in journal if event["event"] == "completed"]
    assert len(completed) == len(result["forced_trajectories"])
    assert all(not event["actor_eligible"] for event in completed)
    assert all(event["counterfactual_budget_debit"] == 0 for event in journal if event["event"] == "clone_acquisition")
    assert {messages[-1]["content"] for messages in seen} == {"development question"}
    assert any(t.total_cost == 0 for t in result["forced_trajectories"])  # Forced ACCEPT does not repay its menu.


@pytest.mark.parametrize("change", [{"task_pipeline": "tools"}, {"initial_workflow": {}},
    {"executor": {"base_url": "http://unrestorable-remote.example"}}])
def test_prompt_forks_reject_missing_restore_adapter_before_any_task_call(tmp_path, monkeypatch, change):
    from specrsi.learning.bridge import collect_prompt_specrsi
    config, seen = prompt_config(tmp_path, monkeypatch)
    if "executor" in change:
        config["collection"]["executor"].update(change["executor"])
    else:
        config["collection"].update(change)
    with pytest.raises(ValueError, match="adapter|stateless"):
        collect_prompt_specrsi(FakeRoles(), 0, config)
    assert not seen


def test_native_science_collector_forks_fresh_registered_executor_instances(tmp_path, monkeypatch):
    import specrsi.learning.science_bridge as collector
    import specrsi.science.executor as native
    instances = []

    def make_scope(domain, split, cases, **kwargs):
        return EvaluationScope("native-dev", tuple(Task(f"case-{case}") for case in cases), "native-test", "native-test",
            scored_fields=("native_score",), aggregation_id="science_native_weighted_mean_v1", split="dev")

    class Executor:
        def __init__(self):
            self.calls = []
            instances.append(self)

        def run(self, candidate, task, scope, cap, resume_token=None):
            self.calls.append(candidate.uid)
            return ExecutionResult("complete", 3, Outcome(.5, {"native_score": 12. if candidate.name == "improved" else 10.}),
                receipt={"wall_seconds": .003, "native_result": {"complete": True}})

    class Drafter:
        def __init__(self, bridge, **kwargs):
            pass

        def draft(self, parent, state, cap, rng):
            return DraftResult(Candidate("improved", {"value": 12.}, parent_id=parent.uid), 2)

    monkeypatch.setattr(collector, "make_scope", make_scope)
    monkeypatch.setattr(collector, "make_candidate", lambda domain: Candidate("initial", {"value": 10.}))
    monkeypatch.setattr(collector, "ScienceExecutor", Executor)
    monkeypatch.setattr(native, "ScienceExecutor", Executor)
    monkeypatch.setattr(collector, "HFParameterDrafter", Drafter)
    config = {"output": str(tmp_path / "science"), "training": {"seed": 17, "cost_coefficient": .001},
        "collection": {"algorithm": "vats_pdf", "domain": "reactor", "case_ids": [0], "group_size": 2,
        "execution_cap_ms": 20, "draft_cap_ms": 10, "budget_ms": 2000, "head_cap_ms": 100,
        "max_actions": 4, "max_candidates": 2, "forced_repetitions": 2,
        "vats": {"max_queries": 8, "max_extensions": 0, "max_draft_depth": 1, "max_draft_parents": 1,
            "max_completion_targets": 1, "max_pair_targets": 0, "max_singletons": 1, "max_menu_options": 4,
            "solver_call_cost": 0, "policy_call_cost": 0, "scorer_query_cap": 100}}}
    result = collector.collect_science_specrsi(FakeRoles(), 0, config)
    assert len(result["forced_trajectories"]) == 4 and len(result["pairs"]) == 1
    assert len(instances) == 1 + 2 + 4  # Initial acquisition, ordinary members, independent suffixes.
    metadata = result["metadata"]
    assert metadata["forced_search_cost"] == sum(t.total_cost for t in result["forced_trajectories"])
    assert metadata["total_acquisition_ms"] == metadata["initial_acquisition_ms"] + metadata["search_cost"] + sum(
        arm["total_acquisition_cost"] for arm in metadata["forced_acquisition"])
    assert all(not t.audit_only and t.forced for t in result["forced_trajectories"])
    json.dumps(metadata)  # Native nested receipts remain serializable acquisition evidence.


def test_frozen_role_guard_rejects_an_optimizer_update_during_collection():
    net = torch.nn.Linear(2, 2)
    guard = FrozenRoleGuard(net)
    guard.check()
    with torch.no_grad():
        net.weight.add_(1.)
    with pytest.raises(RuntimeError, match="roles changed"):
        guard.check()


def test_prospective_forced_configuration_rejects_teacher_mixing_and_duplicate_states():
    with pytest.raises(ValueError, match="warm-up"):
        ForcedCollectionConfig.from_settings({"forced_repetitions": 1}, teacher=True)
    with pytest.raises(ValueError, match="distinct"):
        ForcedCollectionConfig.from_settings({"forced_state_indices": [0, 0]})


def paid_fixture():
    from specrsi.core import BudgetLedger, ConservativeChecker, Ledger
    from specrsi.core.vats import VATSConfig, VATSEngine
    from specrsi.learning.heads import TinyRoleModel, TinyModelConfig
    from specrsi.learning.paper_bridge import VATSSearchBridge, ReferenceVATSIntegerDrafter
    model = TinyRoleModel(TinyModelConfig(option_dim=12, hidden_dim=16))
    bridge = VATSSearchBridge(model, snapshot_id="registered-roles")
    tasks = (Task("a", {"route": "a", "x": 2}, 3), Task("b", {"route": "b", "x": 2}, 4))
    scope = EvaluationScope("development", tasks, "pure-int-v1", "integer-equality-v1", split="dev")
    parent = Candidate("initial", {"routes": {"a": ["x"], "b": ["x"]}},
        {"contract": "pure-int-v1"}, "pure_integer", True)
    ledger, executor = Ledger(), PureIntegerExecutor()
    for task in tasks:
        record = executor.run(parent, task, scope, 10)
        ledger.admit_direct(parent, task, scope, record.outcome, record.receipt)
    cfg = VATSConfig(execution_cap=10, draft_cap=4, max_candidates=2, max_actions=3,
        max_draft_depth=1, max_queries=0, planning_cost=0, policy_call_cost=0, solver_call_cost=0,
        scorer_snapshot=bridge.snapshot_id, policy_snapshot=bridge.snapshot_id, max_sources=0)
    def policy(state, plans):
        return [20. if plan.kind == ("draft" if len(state.candidates) == 1 else "verify") else -10. for plan in plans]
    engine = VATSEngine(scope, parent, ledger, BudgetLedger(100), executor, ReferenceVATSIntegerDrafter(bridge),
        cfg, policy_menu=policy, capture_paid_states=True)
    engine.run()
    return engine.paid_snapshots[1], model, bridge


def test_missing_forced_rollout_keeps_failure_journal_without_fake_zero_label(tmp_path, monkeypatch):
    from specrsi.core import ConservativeChecker
    from specrsi.learning.paper_bridge import VATSSearchBridge, ReferenceVATSIntegerDrafter
    snapshot, model, bridge = paid_fixture()
    original = snapshot.uid
    def fail(*args, **kwargs):
        raise RuntimeError("simulator transport died after launch")
    monkeypatch.setattr(PureIntegerExecutor, "run", fail)
    path = tmp_path / "acquisition.jsonl"
    with pytest.raises(RuntimeError, match="transport died"):
        collect_paid_continuations([snapshot], model=model, snapshot_id=bridge.snapshot_id,
            bridge_factory=lambda: VATSSearchBridge(model, snapshot_id=bridge.snapshot_id),
            executor_factory=PureIntegerExecutor, drafter_factory=lambda b, e: ReferenceVATSIntegerDrafter(b),
            checker_factory=ConservativeChecker, environment_contract="pure_integer",
            config=ForcedCollectionConfig(repetitions=2, arm_rule="non_accept_first"), seed=4,
            lineage_id="development-root", training={}, acquisition_path=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    failure = rows[-1]
    assert failure["event"] == "missing_rollout" and failure["return_label"] is None
    assert failure["executor_events"][0]["cost"] is None
    assert not any(row["event"] == "completed" for row in rows)
    assert snapshot.uid == original


def test_pdf_context_contains_actual_partial_failure_feedback_without_expected_answers():
    snapshot, model, bridge = paid_fixture()
    text = json.loads(bridge.context_text(snapshot.state, bounded=False))
    assert text["resources"]["remaining_attempts"] >= 0
    assert text["observed_outcomes"] and text["observed_outcomes"][0]["score"] == 0.
    assert any(candidate["complete_score"] is None and candidate["missing"] for candidate in text["candidates"])
    assert all("expected" not in row and "reference" not in row for row in text["task_inputs"])
    assert "expected" not in bridge.context_text(snapshot.state, bounded=False)


def test_role_snapshot_hash_changes_when_weights_change_even_with_same_epoch_label():
    from specrsi.learning.continuations import role_snapshot_id
    from specrsi.learning.heads import TinyRoleModel, TinyModelConfig
    model = TinyRoleModel(TinyModelConfig(option_dim=12, hidden_dim=16))
    original = role_snapshot_id(model, "round-0")
    assert original == role_snapshot_id(model, "round-0")
    with torch.no_grad():
        next(model.policy_head.parameters()).add_(.1)
    assert original != role_snapshot_id(model, "round-0")


def test_completed_unsuccessful_forced_rollouts_keep_negative_cost_returns(monkeypatch):
    from specrsi.core import ConservativeChecker
    from specrsi.learning.paper_bridge import VATSSearchBridge, ReferenceVATSIntegerDrafter
    snapshot, model, bridge = paid_fixture()
    monkeypatch.setattr(PureIntegerExecutor, "run", lambda *args, **kwargs:
        ExecutionResult("error", 1, error="bounded evaluator failed without a complete outcome"))
    result = collect_paid_continuations([snapshot], model=model, snapshot_id=bridge.snapshot_id,
        bridge_factory=lambda: VATSSearchBridge(model, snapshot_id=bridge.snapshot_id),
        executor_factory=PureIntegerExecutor, drafter_factory=lambda b, e: ReferenceVATSIntegerDrafter(b),
        checker_factory=ConservativeChecker, environment_contract="pure_integer",
        config=ForcedCollectionConfig(repetitions=2, arm_rule="non_accept_first"), seed=4,
        lineage_id="development-root", training={"cost_coefficient": .1})
    assert len(result["forced_trajectories"]) == 4
    pair = result["pairs"][0]
    assert pair.lineage_id == "development-root"
    assert len(pair.left_returns) == len(pair.right_returns) == 2
    assert all(value < 0 for value in pair.left_returns)
    assert all(t.terminated for t in result["forced_trajectories"])
