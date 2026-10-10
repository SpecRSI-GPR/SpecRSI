from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from specrsi.core.acceptance import accept
from specrsi.core.ledger import Ledger
from specrsi.core.models import Candidate, EvaluationScope, Outcome, Task
from specrsi.learning.demonstrations import DemonstrationQuality, DemonstrationVATSBridge, warmup_records
from specrsi.learning.heads import TinyModelConfig, TinyRoleModel
from specrsi.learning.paper_bridge import VATSSearchBridge, collect_reference_vats
from specrsi.learning.records import DecisionRecord, SearchTrajectory, TransitionRecord


@pytest.mark.parametrize("rule,gain,eligible", [
    ("verified", -.4, True), ("parent_nondegrading", -.4, False),
    ("parent_nondegrading", 0., True), ("parent_improving", 0., False),
    ("parent_improving", .2, True),
])
def test_registered_edit_quality_uses_observed_parent_gain(rule, gain, eligible):
    quality = DemonstrationQuality(edit_rule=rule)
    assert quality.decision(verified=True, parent_verified=True, changed=True,
        child_score=.5 + gain, parent_score=.5)[0] is eligible
    assert not quality.decision(verified=False, parent_verified=True, changed=True,
        child_score=None, parent_score=.5)[0]
    if rule != "verified":
        assert not quality.decision(verified=True, parent_verified=False, changed=True,
            child_score=.5, parent_score=None)[0]


def test_quality_contract_rejects_silent_settings_and_noop():
    for settings in [{"typo": True}, {"min_gain": float("nan")}, {"min_gain": -.1},
                     {"min_gain": .1}, {"plan_rule": "predicted"}, {"require_changed_artifacts": 1}]:
        with pytest.raises(ValueError):
            DemonstrationQuality.from_settings(settings)
    rule = DemonstrationQuality(require_changed_artifacts=True)
    assert rule.decision(verified=True, parent_verified=True, changed=False,
        child_score=1., parent_score=0.) == (False, "unchanged_artifacts")


def _conversion_fixture(monkeypatch, quality, child_raw_score, *, child_penalty=0., complete=True):
    model = TinyRoleModel(TinyModelConfig(option_dim=12, feature_dim=8, hidden_dim=8))
    bridge = DemonstrationVATSBridge(model, demonstration_quality=quality)
    parent = Candidate("parent", {"instruction": "old"}, operating_penalty=.1)
    child = Candidate("child", {"instruction": "changed"}, parent_id=parent.uid,
                      operating_penalty=child_penalty)
    task = Task("task", {"visible": "context"}, expected="must not enter selection")
    scope = EvaluationScope("development", (task,), "fixture-v1", "observed-v1", split="train")
    ledger = Ledger()
    ledger.admit_direct(parent, task, scope, Outcome(score=.6, fields={"answer": "parent"}, admissible=True), {"done": True})
    ledger.register(child, scope)
    if complete:
        ledger.admit_direct(child, task, scope, Outcome(score=child_raw_score, fields={"answer": "child"}, admissible=True), {"done": True})
    bridge._quality_context = (ledger, scope)
    token = DecisionRecord("edit_token", ("3", "4"), 0, math.log(.5), input_ids=(1, 2), transition_index=0)
    sample = SimpleNamespace(prompt_ids=(1, 2), generated_ids=(3,), decisions=(token,), stop_reason="eos")
    bridge.record_executed_edit(child, parent, sample)
    decisions = (DecisionRecord("plan", ("ACCEPT", "draft:parent"), 1, None,
                                transition_index=0, exploration=0., deterministic=True), token,
                 DecisionRecord("plan", ("ACCEPT", "verify:child"), 1, None,
                                transition_index=1, exploration=0., deterministic=True),
                 DecisionRecord("plan", ("ACCEPT",), 0, None, transition_index=2, exploration=0., deterministic=True))
    bridge.teacher_choice_evidence = [{"transition_index": d.transition_index,
        "support": list(d.support), "selected_id": d.support[d.selected_index]}
        for d in decisions if d.kind == "plan"]
    transitions = (TransitionRecord(0., 0., 4.), TransitionRecord(0., 0., 7.), TransitionRecord(0., 0., 0.))
    trajectory = SearchTrajectory("group", "root", "search", 17, transitions, decisions, lineage_id="root")
    monkeypatch.setattr(VATSSearchBridge, "convert", lambda self, result, lineage_id: trajectory)
    result = SimpleNamespace(ledger_json=ledger.to_json(), acceptance=accept(parent, (child,), ledger, scope),
        transitions=(SimpleNamespace(events=({"event": "freeze", "candidate_id": child.uid},)),
            SimpleNamespace(events=({"event": "acquisition", "candidate_id": child.uid},)),
            SimpleNamespace(events=({"event": "accept"},))))
    actual = bridge.convert(result, lineage_id="root")
    collection = {"groups": [[actual]], "edits": bridge.executed_edits,
                  "metadata": {"sft_plan_evidence": bridge.plan_evidence}}
    return bridge, trajectory, actual, collection


@pytest.mark.parametrize("complete,score,eligible", [(True, .3, False), (True, .5, True), (False, 1., False)])
def test_rejected_edits_keep_all_critic_targets_costs_and_no_actor(monkeypatch, complete, score, eligible):
    quality = {"edit_rule": "parent_nondegrading", "require_changed_artifacts": True,
               "plan_rule": "qualified_edits"}
    bridge, original, actual, collection = _conversion_fixture(monkeypatch, quality, score, complete=complete)
    assert bool(bridge.executed_edits) is eligible
    assert actual.transitions == original.transitions
    assert actual.total_cost == original.total_cost == 11.
    assert actual.full_return(.01) == original.full_return(.01)
    assert actual.demonstration
    assert all(d.deterministic and d.behavior_logprob is None for d in actual.decisions if d.kind == "plan")
    assert [actual.target(d, .01) for d in actual.decisions] == [original.target(d, .01) for d in original.decisions]
    records = warmup_records(collection)
    assert sum(r["record_type"] == "trajectory" for r in records) == 1
    assert sum(r["record_type"] == "plan" for r in records) == (3 if eligible else 1)
    assert sum(r["record_type"] == "edit" for r in records) == int(eligible)
    evidence = bridge.edit_evidence[0]
    assert evidence["parent_utility"] == .5  # Includes .1 operating penalty once.
    assert evidence["sft_eligible"] is eligible
    assert "expected" not in str(evidence)


def test_native_utility_penalty_is_not_subtracted_twice(monkeypatch):
    bridge, _, _, _ = _conversion_fixture(monkeypatch, {"edit_rule": "parent_nondegrading"},
                                           .7, child_penalty=.2)
    # Registered epsilon treats roundoff from .7 - .2 as equality to .6 - .1.
    assert bridge.edit_evidence[0]["child_utility"] == pytest.approx(.5)
    assert bridge.edit_evidence[0]["observed_parent_gain"] == pytest.approx(0.)
    assert bridge.edit_evidence[0]["child_utility"] != pytest.approx(.3)
    assert bridge.edit_evidence[0]["sft_eligible"]


def test_plan_filter_evidence_cannot_silently_refer_to_another_action(monkeypatch):
    bridge, _, _, collection = _conversion_fixture(monkeypatch, {}, .7)
    bridge.plan_evidence[0]["selected_id"] = "ACCEPT"
    with pytest.raises(ValueError, match="actual executed decision"):
        warmup_records(collection)


def test_reference_collection_applies_filter_and_keeps_complete_critic_traces():
    torch.manual_seed(2)
    model = TinyRoleModel(TinyModelConfig(option_dim=12, feature_dim=8, hidden_dim=8))
    config = {"stage": "warmup", "training": {"seed": 2}, "collection": {
        "algorithm": "vats_pdf", "contexts": 1, "group_size": 2, "budget": 500,
        "max_candidates": 3, "max_actions": 8, "exploration": 0,
        "demonstration_policy": "complete_then_draft", "vats": {"max_queries": 0},
        "demonstration_quality": {"edit_rule": "parent_improving", "min_gain": 10.,
                                   "plan_rule": "qualified_edits"}}}
    collected = collect_reference_vats(model, 0, config)
    records = warmup_records(collected)
    assert len([r for r in records if r["record_type"] == "trajectory"]) == 2
    assert all(r["decision"]["support"][r["decision"]["selected_index"]] == "ACCEPT"
               for r in records if r["record_type"] == "plan")
    assert sum(t.total_cost for g in collected["groups"] for t in g) == collected["metadata"]["search_cost"] > 0
    assert all(not e["sft_eligible"] for e in collected["metadata"]["edit_evidence"])
