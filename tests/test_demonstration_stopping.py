from types import SimpleNamespace

import pytest
import torch

from specrsi.core.ledger import Ledger
from specrsi.core.models import Candidate, EvaluationScope, Outcome, Task
from specrsi.core.vats import VATSPlan
from specrsi.learning.demonstrations import (
    DemonstrationStopping, DemonstrationVATSBridge, demonstration_bridge_kwargs, warmup_records,
)
from specrsi.learning.heads import TinyModelConfig, TinyRoleModel
from specrsi.learning.paper_bridge import VATSSearchBridge, collect_reference_vats
from specrsi.learning.pipeline import stage_configs


def bridge(**settings):
    model = TinyRoleModel(TinyModelConfig(option_dim=12, hidden_dim=8))
    return DemonstrationVATSBridge(model, demonstration_policy="observed_plateau", **settings)


def observed_state(scores, *, incomplete_last=False, inadmissible_last=False):
    task = Task("visible task", {"prompt": "public input"}, "PRIVATE_REFERENCE")
    scope = EvaluationScope("local", (task,), "execution", "scorer", split="train")
    ledger = Ledger()
    root = Candidate("root", {"instruction": "root"})
    candidates = {root.uid: root}
    ledger.admit_direct(root, task, scope, Outcome(score=.5, fields={"answer": "seen root"}), {"ok": True})
    for index, score in enumerate(scores):
        child = Candidate(f"child-{index}", {"instruction": str(index)}, parent_id=root.uid)
        candidates[child.uid] = child
        ledger.register(child, scope)
        if not (incomplete_last and index == len(scores) - 1):
            ledger.admit_direct(child, task, scope,
                Outcome(score=score, admissible=not (inadmissible_last and index == len(scores) - 1),
                        fields={"answer": "observed child"}), {"ok": True})
    return SimpleNamespace(candidates=candidates, ledger=ledger, scope=scope, version=5,
                           current_transition_index=5), root


@pytest.mark.parametrize("settings", [{"patience": 0}, {"patience": True},
    {"min_completed_edits": 0}, {"min_gain": float("nan")}, {"min_gain": -1},
    {"min_gain": True}, {"typo": 1}])
def test_stopping_configuration_rejects_invalid_rules(settings):
    with pytest.raises(ValueError):
        DemonstrationStopping.from_settings(settings)


def test_pipeline_preserves_registered_teacher_and_removes_it_from_gvo(tmp_path):
    config = {"stage": "two_stage", "collector": "actual:collect",
        "collection": {"algorithm": "vats_pdf"},
        "warmup": {"collection": {"demonstration_policy": "observed_plateau",
            "demonstration_stopping": {"patience": 2}}}}
    warmup, gvo = stage_configs(config, tmp_path)
    assert warmup["collection"]["demonstration_policy"] == "observed_plateau"
    assert warmup["collection"]["demonstration_stopping"]["patience"] == 2
    assert "demonstration_stopping" not in gvo["collection"]
    assert "demonstration_policy" not in gvo["collection"]
    with pytest.raises(ValueError, match="Unknown registered"):
        demonstration_bridge_kwargs({"demonstration_policy": "peek_at_holdout"})
    with pytest.raises(ValueError, match="requires observed_plateau"):
        demonstration_bridge_kwargs({"demonstration_policy": "complete_then_draft",
                                     "demonstration_stopping": {"patience": 2}})


@pytest.mark.parametrize("scores,stop", [([], False), ([.6], False), ([.5], True),
    ([.3], True), ([.3, .6], False), ([.6, .3], True)])
def test_plateau_depends_only_on_actual_observed_completed_edits(scores, stop):
    state, _ = observed_state(scores)
    result = bridge().observed_stopping_evidence(state)
    assert result["stop"] is stop
    assert "PRIVATE_REFERENCE" not in str(result)
    assert len(result["completed_edits"]) == len(scores)


def test_incomplete_and_inadmissible_edits_are_distinguished():
    state, _ = observed_state([.1], incomplete_last=True)
    assert bridge().observed_stopping_evidence(state)["completed_edits"] == []
    assert not bridge().observed_stopping_evidence(state)["stop"]
    state, _ = observed_state([.9], inadmissible_last=True)
    result = bridge().observed_stopping_evidence(state)
    assert result["stop"] and result["completed_edits"][0]["child_utility"] is None


def test_patience_and_minimum_completed_edits_do_not_use_forecasts():
    teacher = bridge(demonstration_stopping={"patience": 2, "min_completed_edits": 3})
    for scores, expected in [([.3, .3], False), ([.6, .3, .3], True), ([.3, .3, .6], False)]:
        state, _ = observed_state(scores)
        assert teacher.observed_stopping_evidence(state)["stop"] is expected


def test_accept_is_selected_in_a_nontrivial_live_menu(monkeypatch):
    monkeypatch.setattr(VATSSearchBridge, "policy_menu", lambda self, state, plans: [0.] * len(plans))
    state, root = observed_state([.4])
    teacher = bridge()
    plans = (VATSPlan("accept"), VATSPlan("draft", (root.uid,), work_cap=1))
    logits = teacher.policy_menu(state, plans)
    assert max(range(len(plans)), key=logits.__getitem__) == 0
    assert teacher.teacher_choice_evidence[-1]["reason"] == "observed_plateau"
    assert teacher.teacher_choice_evidence[-1]["support"] == [p.uid for p in plans]
    legacy = DemonstrationVATSBridge(teacher.model)
    logits = legacy.policy_menu(state, plans)
    assert max(range(len(plans)), key=logits.__getitem__) == 1


@pytest.mark.parametrize("temperature", [1., 10000.])
def test_actual_collection_records_nontrivial_accept_without_inventing_actor_labels(temperature):
    torch.set_num_threads(2)
    model = TinyRoleModel(TinyModelConfig(option_dim=12, hidden_dim=8, seed=3))
    collection = collect_reference_vats(model, 0, {"stage": "warmup", "training": {"seed": 3},
        "collection": {"algorithm": "vats_pdf", "contexts": 1, "group_size": 2, "budget": 2000,
            "max_actions": 9, "max_candidates": 5, "max_children": 4, "exploration": 0., "temperature": temperature,
            "demonstration_policy": "observed_plateau", "demonstration_stopping": {"min_gain": 2.},
            "vats": {"max_queries": 0, "max_draft_depth": 1, "max_draft_parents": 1}}})
    choices = collection["metadata"]["teacher_choice_evidence"]
    stopping = [row for row in choices if row["reason"] == "observed_plateau"]
    assert len(stopping) == 2
    assert all(row["selected_id"] == "ACCEPT" and len(row["support"]) > 1 for row in stopping)
    plans = [row["decision"] for row in warmup_records(collection) if row["record_type"] == "plan"]
    accepted = [row for row in plans if row["support"][row["selected_index"]] == "ACCEPT"]
    assert len(accepted) == 2 and all(len(row["support"]) > 1 for row in accepted)
    assert all(row["deterministic"] and row["behavior_logprob"] is None for row in accepted)
    assert collection["metadata"]["search_cost"] == sum(t.total_cost for g in collection["groups"] for t in g)
    assert all(t.demonstration and not t.forced and t.total_cost > 0 for g in collection["groups"] for t in g)
    intended = {(row["search_id"], row["transition_index"]): row for row in choices}
    for group in collection["groups"]:
        for trajectory in group:
            for decision in trajectory.decisions:
                if decision.kind == "plan":
                    evidence = intended[(trajectory.search_id, decision.transition_index)]
                    assert evidence["selected_id"] == decision.support[decision.selected_index]
                    assert tuple(evidence["support"]) == decision.support
                elif decision.kind == "edit":
                    assert not decision.deterministic  # The drafter remains sampled.


@pytest.mark.parametrize("corruption", ["missing", "selected", "support", "index", "sampled"])
def test_conversion_rejects_teacher_evidence_or_determinism_mismatch(monkeypatch, corruption):
    from dataclasses import replace
    from specrsi.learning.records import DecisionRecord, SearchTrajectory, TransitionRecord
    teacher = bridge()
    decision = DecisionRecord("plan", ("ACCEPT", "draft:root"), 0, None,
        deterministic=True, transition_index=0, exploration=0.)
    teacher.teacher_choice_evidence = [{"selected_id": "ACCEPT", "support": list(decision.support), "transition_index": 0}]
    if corruption == "missing":
        teacher.teacher_choice_evidence = []
    elif corruption == "sampled":
        decision = replace(decision, deterministic=False, behavior_logprob=-.6931471805599453)
    elif corruption == "selected":
        teacher.teacher_choice_evidence[0]["selected_id"] = "draft:root"
    elif corruption == "support":
        teacher.teacher_choice_evidence[0]["support"] = ["ACCEPT"]
    else:
        teacher.teacher_choice_evidence[0]["transition_index"] = 1
    trajectory = SearchTrajectory("group", "root", "search", 0, (TransitionRecord(0., 0., 0.),),
                                  (decision,), lineage_id="root")
    monkeypatch.setattr(VATSSearchBridge, "convert", lambda *args, **kwargs: trajectory)
    with pytest.raises(ValueError, match="deterministically|exactly match"):
        teacher.convert(SimpleNamespace(), lineage_id="root")
