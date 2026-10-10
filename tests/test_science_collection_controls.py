"""Registered science collection controls must govern actual work and likelihoods."""
import math

import pytest

from specrsi.core import Candidate, EvaluationScope, Outcome, Task
from specrsi.core.execution import ExecutionResult
from specrsi.core.search import DraftResult
from test_runner_integration import FakeRoles


@pytest.fixture
def science_collection(tmp_path, monkeypatch):
    import specrsi.learning.science_bridge as collector

    calls, drafts = [], []

    def make_scope(domain, split, cases, **kwargs):
        return EvaluationScope("science-controls", tuple(Task(f"case-{case}") for case in cases),
            "native-test", "native-test", scored_fields=("native_score",),
            aggregation_id="science_native_weighted_mean_v1", split="dev")

    class Executor:
        def run(self, candidate, task, scope, cap, resume_token=None):
            calls.append(candidate.uid)
            score = 11. if candidate.parent_id else 10.
            return ExecutionResult("complete", 3, Outcome(.5, {"native_score": score}),
                receipt={"wall_seconds": .003, "native_result": {"complete": True}})

    class Drafter:
        def __init__(self, bridge, **kwargs):
            pass

        def draft(self, parent, state, cap, rng):
            drafts.append(parent.uid)
            return DraftResult(Candidate("improved", {"value": 11.}, parent_id=parent.uid), 2)

    monkeypatch.setattr(collector, "make_scope", make_scope)
    monkeypatch.setattr(collector, "make_candidate", lambda domain: Candidate("initial", {"value": 10.}))
    monkeypatch.setattr(collector, "ScienceExecutor", Executor)
    monkeypatch.setattr(collector, "HFParameterDrafter", Drafter)
    config = {"output": str(tmp_path), "training": {"seed": 17}, "collection": {
        "algorithm": "vats_pdf", "domain": "reactor", "case_ids": [0], "group_size": 2,
        "execution_cap_ms": 20, "draft_cap_ms": 10, "budget_ms": 2000, "head_cap_ms": 100,
        "max_actions": 4, "max_candidates": 2, "max_children": 1,
        "temperature": .37, "exploration": .1,
        "vats": {"max_queries": 8, "max_extensions": 0, "max_draft_depth": 1,
            "max_draft_parents": 1, "max_completion_targets": 1, "max_pair_targets": 0,
            "max_singletons": 1, "max_menu_options": 4, "solver_call_cost": 0,
            "policy_call_cost": 0, "scorer_query_cap": 100}}}
    return collector, config, calls, drafts


def test_zero_children_removes_draft_work_from_science_search(science_collection):
    collector, config, calls, drafts = science_collection
    config["collection"]["max_children"] = 0
    result = collector.collect_science_specrsi(FakeRoles(), 0, config)
    assert len(calls) == 1  # Only the shared initial root is evaluated.
    assert not drafts
    for trajectory in result["groups"][0]:
        assert trajectory.decisions[0].support == ("ACCEPT",)


def test_science_policy_records_configured_temperature_and_actual_probability(science_collection):
    collector, config, _, _ = science_collection
    result = collector.collect_science_specrsi(FakeRoles(), 0, config)
    # Root has ACCEPT (logit 0) and one DRAFT (logit 1) before any new work.
    expected_accept = .05 + .9 / (1. + math.exp(1. / .37))
    for trajectory in result["groups"][0]:
        choice = trajectory.decisions[0]
        assert choice.temperature == .37
        assert len(choice.support) == 2
        index = choice.support.index("ACCEPT")
        assert choice.behavior_probabilities[index] == pytest.approx(expected_accept)
        assert all(decision.temperature == .37 for decision in trajectory.decisions if decision.kind == "plan")


@pytest.mark.parametrize("field,value,nested", [
    ("max_children", -1, False), ("temperature", 0., False), ("budget_ms", -1, False),
    ("execution_cap_ms", -1, False), ("max_menu_options", 1, True), ("max_draft_depth", 0, True),
])
def test_invalid_science_contract_is_rejected_before_native_root_calls(science_collection, field, value, nested):
    collector, config, calls, drafts = science_collection
    settings = config["collection"]["vats"] if nested else config["collection"]
    settings[field] = value
    with pytest.raises(ValueError):
        collector.collect_science_specrsi(FakeRoles(), 0, config)
    assert not calls
    assert not drafts
