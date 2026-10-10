from dataclasses import replace

import pytest

from specrsi.core import (AcquisitionCaps, BudgetLedger, Candidate, ConservativeChecker, EvaluationScope,
                      Ledger, PureIntegerExecutor, ResultAcquirer, Task)
from specrsi.core.demo import IntegerRepairDrafter, run_demo
from specrsi.core.search import DraftResult, SearchConfig, SearchEngine, ValueEstimate


def make_engine(config=None, total=200, **kwargs):
    tasks = (Task("read", {"route": "read", "x": 2}, 2), Task("write", {"route": "write", "x": 2}, 3))
    scope = EvaluationScope("search", tasks, "pure-int-v1", "integer-equality-v1", ("answer", "primitives"))
    incumbent = Candidate("base", {"routes": {"read": ("x",), "write": ("x",)}}, {"contract": "pure-int-v1"}, "pure_integer", True)
    ledger, executor = Ledger(ConservativeChecker(1)), PureIntegerExecutor()
    setup_budget = BudgetLedger(100)
    setup = ResultAcquirer(ledger, setup_budget, executor)
    for task in tasks:
        setup.acquire(incumbent, task, scope, (), AcquisitionCaps(10))
    config = config or SearchConfig(execution_cap=10, draft_cap=1, max_actions=4, max_candidates=3, max_children=1)
    return SearchEngine(scope, incumbent, ledger, BudgetLedger(total), executor, IntegerRepairDrafter(), config, **kwargs)


def test_demo_finishes_improvement_with_paid_p_and_u_evidence():
    result = run_demo()
    assert result["acceptance"]["selected_score"] == 1.
    assert result["acceptance"]["improved"]
    assert {r["route"] for r in result["ledger"]["records"]} == {"P", "U"}
    assert result["value_source"] == "heuristic-fallback"
    assert result["total_cost"] == sum(t["cost"] for t in result["transitions"])
    assert result["total_cost"] == sum(c["actual"] for c in result["charges"])
    assert all(c["actual"] <= c["cap"] for c in result["charges"])


def test_all_u_ablation_disables_every_inheritance_link():
    result = run_demo(all_u=True)
    assert result["acceptance"]["selected_score"] == 1.
    assert {r["route"] for r in result["ledger"]["records"]} == {"U"}
    assert all(not event.get("checks") for t in result["transitions"] for event in t["events"])


def test_finalist_completes_after_last_allowed_regular_action():
    engine = make_engine(SearchConfig(execution_cap=10, draft_cap=1, max_actions=1, max_children=1), total=25)
    result = engine.run()
    assert result.acceptance.improved
    assert [t.action for t in result.transitions] == ["draft", "finalist_completion", "finalist_completion"]
    assert result.total_cost == 6
    assert engine.state.budget.pending == 0
    assert engine.state.budget.invariant()


def test_unaffordable_completion_makes_no_promise_and_incumbent_survives():
    engine = make_engine(SearchConfig(execution_cap=10, draft_cap=1, max_actions=1, max_children=1), total=5)
    result = engine.run()
    assert result.selected == engine.state.incumbent
    assert len(engine.state.candidates) == 2
    assert engine.state.finalist_id is None
    assert engine.state.budget.pending == 0


def test_partial_parent_can_expand_and_source_records_are_retained():
    seen = []

    class ExpandDrafter:
        def draft(self, parent, state, cap, rng):
            seen.append((parent.uid, len(state.ledger.missing(parent, state.scope))))
            return IntegerRepairDrafter().draft(parent, state, cap, rng)

    def prefer_deep_draft(state, action):
        if action.kind == "verify":
            return -100.
        return 100. if action.candidate_id != state.incumbent.uid else 1.

    cfg = SearchConfig(execution_cap=10, draft_cap=1, max_actions=2, max_candidates=3, max_children=1, reserve_completion=False)
    engine = make_engine(cfg, outer_value=prefer_deep_draft)
    engine.drafter = ExpandDrafter()
    result = engine.run()
    assert seen[0][1] == 0 and seen[1][1] == 2
    assert len(engine.state.candidates) == 3
    assert len(engine.state.ledger.records) == 2
    assert not result.acceptance.improved


def test_state_changes_trigger_complete_rescoring_and_capture_original_features():
    calls = {}

    def head(state, action):
        calls.setdefault(state.version, []).append((action.uid, state.features()))
        return ValueEstimate(10. if action.kind == "verify" else 1., 1)

    engine = make_engine(outer_value=head)
    result = engine.run()
    for decision in result.decisions:
        if decision.kind == "outer":
            rows = calls[decision.state_version]
            assert tuple(uid for uid, _ in rows) == decision.support
            assert all(features == decision.state_features for _, features in rows)
    assert len(calls) > 1


def test_collection_records_exact_support_positive_probabilities_and_cost_origins():
    engine = make_engine(SearchConfig(execution_cap=10, draft_cap=1, max_actions=6, max_children=1,
                                      max_candidates=3, mode="collect", seed=3, exploration=.2))
    result = engine.run()
    assert result.decisions and all(d.stochastic for d in result.decisions)
    for decision in result.decisions:
        assert sum(decision.probabilities) == pytest.approx(1.)
        assert all(p > 0 for p in decision.probabilities)
        assert decision.selected_id in decision.support
        assert len(decision.support) == len(set(decision.support))
        if decision.kind == "batch":
            assert decision.cost_before_origin > 0
            assert decision.planning_cost_paid == engine.config.planning_cost
            assert decision.proposal_snapshot


def test_native_aggregate_and_penalty_remain_external_to_search_cost():
    engine = make_engine()
    original = engine.drafter

    class ExpensiveRepair:
        def draft(self, parent, state, cap, rng):
            result = original.draft(parent, state, cap, rng)
            return replace(result, candidate=replace(result.candidate, operating_penalty=.6))

    engine.drafter = ExpensiveRepair()
    result = engine.run()
    assert not result.acceptance.improved
    assert result.total_cost > 0
    assert result.acceptance.selected_score == .5


def test_stale_or_invalid_candidate_drafts_fail_before_admission():
    engine = make_engine()

    class WrongParent:
        def draft(self, parent, state, cap, rng):
            return DraftResult(replace(parent, name="bad", parent_id="wrong-parent"), 1)

    engine.drafter = WrongParent()
    with pytest.raises(ValueError, match="actual parent"):
        engine.run()
    assert len(engine.state.candidates) == 1
    assert engine.state.budget.spent == 2


def test_no_legal_selection_cost_preserves_budget_and_incumbent():
    engine = make_engine(total=0)
    result = engine.run()
    assert result.total_cost == 0 and not result.decisions
    assert result.selected == engine.state.incumbent
    assert result.stop_reason == "no_affordable_legal_action"


def test_no_reusing_search_state_for_independent_group_members():
    engine = make_engine()
    engine.run()
    with pytest.raises(ValueError, match="single-use"):
        engine.run()


def test_group_context_identity_includes_actual_initial_outcomes():
    from specrsi.core import Outcome
    config = SearchConfig(max_actions=0)
    first = make_engine(config)
    second = make_engine(config)
    initial_identity = first.run().initial_context_id
    assert initial_identity == second.run().initial_context_id
    third = make_engine(config)
    original = third.state.ledger
    altered = Ledger()
    for task in third.state.scope.tasks:
        record = original.get(third.state.incumbent, task, third.state.scope)
        changed = Outcome(0., {"answer": -1, "primitives": 1})
        altered.admit_direct(third.state.incumbent, task, third.state.scope, changed, record.receipt)
    third.state.ledger = altered
    assert third.run().initial_context_id != initial_identity


def test_deployment_exploration_has_a_separate_enforced_budget():
    config = SearchConfig(execution_cap=10, draft_cap=1, max_actions=3, max_candidates=2, max_children=1,
                          planning_cost=0, outer_value_cap=0, batch_value_cap=0, exploration=1., exploration_budget=2)
    engine = make_engine(config)
    result = engine.run()
    assert result.acceptance.improved
    assert engine._exploration_spent <= config.exploration_budget
