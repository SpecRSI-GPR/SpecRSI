import copy
import json
import math

import pytest

from specrsi.learning.observations import FrozenDonor, rule_values, views


FEATURES = (.8, 2, 1, .5, .1, 3, 4, 5)
OPTIONS = ((1, .2, .5, .7, .9, 2, .03, .6),)


def context():
    return {"scope": "test", "budget": {"total": 100, "available": 80, "pending": 10},
            "tasks": [{"id": "t1", "inputs": {"prompt": "Public task"}}, {"id": "t2", "inputs": {}}],
            "candidates": [{"id": "c", "artifacts": {"system_prompt": "instruction"},
                            "complete_score": .7, "missing": ["t1"], "sources": ["src"],
                            "dependencies": [["j1", "j2"]]}],
            "recent_history": [{"candidate_id": "c", "task_id": "t1", "route": "P", "cost": 3}],
            "per_item_records": [{"source_id": "src", "outcome": {"answer": "c", "score": .2}}]}


def test_blind_masks_numeric_and_text_channels_without_changing_inputs():
    body = context()
    before = copy.deepcopy(body)
    observed = views(FEATURES, body, OPTIONS, "outer", mode="evidence_blind")
    assert observed.features == (.8, 2, 0, 0, .1, 0, 0, 0)
    assert observed.options[0] == (1, 0, 0, .7, 0, 2, .03, 0)
    assert observed.context["budget"] == body["budget"]
    assert observed.context["candidates"][0]["complete_score"] == .7
    assert "missing" not in observed.context_text and "recent_history" not in observed.context_text
    assert "per_item_records" not in observed.context_text and "artifacts" not in observed.context_text
    assert body == before
    exposed = observed.context
    exposed["budget"]["available"] = 0
    assert observed.context["budget"]["available"] == 80


def test_coarse_counts_drop_scope_and_topology_identities():
    body = context()
    one = views(FEATURES, body, OPTIONS, "batch", mode="coarse_counts")
    body["candidates"][0]["missing"] = ["t2"]
    body["candidates"][0]["sources"] = ["different"]
    body["candidates"][0]["dependencies"] = [["k7", "k9"]]
    two = views(FEATURES, body, OPTIONS, "batch", mode="coarse_counts")
    assert one == two
    assert one.context["candidates"][0]["missing_count"] == 1
    assert one.options[0][3] == 0  # Detailed bounded cost is hidden.


@pytest.mark.parametrize("mode,blind_kind,visible_kind", [("outer_blind", "outer", "batch"), ("inner_blind", "batch", "outer")])
def test_role_controls_only_mask_the_selected_role(mode, blind_kind, visible_kind):
    hidden = views(FEATURES, context(), OPTIONS, blind_kind, mode=mode)
    retained = views(FEATURES, context(), OPTIONS, visible_kind, mode=mode)
    assert hidden.features[3] == 0
    assert retained.features == FEATURES
    assert "recent_history" in retained.context


def test_sham_is_deterministic_structural_bijection_and_preserves_outcomes():
    body = context()
    first = views(FEATURES, body, OPTIONS, "outer", mode="sham_relabel", relabel_seed=3)
    second = views(FEATURES, body, OPTIONS, "outer", mode="sham_relabel", relabel_seed=3)
    assert first == second
    assert first.features == FEATURES and first.options == OPTIONS
    transformed = first.context
    alias = transformed["candidates"][0]["id"]
    assert alias != "c" and transformed["recent_history"][0]["candidate_id"] == alias
    assert transformed["candidates"][0]["missing"][0] == transformed["tasks"][0]["id"]
    assert transformed["per_item_records"][0]["outcome"] == {"answer": "c", "score": .2}
    assert len({row["id"] for row in transformed["tasks"]} | {alias}) == 3
    assert first.context != views(FEATURES, body, OPTIONS, "outer", mode="sham_relabel", relabel_seed=4).context


def test_mismatch_requires_frozen_donor_and_keeps_current_quality_budget():
    body = context()
    with pytest.raises(ValueError, match="frozen"):
        views(FEATURES, body, OPTIONS, "outer", mode="mismatched_state")
    donor_body = context()
    donor_body["budget"]["available"] = 1
    donor_body["candidates"][0].update(id="donor-c", complete_score=.01, missing=["t2"])
    donor_body["per_item_records"][0]["outcome"] = {"answer": "donor answer", "score": .99}
    donor = FrozenDonor.from_view(views((.01, 2, 0, 1, .9, 8, 6, 2), donor_body,
                            ((1, .9, 1, .01, 0, 7, .2, .1),), "outer"), donor_id="independent-search-17")
    donor_body["candidates"][0]["missing"].clear()  # Frozen donor is unaffected.
    result = views(FEATURES, body, OPTIONS, "outer", mode="mismatched_state", donor=donor)
    assert result.features[0] == .8 and result.features[4] == .1 and result.features[3] == 1
    assert result.options[0][3] == .7 and result.options[0][6] == .03
    assert result.context["budget"]["available"] == 80
    assert result.context["candidates"][0]["id"] == "c"
    assert result.context["candidates"][0]["complete_score"] == .7
    assert result.context["candidates"][0]["missing"] == ["t2"]
    assert "outcome" not in result.context["per_item_records"][0]
    assert result.context["retained_observed_quality"][0]["outcome"] == {"answer": "c", "score": .2}
    assert dict(result.provenance)["donor_id"] == "independent-search-17"
    identical = FrozenDonor.from_view(views(FEATURES, body, OPTIONS, "outer"), donor_id="identical")
    with pytest.raises(ValueError, match="does not change"):
        views(FEATURES, body, OPTIONS, "outer", mode="mismatched_state", donor=identical)


def test_invalid_truncated_or_unknown_observations_fail_closed():
    with pytest.raises(ValueError, match="before text truncation"):
        views(FEATURES, '{"candidates":', OPTIONS, "outer", mode="evidence_blind")
    with pytest.raises(ValueError, match="Unknown"):
        views(FEATURES, {}, OPTIONS, "outer", mode="wrong")
    with pytest.raises(ValueError, match="finite"):
        views((float("nan"),)*8, {}, OPTIONS, "outer")


def test_rules_distinguish_quality_from_completion_cost_and_use_real_draft_spend():
    args = dict(total_budget=100, spent=20, draft_spent=10)
    quality = rule_values([.9, .5], [10, 1], ["verify", "verify"], rule="quality_only", **args)
    cost = rule_values([.9, .5], [10, 1], ["verify", "verify"], rule="completion_cost", **args)
    assert quality[0] > quality[1] and cost[1] > cost[0]
    assert all(math.isfinite(x) for x in quality + cost)
    boot = rule_values([None, .99], [5, 1], ["draft", "verify"], rule="quality_only", total_budget=100, spent=0, draft_spent=0)
    assert boot[0] > boot[1]
    with pytest.raises(ValueError, match="spending"):
        rule_values([.9], [1], ["verify"], rule="quality_only", total_budget=100, spent=1, draft_spent=2)


def rule_state():
    from types import SimpleNamespace
    from specrsi.core import BudgetLedger, Candidate, Task
    tasks = (Task("one", {}, 1), Task("two", {}, 2))
    candidates = {name: Candidate(name, {"value": name}, {}, operating_penalty=0.)
                  for name in ("high", "low", "unknown")}

    class Ledger:
        def complete(self, candidate, scope):
            return candidate is not candidates["unknown"]

        def score(self, candidate, scope):
            assert self.complete(candidate, scope), "Partial quality must never be fabricated"
            return .9 if candidate is candidates["high"] else .3

        def missing(self, candidate, scope):
            return () if self.complete(candidate, scope) else tasks

    return SimpleNamespace(candidates={c.uid: c for c in candidates.values()},
        scope=SimpleNamespace(tasks=tasks), ledger=Ledger(), budget=BudgetLedger(100)), candidates


def test_rule_callbacks_use_actual_spending_and_known_quality_without_mutation():
    from specrsi.core.search import Action, SearchConfig
    from specrsi.learning.observations import make_rule_controller
    state, candidates = rule_state()
    control = make_rule_controller(SearchConfig(cost_unit="tokens"))
    actions = {name: Action("draft", c.uid, 10) for name, c in candidates.items()}
    values = {name: control.outer_value(state, a) for name, a in actions.items()}
    assert values["high"].value > values["low"].value > values["unknown"].value
    assert all(v.actual_cost == 0 for v in values.values())
    assert state.budget.spent == 0 and not state.budget.charges
    state.budget.reserve("pending-draft", 80, "draft")
    control.outer_value(state, actions["high"])
    assert control.physical_inferences[-1]["actual_draft_spent"] == 0  # Reservation is not spending.
    state.budget.release("pending-draft")
    state.budget.charge("executed-draft", 10, "draft")
    state.budget.charge("executed-check", 10, "acquisition")
    draft = control.outer_value(state, actions["high"])
    verify = control.outer_value(state, Action("verify", candidates["unknown"].uid, 10))
    assert verify.value > draft.value  # Actual draft share .5 exceeds configured .35.
    assert control.physical_inferences[-1]["actual_spent"] == 20
    assert control.physical_inferences[-1]["actual_draft_spent"] == 10
    assert all(r["gpu_count"] == 0 and r["elapsed_seconds"] >= 0 for r in control.physical_inferences)
    state.budget.cost_valid = False
    with pytest.raises(ValueError, match="invalid actual cost"):
        control.outer_value(state, actions["high"])


def test_completion_rule_counts_reserved_work_shared_setup_and_residual_once():
    from specrsi.core.search import SearchConfig
    from specrsi.learning.observations import make_rule_controller
    from specrsi.planning.batches import Batch, BatchProblem, CheckOption
    state, candidates = rule_state()
    candidate = candidates["unknown"]
    a, b = state.scope.tasks
    control = make_rule_controller(SearchConfig(execution_cap=10, cost_unit="tokens"), rule="completion_cost")
    options = (CheckOption("a", candidate.uid, (a.uid,), 10, "shared", reserved_cap=10, reservation_id="protected-a"),
               CheckOption("b", candidate.uid, (b.uid,), 10, "shared", reserved_cap=10, reservation_id="protected-b"))
    problem = BatchProblem(options, {"shared": 3}, 3)
    batch = Batch(("a", "b"))
    assert problem.feasible(batch.option_ids) and problem.cost(batch.option_ids) == 3
    score = control.batch_value(state, candidate, batch, problem)
    assert score.value == pytest.approx(1 / 1.23)
    receipt = control.physical_inferences[-1]
    assert receipt["physical_attempt_cap"] == 23  # Both unpaid reservations + setup once.
    assert receipt["remaining_direct_attempt_cap"] == 0
    control.batch_value(state, candidate, Batch(("a",)), problem)
    assert control.physical_inferences[-1]["physical_attempt_cap"] == 13
    assert control.physical_inferences[-1]["remaining_direct_attempt_cap"] == 10


def test_rule_native_meter_covers_ledger_feature_work(monkeypatch):
    from specrsi.core.search import Action, SearchConfig
    from specrsi.learning.observations import make_rule_controller
    state, candidates = rule_state()
    clock = [1.]
    original = state.ledger.complete

    def costly_complete(candidate, scope):
        clock[0] += .02
        return original(candidate, scope)

    state.ledger.complete = costly_complete
    monkeypatch.setattr("specrsi.learning.observations.time.perf_counter", lambda: clock[0])
    control = make_rule_controller(SearchConfig(cost_unit="wall-ms"))
    estimate = control.outer_value(state, Action("draft", candidates["high"].uid, 10))
    assert estimate.actual_cost >= 80
    assert control.physical_inferences[-1]["elapsed_seconds"] >= .08
