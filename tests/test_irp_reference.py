import copy
import random

import numpy as np
import pytest

from specrsi.science.common import InvalidCandidate
from specrsi.science.executor import ScienceExecutor, make_candidate, make_scope, mutate
from specrsi.science.irp import simulate
from specrsi.science.irp_reference import demand_sequence, delivery_trips, nearest_neighbor, run_policy
from specrsi.science.registry import case_count, evaluate, load_config


EXPECTED_SIZES = [(3,7), (4,7), (6,10), (6,14), (7,10), (7,14),
                  (9,14), (11,14), (11,21), (13,14), (15,21), (15,28)]


@pytest.mark.parametrize("case_id", range(12))
def test_twelve_reference_cases_enforce_all_daily_constraints(case_id):
    c = load_config("irp", "reference", case_id)
    assert (c["retailers"], c["days"]) == EXPECTED_SIZES[case_id]
    assert c["demand_seed"] == 1001 + case_id
    r = evaluate("irp", c["baseline"], scenario_set="reference", case_id=case_id)
    assert r["valid"] and r["complete"]
    assert r["native_score"] == 0  # The full-capacity policy defines TC_ref.
    assert r["service_level"] >= .95
    assert r["satisfied"] + r["shortage"] == r["total_demand"]
    for day in r["daily"]:
        assert all(distance <= c["max_daily_distance_per_vehicle"] + 1e-9
                   for distance in day["vehicle_distance_km"])
        assert all(trip["load"] <= 500 for trip in day["trips"])
        assert all(0 <= inv <= cap for inv, cap in
                   zip(day["ending_inventory"], c["store_capacities"]))
        assert all(trip["vehicle"] < c["vehicle_count"] for trip in day["trips"])
    assert r["total_cost"] == pytest.approx(sum(r["costs"].values()))


def tiny_case(**updates):
    c = copy.deepcopy(load_config("irp", "reference", 0))
    c.update(retailers=1, days=2, store_capacities=[20], initial_inventory=[0],
             locations=[[1,0]], vehicle_count=1, max_daily_distance_per_vehicle=10)
    c.update(updates)
    return c


def test_expiry_is_not_counted_as_unmet_demand_or_charged_twice():
    c = tiny_case(shelf_life_days=3, initial_inventory=[2])
    r = run_policy({"target_stock": [0]}, c, np.array([[0], [1]]))
    assert [day["expired"] for day in r["daily"]] == [0, 2]
    assert r["shortage"] == 1 and r["expired"] == 2
    assert r["costs"] == {"holding": 1.0, "shortage": 10.0, "transport": 0.0}


def test_same_day_delivery_and_oldest_first_consumption():
    c = tiny_case(shelf_life_days=3, initial_inventory=[1])
    r = run_policy({"target_stock": [2]}, c, np.array([[1], [1]]))
    assert r["satisfied"] == 2 and r["expired"] == 0
    assert [day["shipments"] for day in r["daily"]] == [[1], [1]]
    assert r["costs"] == {"holding": 1.0, "shortage": 0.0, "transport": 12.0}


def test_next_fit_is_not_first_fit_and_routes_have_stable_ties():
    c = load_config("irp", "reference", 0)
    trips, _ = delivery_trips(np.array([300, 250, 200]), c)
    assert [sorted(t["stores"]) for t in trips] == [[0], [1, 2]]
    assert [t["load"] for t in trips] == [300, 450]
    c["locations"] = [[1,0], [-1,0], [0,1]]
    order, distance = nearest_neighbor([2, 1, 0], c)
    assert order == [0, 2, 1]
    assert distance == pytest.approx(2 + 2 * np.sqrt(2))


def test_daily_mileage_limit_accumulates_all_trips_of_one_vehicle():
    c = load_config("irp", "reference", 0)
    c.update(locations=[[1,0], [1,0], [1,0]], vehicle_count=1,
             max_daily_distance_per_vehicle=3)
    with pytest.raises(InvalidCandidate, match="cumulative daily"):
        delivery_trips(np.array([300, 300, 0]), c)


def test_service_threshold_is_inclusive_and_uses_satisfied_demand(monkeypatch):
    from specrsi.science import irp_reference
    c = tiny_case(days=1)
    monkeypatch.setattr(irp_reference, "demand_sequence", lambda _: np.array([[20]]))
    assert simulate({"target_stock": [19]}, c)["service_level"] == .95
    with pytest.raises(InvalidCandidate, match="below 95%"):
        simulate({"target_stock": [18]}, c)


@pytest.mark.parametrize("targets", [[-1], [21], [10.0], [True], [], [1,2]])
def test_invalid_inventory_decisions_fail_closed(targets):
    with pytest.raises(InvalidCandidate):
        simulate({"target_stock": targets}, tiny_case())


def test_candidate_cannot_change_demand_or_capacity():
    c = load_config("irp", "reference", 0)
    with pytest.raises(InvalidCandidate, match="exactly"):
        simulate({**c["baseline"], "demand_seed": 99}, c)
    np.random.seed(7)
    first = demand_sequence(c)
    np.random.seed(99)
    assert np.array_equal(demand_sequence(c), first)
    assert first.shape == (7,3) and np.issubdtype(first.dtype, np.integer)
    assert np.all(first >= 0)


def test_feasible_lower_stock_policy_uses_fixed_reference_cost_and_demand():
    c = load_config("irp", "reference", 0)
    root = simulate(c["baseline"], c)
    candidate = simulate({"target_stock": [60] * 3}, c)
    assert candidate["demand_sha256"] == root["demand_sha256"]
    assert candidate["reference_cost"] == root["total_cost"]
    assert 0 < candidate["native_score"] < 1
    assert candidate["native_score"] == pytest.approx(1 - candidate["total_cost"]/root["total_cost"])


def test_reference_registry_and_worker_reach_last_instance():
    assert case_count("irp", "reference") == 12
    assert case_count("irp", "selection") == 11
    for bad in [-1,12,True]:
        with pytest.raises(ValueError, match="case ID"):
            load_config("irp", "reference", bad)
    scope = make_scope("irp", "reference", (11,))
    candidate = make_candidate("irp", scenario_set="reference", case_id=11)
    result = ScienceExecutor().run(candidate, scope.tasks[0], scope, 30000)
    assert result.status == "complete", result.error
    assert result.outcome.admissible
    assert result.receipt["native_result"]["instance_id"] == "irp_case_12"
    assert scope.aggregation_id == "science_native_weighted_mean_v1"


def test_new_mutations_preserve_schema_and_old_irp_remains_available():
    mutated = mutate("irp", {"target_stock": [100, 100, 100]}, random.Random(2))
    assert set(mutated) == {"target_stock"}
    assert all(type(v) is int and v >= 0 for v in mutated["target_stock"])
    old = load_config("irp")["baseline"]
    assert set(mutate("irp", old, random.Random(2))) == set(old)
