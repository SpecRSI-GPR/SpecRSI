"""Same-day perishable inventory routing with fixed, candidate-independent demand."""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import numpy as np

from .common import InvalidCandidate, exact_keys


def demand_sequence(c: dict) -> np.ndarray:
    """Instance seeds fix demand; search/training seeds do not change this array."""
    rng = np.random.Generator(np.random.PCG64(c["demand_seed"]))
    draws = rng.normal(c["demand_mean"], c["demand_mean"] * c["demand_cv"],
                       size=(c["days"], c["retailers"]))
    return np.floor(np.maximum(draws, 0)).astype(np.int64)


def nearest_neighbor(stops: list[int], c: dict) -> tuple[list[int], float]:
    """Return retailer indices and depot-to-depot distance; ties use retailer ID."""
    remaining = set(stops)
    point = c["depot"]
    order, distance = [], 0.0
    while remaining:
        i = min(remaining, key=lambda j: (math.dist(point, c["locations"][j]), j))
        destination = c["locations"][i]
        distance += math.dist(point, destination)
        order.append(i)
        remaining.remove(i)
        point = destination
    return order, distance + (math.dist(point, c["depot"]) if order else 0.0)


def delivery_trips(shipments: np.ndarray, c: dict) -> tuple[list[dict], list[float]]:
    """Next-fit, whole-store orders; trips cycle through the fixed fleet each day."""
    groups, current, load = [], [], 0
    for i, amount in enumerate(shipments.tolist()):
        if not amount:
            continue
        if amount < 0 or amount > c["vehicle_capacity"]:
            raise InvalidCandidate("IRP store delivery exceeds vehicle capacity")
        if load + amount > c["vehicle_capacity"]:
            groups.append(current)
            current, load = [], 0
        current.append(i)
        load += amount
    if current:
        groups.append(current)
    trips, vehicle_distances = [], [0.0] * c["vehicle_count"]
    for trip_index, stores in enumerate(groups):
        order, distance = nearest_neighbor(stores, c)
        vehicle = trip_index % c["vehicle_count"]
        vehicle_distances[vehicle] += distance
        if vehicle_distances[vehicle] > c["max_daily_distance_per_vehicle"] + 1e-9:
            raise InvalidCandidate("IRP cumulative daily vehicle distance exceeds limit")
        trips.append({"vehicle": vehicle, "stores": order,
                      "load": int(shipments[stores].sum()), "distance_km": distance})
    return trips, vehicle_distances


def _targets(candidate: dict, c: dict) -> np.ndarray:
    exact_keys(candidate, ("target_stock",))
    values = candidate["target_stock"]
    if not isinstance(values, (list, tuple)) or len(values) != c["retailers"]:
        raise InvalidCandidate("IRP target_stock must contain one integer per retailer")
    if any(type(x) is not int or not 0 <= x <= cap
           for x, cap in zip(values, c["store_capacities"])):
        raise InvalidCandidate("IRP target_stock violates integer inventory capacity bounds")
    return np.asarray(values, dtype=np.int64)


def run_policy(candidate: dict, c: dict, demands: np.ndarray) -> dict:
    """Run an order-up-to policy without exposing current or future demand to it."""
    targets = _targets(candidate, c)
    n, life = c["retailers"], c["shelf_life_days"]
    if demands.shape != (c["days"], n) or np.any(demands < 0):
        raise ValueError("Malformed exogenous demand")
    capacities = np.asarray(c["store_capacities"], dtype=np.int64)
    initial = np.asarray(c["initial_inventory"], dtype=np.int64)
    if initial.shape != (n,) or np.any(initial < 0) or np.any(initial > capacities):
        raise ValueError("Invalid initial inventory")
    inventory = np.zeros((n, life), dtype=np.int64)  # column is age in days
    age = c["initial_inventory_age_days"]
    if type(age) is not int or not 0 <= age < life:
        raise ValueError("Invalid initial inventory age")
    inventory[:, age] = initial
    daily, costs = [], {"holding": 0.0, "shortage": 0.0, "transport": 0.0}
    shipped_total = sold_total = lost_total = expired_total = 0
    for day in range(c["days"]):
        # Age once at day start, expire, decide, deliver, then realize demand.
        expired = int(inventory[:, -1].sum())
        inventory[:, 1:] = inventory[:, :-1].copy()
        inventory[:, 0] = 0
        shipments = np.maximum(targets - inventory.sum(axis=1), 0)
        trips, vehicle_distances = delivery_trips(shipments, c)
        inventory[:, 0] += shipments
        if np.any(inventory.sum(axis=1) > capacities):
            raise InvalidCandidate("IRP store inventory capacity exceeded")
        unmet = demands[day].copy()
        for batch_age in reversed(range(life)):
            served = np.minimum(inventory[:, batch_age], unmet)
            inventory[:, batch_age] -= served
            unmet -= served
        demand = int(demands[day].sum())
        lost = int(unmet.sum())
        sold = demand - lost
        distance = math.fsum(vehicle_distances)
        day_costs = {"holding": float(inventory.sum()) * c["costs"]["holding"],
                     "shortage": lost * c["costs"]["shortage"],
                     "transport": distance * c["costs"]["distance"]}
        for key in costs:
            costs[key] += day_costs[key]
        shipped_total += int(shipments.sum())
        sold_total += sold
        lost_total += lost
        expired_total += expired
        daily.append({"day": day + 1, "demand": demand, "satisfied": sold,
                      "shortage": lost, "expired": expired, "shipments": shipments.tolist(),
                      "ending_inventory": inventory.sum(axis=1).tolist(),
                      "vehicle_distance_km": vehicle_distances, "trips": trips,
                      "costs": day_costs})
    total_demand = int(demands.sum())
    if total_demand <= 0:
        raise InvalidCandidate("IRP service level is undefined without demand")
    if sold_total + lost_total != total_demand:
        raise RuntimeError("IRP demand conservation failed")
    if int(initial.sum()) + shipped_total != sold_total + expired_total + int(inventory.sum()):
        raise RuntimeError("IRP inventory conservation failed")
    return {"total_cost": math.fsum(costs.values()), "costs": costs,
            "service_level": sold_total / total_demand, "total_demand": total_demand,
            "satisfied": sold_total, "shortage": lost_total, "expired": expired_total,
            "daily": daily}


def simulate(candidate: dict, c: dict) -> dict:
    demands = demand_sequence(c)
    result = run_policy(candidate, c, demands)
    if result["service_level"] < c["minimum_service_level"]:
        raise InvalidCandidate(f"IRP service level {result['service_level']:.6f} is below 95%")
    reference_candidate = {"target_stock": c["store_capacities"]}
    reference = result if candidate == reference_candidate else run_policy(reference_candidate, c, demands)
    if reference["service_level"] < c["minimum_service_level"]:
        raise ValueError("IRP reference policy does not meet the service constraint")
    if reference["total_cost"] <= 0:
        raise ValueError("IRP reference cost must be positive")
    return {"native_score": max(0.0, 1.0 - result["total_cost"] / reference["total_cost"]),
            "native_unit": "dimensionless normalized cost reduction",
            "instance_id": c["instance_id"], "reference_cost": reference["total_cost"],
            "demand_seed": c["demand_seed"],
            "demand_sha256": hashlib.sha256(demands.astype("<i8").tobytes()).hexdigest(),
            "simulator": "daily FIFO inventory; next-fit loads; nearest-neighbor routes",
            **result}


def main() -> None:
    from .registry import case_count, evaluate, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    ids = range(case_count("irp", "reference")) if args.all_cases else [args.case_id]
    results = []
    for i in ids:
        c = load_config("irp", "reference", i)
        results.append(evaluate("irp", c["baseline"], scenario_set="reference", case_id=i))
    output = json.dumps({"scenario_set": "reference", "results": results}, indent=2) + "\n"
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output)
    else:
        print(output, end="")
    if not all(row["valid"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
