"""Discrete daily FIFO inventory simulation with fixed exogenous demand."""
from __future__ import annotations

import hashlib
import numpy as np

from specrsi.benchmarks.science import irp_score
from .common import InvalidCandidate, exact_keys, scalar


def demand_episode(c: dict, seed: int, episode: int):
    # This function never sees a candidate: common random numbers are immutable.
    rng = np.random.default_rng(seed)
    days = np.arange(c["days"] + c["forecast_days"])
    seasonal = 1 + c["weekly_amplitude"] * np.sin(2 * np.pi * days / 7)
    means = seasonal[:, None] * np.array(c["demand_means"])[None, :]
    means *= c["episode_demand_multipliers"][episode % len(c["episode_demand_multipliers"])]
    return rng.poisson(means).astype(np.int64), means


def route_distance(shipments, c):
    remaining = set(np.flatnonzero(shipments).tolist())
    position, total = np.array(c["depot"], float), 0.
    while remaining:
        # Stable index tie-breaking makes the route candidate-independent given stops.
        j = min(remaining, key=lambda i: (float(np.linalg.norm(position - c["locations"][i])), i))
        destination = np.array(c["locations"][j], float)
        total += float(np.linalg.norm(position - destination))
        position = destination
        remaining.remove(j)
    return total + float(np.linalg.norm(position - c["depot"]))


def policy(state: dict, candidate: dict) -> np.ndarray:
    """Bounded order-up-to family; no access to realized future demand."""
    forecast = np.mean(state["forecast"], axis=0)
    recent = np.mean(state["recent_demand"], axis=0) if len(state["recent_demand"]) else forecast
    blend = candidate["forecast_blend"]
    target = candidate["target_days"] * (blend * forecast + (1-blend) * recent) + candidate["safety_stock"]
    current = np.sum(state["inventory"], axis=1)
    quantity = np.maximum(0, np.ceil(target - current)).astype(np.int64)
    quantity = np.minimum(quantity, np.array(state["capacities"]) - current)
    capacity = state["vehicle_capacity"]
    if quantity.sum() > capacity:
        scaled = quantity * capacity / quantity.sum()
        quantity = np.floor(scaled).astype(np.int64)
        for i in np.argsort(-(scaled - quantity), kind="stable")[:capacity-int(quantity.sum())]:
            quantity[i] += 1
    return quantity


def simulate(candidate: dict, c: dict) -> dict:
    exact_keys(candidate, ("target_days", "safety_stock", "forecast_blend"))
    scalar(candidate["target_days"], 0, 4, "target_days")
    scalar(candidate["safety_stock"], 0, 30, "safety_stock")
    scalar(candidate["forecast_blend"], 0, 1, "forecast_blend")
    episodes, demands_hash = [], hashlib.sha256()
    costs = c["costs"]
    for e, seed in enumerate(c["seeds"]):
        demand, forecast = demand_episode(c, seed, e)
        demands_hash.update(demand.tobytes())
        inventory = np.zeros((c["retailers"], c["shelf_life_days"]), dtype=np.int64)
        cumulative = dict.fromkeys(["purchase", "holding", "waste", "lost_sale", "vehicle", "distance"], 0.)
        sold, lost, wasted, max_vehicle = 0, 0, 0, 0
        for day in range(c["days"]):
            state = {"inventory": inventory.copy(), "forecast": forecast[day:day+c["forecast_days"]].copy(),
                     "recent_demand": demand[max(0,day-7):day].copy(), "capacities": c["store_capacities"],
                     "vehicle_capacity": c["vehicle_capacity"], "locations": c["locations"], "costs": costs}
            shipment = policy(state, candidate)
            if shipment.shape != (c["retailers"],) or np.any(shipment < 0) or not np.all(np.isfinite(shipment)):
                raise InvalidCandidate("Malformed shipment vector")
            if shipment.sum() > c["vehicle_capacity"] or np.any(inventory.sum(axis=1) + shipment > c["store_capacities"]):
                raise InvalidCandidate("Vehicle/store capacity violated")
            max_vehicle = max(max_vehicle, int(shipment.sum()))
            inventory[:, -1] += shipment
            remaining = demand[day].copy()
            for age in range(c["shelf_life_days"]):
                used = np.minimum(inventory[:, age], remaining)
                remaining -= used
                inventory[:, age] -= used
            daily_lost = int(remaining.sum())
            daily_waste = int(inventory[:, 0].sum())
            sold += int(demand[day].sum()) - daily_lost
            lost += daily_lost
            wasted += daily_waste
            cumulative["purchase"] += float(shipment.sum()) * costs["purchase"]
            cumulative["lost_sale"] += daily_lost * costs["lost_sale"]
            cumulative["waste"] += daily_waste * costs["waste"]
            cumulative["holding"] += float(inventory[:, 1:].sum()) * costs["holding"]
            if shipment.sum():
                cumulative["vehicle"] += costs["vehicle_fixed"]
                cumulative["distance"] += route_distance(shipment, c) * costs["distance"]
            inventory[:, :-1] = inventory[:, 1:]
            inventory[:, -1] = 0
        total_demand = int(demand[:c["days"]].sum())
        if sold + lost != total_demand:
            raise RuntimeError("Inventory demand conservation failed")
        episodes.append({"seed": seed, "total_cost": sum(cumulative.values()), "total_demand": total_demand,
                         "sold": sold, "lost": lost, "wasted": wasted, "max_shipped": max_vehicle, "cost_breakdown": cumulative})
    result = irp_score(episodes, valid=True, ddof=c["std_ddof"])
    return {"native_score": result.score, "native_unit": result.metrics["native_unit"], "episodes": episodes,
            "demand_sha256": demands_hash.hexdigest(), "simulator": "daily FIFO discrete inventory and Euclidean nearest-neighbor routing"}
