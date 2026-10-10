"""Aggregate complete trusted simulator outputs into task utilities.

Each transform consumes the full native result vector for its domain.
The simulator and scenario configuration remain external to aggregation.
"""
from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

from .base import ScoreResult


def _complete(values: Sequence[float], count: int):
    if len(values) != count or not all(math.isfinite(float(x)) for x in values):
        raise ValueError(f"Expected {count} finite complete native scenario outputs")


def pybamm_score(end_of_life_cycles: Sequence[float], *, valid: bool, temperatures_c: Sequence[float]) -> ScoreResult:
    _complete(end_of_life_cycles, 3)
    if sorted(temperatures_c) != [15, 25, 35]:
        raise ValueError("Expected the three native ambient temperatures")
    if any(c < 0 for c in end_of_life_cycles):
        raise ValueError("Cycle count cannot be negative")
    return ScoreResult(statistics.mean(end_of_life_cycles) / 2000 if valid else 0.0,
                       {"valid": valid, "native_unit": "mean_predicted_lifetime_div_2000"})


def aqm_score(runs: Sequence[Mapping], *, valid: bool) -> ScoreResult:
    if len(runs) != 6:
        raise ValueError("AQM native aggregate needs all six packet-level runs")
    values = []
    for row in runs:
        g, d, loss, fairness = (float(row[k]) for k in ["normalized_goodput", "delay_p95_seconds", "loss_rate", "jain_fairness"])
        if not all(math.isfinite(x) for x in (g, d, loss, fairness)) or d < 0 or not all(0 <= x <= 1 for x in (g, loss, fairness)):
            raise ValueError("AQM output is outside native metric bounds")
        values.append(g - d - 2 * loss - 0.5 * (1 - fairness))
    return ScoreResult(statistics.mean(values) if valid else 0.0, {"valid": valid, "native_unit": "aqm_utility"})


def fdtd_score(transmissions: Mapping, *, valid: bool) -> ScoreResult:
    values = [float(transmissions[k]) for k in ["T1A", "T2B", "T1B", "T2A"]]
    _complete(values, 4)
    if any(v < 0 for v in values):
        raise ValueError("Normalized transmitted power cannot be negative")
    return ScoreResult((0.5 * (values[0] + values[1]) - 0.25 * (values[2] + values[3])) if valid else 0.0,
                       {"valid": valid, "native_unit": "fdtd_routing_utility"})


def reactor_score(scenarios: Sequence[Mapping], *, valid: bool) -> ScoreResult:
    if len(scenarios) != 3:
        raise ValueError("Reactor native aggregate needs all three complete scenarios")
    values = [(float(row["n_C"]) - float(row["n_D"])) / 85 for row in scenarios]
    _complete(values, 3)
    if any(float(row[k]) < 0 for row in scenarios for k in ["n_C", "n_D"]):
        raise ValueError("Moles cannot be negative")
    return ScoreResult(statistics.mean(values) if valid else 0.0, {"valid": valid, "native_unit": "reactor_yield_utility"})


def irp_score(episodes: Sequence[Mapping], *, valid: bool, ddof: int) -> ScoreResult:
    """ddof must be registered: the source does not specify its SD convention."""
    if len(episodes) != 36 or ddof not in {0, 1}:
        raise ValueError("IRP needs 36 complete episodes and explicit ddof=0 or 1")
    values = []
    for episode in episodes:
        cost, demand = float(episode["total_cost"]), float(episode["total_demand"])
        if not math.isfinite(cost) or not math.isfinite(demand) or cost < 0 or demand <= 0:
            raise ValueError("Invalid IRP episode cost/demand")
        values.append(1 - cost / (12 * demand))
    sd = statistics.pstdev(values) if ddof == 0 else statistics.stdev(values)
    return ScoreResult(statistics.mean(values) - 0.15 * sd if valid else 0.0,
                       {"valid": valid, "native_unit": "irp_risk_adjusted_utility", "ddof": ddof})
