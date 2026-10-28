import copy

import numpy as np
import pytest

from specrsi.science import irp, load_config
from specrsi.science.common import InvalidCandidate


def deterministic_case(monkeypatch, demands, shipments):
    c = copy.deepcopy(load_config("irp"))
    c.update(days=1, forecast_days=1, retailers=1, shelf_life_days=1,
             store_capacities=[10000], vehicle_capacity=10000,
             depot=[0., 0.], locations=[[1., 0.]])

    def demand(config, seed, episode):
        data = np.array([[demands[episode]], [demands[episode]]], dtype=np.int64)
        return data, data.astype(float)

    deliveries = iter(shipments)
    monkeypatch.setattr(irp, "demand_episode", demand)
    monkeypatch.setattr(irp, "policy", lambda state, candidate: np.array([next(deliveries)], dtype=np.int64))
    return c


def test_service_level_accepts_exact_95_percent(monkeypatch):
    c = deterministic_case(monkeypatch, [100]*36, [95]*36)
    result = irp.simulate(c["baseline"], c)
    assert result["service_level"] == .95
    assert all(r["sold"]+r["lost"] == r["total_demand"] for r in result["episodes"])


def test_service_level_rejects_one_unit_below_threshold(monkeypatch):
    c = deterministic_case(monkeypatch, [100]*36, [94]+[95]*35)
    with pytest.raises(InvalidCandidate, match="service level.*below 95%"):
        irp.simulate(c["baseline"], c)


def test_service_level_pools_demand_not_episode_rates(monkeypatch):
    # A 0%-served small episode does not outweigh a fully served large episode.
    c = deterministic_case(monkeypatch, [1, 100]*18, [0, 100]*18)
    result = irp.simulate(c["baseline"], c)
    assert result["service_level"] == pytest.approx(100/101)


def test_high_episode_average_cannot_hide_large_unmet_demand(monkeypatch):
    c = deterministic_case(monkeypatch, [1000]+[1]*35, [0]+[1]*35)
    with pytest.raises(InvalidCandidate, match="service level"):
        irp.simulate(c["baseline"], c)


def test_no_demand_cannot_produce_service_ratio(monkeypatch):
    c = deterministic_case(monkeypatch, [0]*36, [0]*36)
    with pytest.raises(InvalidCandidate, match="undefined without demand"):
        irp.simulate(c["baseline"], c)
