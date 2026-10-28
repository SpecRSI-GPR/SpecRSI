import json

import numpy as np
import pytest

from specrsi.science import evaluate, load_config
from specrsi.science import fdtd
from specrsi.science.common import InvalidCandidate


def stub_fields(monkeypatch, powers):
    monkeypatch.setattr(fdtd, "design", lambda candidate, c: np.ones((c["nx"], c["ny"])))
    runs = iter(powers)

    def solve(epsilon, c, wavelength, reference=False):
        return {"flux": [1., 1.] if reference else next(runs)}

    monkeypatch.setattr(fdtd, "solve_field", solve)


def test_crosstalk_boundary_uses_each_wavelength_target(monkeypatch):
    stub_fields(monkeypatch, [[1., .01], [.005, .5]])
    c = load_config("fdtd")
    result = fdtd.simulate(c["baseline"], c)
    assert [r["crosstalk_db"] for r in result["wavelength_runs"]] == [-20., -20.]
    assert result["native_score"] == pytest.approx(.5*(1.+.5)-.25*(.01+.005))


@pytest.mark.parametrize("powers", [
    [[1., .0101], [0., 1.]],
    [[1., 0.], [.0101, 1.]],
    [[.005, .5], [1., .01]],
])
def test_one_failing_wavelength_is_inadmissible(monkeypatch, powers):
    stub_fields(monkeypatch, powers)
    result = evaluate("fdtd", load_config("fdtd")["baseline"])
    assert result["valid"] is False
    assert result["native_score"] is None
    assert "crosstalk" in result["violations"][0]


@pytest.mark.parametrize("powers", [
    [[0., 0.], [0., 1.]],
    [[1., 0.], [0., 0.]],
    [[0., 1.], [0., 1.]],
])
def test_zero_target_power_cannot_pass(monkeypatch, powers):
    stub_fields(monkeypatch, powers)
    c = load_config("fdtd")
    with pytest.raises(InvalidCandidate, match="target-port power"):
        fdtd.simulate(c["baseline"], c)


def test_zero_leakage_passes_without_nonfinite_json(monkeypatch):
    stub_fields(monkeypatch, [[.8, 0.], [0., .7]])
    c = load_config("fdtd")
    result = fdtd.simulate(c["baseline"], c)
    assert all(r["crosstalk_ratio"] == 0 for r in result["wavelength_runs"])
    assert all(r["crosstalk_db"] is None for r in result["wavelength_runs"])
    json.dumps(result, allow_nan=False)


def test_symmetric_field_diagnostic_does_not_imply_admissibility(monkeypatch):
    c = load_config("fdtd")
    stub_fields(monkeypatch, [[.4, .4], [.3, .3]])
    diagnostic = fdtd._simulate_fields(c["baseline"], c)
    assert diagnostic["transmissions"]["T1A"] == diagnostic["transmissions"]["T1B"]
    stub_fields(monkeypatch, [[.4, .4], [.3, .3]])
    with pytest.raises(InvalidCandidate, match="crosstalk"):
        fdtd.simulate(c["baseline"], c)
