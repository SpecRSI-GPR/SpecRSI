"""Six-state stiff material/energy balances for a semibatch chemical reactor."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from specrsi.benchmarks.science import reactor_score
from .common import InvalidCandidate, exact_keys, vector


def simulate(candidate: dict, c: dict) -> dict:
    exact_keys(candidate, ("feed_mol_per_second", "jacket_temperature_k"))
    n = c["segments"]
    feeds = vector(candidate["feed_mol_per_second"], n, *c["feed_bounds_mol_second"], "feed")
    jackets = vector(candidate["jacket_temperature_k"], n, *c["jacket_bounds_k"], "jacket")
    interval = c["duration_seconds"] / n
    total_feed = sum(feeds) * interval
    if total_feed > c["total_feed_limit_mol"] + 1e-10:
        raise InvalidCandidate("Total B feed exceeds 85 mol")
    rows = []
    for factor in c["scenario_cooling_factors"]:
        y = np.array([c["initial_moles_A"], 0., 0., 0., c["initial_volume_litres"], c["initial_temperature_k"]])
        max_temp, min_moles, nfev = y[5], 0., 0
        for segment, (feed, jacket) in enumerate(zip(feeds, jackets)):
            def rhs(t, z):
                a, b, product, waste, volume, temp = z
                k1 = c["k1_ref_litre_mol_second"] * np.exp(c["activation_1_j_mol"] / 8.314462618 * (1 / c["reference_temperature_k"] - 1 / temp))
                k2 = c["k2_ref_per_second"] * np.exp(c["activation_2_j_mol"] / 8.314462618 * (1 / c["reference_temperature_k"] - 1 / temp))
                rate1, rate2 = k1 * a * b / volume, k2 * product
                flow = feed / c["feed_concentration_mol_per_litre"]
                heat = -c["reaction_enthalpy_1_j_mol"] * rate1 - c["reaction_enthalpy_2_j_mol"] * rate2
                cooling = factor * c["cooling_ua_w_k"] * (temp - jacket)
                inlet = c["heat_capacity_j_litre_k"] * flow * (c["feed_temperature_k"] - temp)
                return [-rate1, feed-rate1, rate1-rate2, rate2, flow,
                        (heat - cooling + inlet) / (c["heat_capacity_j_litre_k"] * volume)]
            def unsafe(t, z):
                return c["temperature_limit_k"] - z[5]
            unsafe.terminal, unsafe.direction = True, -1
            sol = solve_ivp(rhs, (segment * interval, (segment + 1) * interval), y,
                            method=c["solver"]["method"], rtol=c["solver"]["rtol"], atol=c["solver"]["atol"],
                            max_step=c["solver"]["max_step_seconds"], events=unsafe)
            nfev += sol.nfev
            if not sol.success or len(sol.t_events[0]) or not np.all(np.isfinite(sol.y)):
                raise InvalidCandidate("Reactor integration failed or temperature constraint crossed")
            max_temp = max(max_temp, float(sol.y[5].max()))
            min_moles = min(min_moles, float(sol.y[:4].min()))
            if min_moles < -1e-7 or np.any(sol.y[4] <= 0):
                raise InvalidCandidate("Nonphysical concentration/volume")
            y = sol.y[:, -1]
        # Elemental A and B balances independently check the numerical solution.
        balance_a = abs(y[0] + y[2] + y[3] - c["initial_moles_A"])
        balance_b = abs(y[1] + y[2] + y[3] - total_feed)
        if max(balance_a, balance_b) > 1e-5:
            raise InvalidCandidate("Material balance residual exceeds tolerance")
        rows.append({"cooling_factor": factor, "n_A": float(y[0]), "n_B": float(y[1]),
                     "n_C": float(y[2]), "n_D": float(y[3]), "volume_litres": float(y[4]),
                     "max_temperature_k": max_temp, "material_balance_residual_mol": max(balance_a, balance_b), "nfev": nfev})
    result = reactor_score(rows, valid=True)
    return {"native_score": result.score, "native_unit": result.metrics["native_unit"], "scenarios": rows,
            "simulator": "scipy.integrate.solve_ivp/Radau six-state mass-energy ODE"}
