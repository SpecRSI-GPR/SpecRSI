"""Reference three-state CSTR with continuous backpressure and safety valves.

Time is in minutes, concentration in mol/L, flow in L/min and pressure in MPa.
The model uses the author-confirmed dynamic equations and instance parameters.
Numerical validation of this profile is separate from historical search results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from .common import InvalidCandidate, exact_keys, scalar


def pressure_mpa(gas_mol, temperature_k, c):
    return gas_mol * c["gas_constant_j_mol_k"] * temperature_k / c["headspace_volume_m3"] * 1e-6


def gas_outflow_mol_min(pressure, c):
    return (c["backpressure_outflow_mol_min_mpa"] * max(pressure-c["backpressure_mpa"], 0.0)
            + c["safety_valve_outflow_mol_min_mpa"] * max(pressure-c["safety_valve_pressure_mpa"], 0.0))


def reaction_rate_per_min(temperature, c):
    return c["k0_per_min"] * math.exp(-c["activation_energy_j_mol"] / (c["gas_constant_j_mol_k"] * temperature))


def _initial_state(c):
    return np.array([c["initial_concentration_fraction"] * c["feed_concentration_mol_l"],
                     c["initial_temperature_k"], c["initial_gas_mol"]], dtype=float)


def _valid_initial_state(y, c):
    if not np.all(np.isfinite(y)) or y[0] < 0 or y[2] < 0:
        raise InvalidCandidate("Nonphysical Reactor initial concentration or gas amount")
    tlo, thi = c["temperature_bounds_k"]
    plo, phi = c["pressure_bounds_mpa"]
    if not tlo <= y[1] <= thi:
        raise InvalidCandidate("Reactor initial temperature outside safety bounds")
    if not plo <= pressure_mpa(y[2], y[1], c) <= phi:
        raise InvalidCandidate("Reactor initial pressure outside safety bounds")


def _trajectory(coolant_flow, c):
    y0 = _initial_state(c)
    _valid_initial_state(y0, c)
    volume_l = c["liquid_volume_litres"]
    volume_m3 = volume_l * 1e-3
    heat_capacity = c["liquid_density_kg_m3"] * c["liquid_heat_capacity_j_kg_k"] * volume_m3
    flow_fraction = c["feed_flow_l_min"] / volume_l

    def rhs(t, y):
        concentration, temperature, gas = y
        rate = reaction_rate_per_min(temperature, c)
        generated = rate * concentration * volume_l
        pressure = pressure_mpa(gas, temperature, c)
        # Preserve the supplied coolant-outlet/energy equations, converting L to m3.
        coolant_out = temperature - (temperature-c["coolant_inlet_temperature_k"]) * math.exp(
            -c["ua_j_min_k"] / (coolant_flow*1e-3*c["coolant_density_kg_m3"]*c["coolant_heat_capacity_j_kg_k"]))
        d_temperature = (flow_fraction*(c["feed_temperature_k"]-temperature)
                         - c["reaction_enthalpy_j_mol"]*generated/heat_capacity
                         - c["ua_j_min_k"]*(temperature-coolant_out)/heat_capacity)
        return [flow_fraction*(c["feed_concentration_mol_l"]-concentration)-rate*concentration,
                d_temperature, generated-gas_outflow_mol_min(pressure, c)]

    tlo, thi = c["temperature_bounds_k"]
    plo, phi = c["pressure_bounds_mpa"]
    functions = [lambda t,y: phi-pressure_mpa(y[2],y[1],c),
                 lambda t,y: pressure_mpa(y[2],y[1],c)-plo,
                 lambda t,y: thi-y[1], lambda t,y: y[1]-tlo,
                 lambda t,y: y[0], lambda t,y: y[2]]
    names = ["pressure upper bound", "pressure lower bound", "temperature upper bound",
             "temperature lower bound", "nonnegative concentration", "nonnegative gas amount"]
    # An inclusive boundary is safe when the state stays on it or moves inward.
    # solve_ivp otherwise treats a constant-zero event as an immediate crossing.
    def directional_event(gap, index):
        def event(t, y):
            value = gap(t, y)
            if value == 0.0:
                dc, dt, dn = rhs(t, y)
                dp = c["gas_constant_j_mol_k"] / c["headspace_volume_m3"] * 1e-6 * (dn*y[1] + y[2]*dt)
                rates = (-dp, dp, -dt, dt, dc, dn)
                if rates[index] >= 0.0:
                    return np.finfo(float).eps
            return value
        return event
    functions = [directional_event(gap, i) for i, gap in enumerate(functions)]
    for event in functions:
        event.terminal, event.direction = True, -1
    def safety_open(t, y):
        return pressure_mpa(y[2], y[1], c)-c["safety_valve_pressure_mpa"]
    safety_open.terminal, safety_open.direction = False, 1
    solver = c["solver"]
    sol = solve_ivp(rhs, (0.0, c["duration_min"]), y0, method=solver["method"],
                    rtol=solver["rtol"], atol=solver["atol"], max_step=solver["max_step_min"],
                    dense_output=True, events=functions+[safety_open])
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise InvalidCandidate("Reactor integration did not converge to a finite trajectory")
    crossing = next((names[i] for i, times in enumerate(sol.t_events[:-1]) if len(times)), None)
    if crossing is not None and sol.t[-1] != c["duration_min"]:
        raise InvalidCandidate(f"Reactor trajectory crossed {crossing} at {sol.t[-1]:.9g} min")
    if sol.t[-1] != c["duration_min"]:
        raise InvalidCandidate("Reactor integration did not complete the full horizon")

    # Include every solver node and a dense 0.01-min grid; refine local extrema.
    # Hard-limit events run during integration, so a transient cannot be hidden by
    # reporting only the final state or coarse output samples.
    grid = np.unique(np.concatenate((sol.t, np.linspace(0, sol.t[-1],
        max(2, int(np.ceil(sol.t[-1]/solver["extrema_grid_step_min"]))+1)))))
    state = sol.sol(grid)
    if not np.all(np.isfinite(state)):
        raise InvalidCandidate("Nonfinite Reactor dense-output trajectory")

    def extremum(values, fn, sign):
        candidates = [(float(values[0]), float(grid[0])), (float(values[-1]), float(grid[-1]))]
        # Strict on one side avoids optimizing long constant plateaus.
        left, middle, right = sign*values[:-2], sign*values[1:-1], sign*values[2:]
        ids = np.flatnonzero(((middle < left) & (middle <= right)) |
                            ((middle <= left) & (middle < right)))+1
        for j in ids:
            fit = minimize_scalar(lambda t: sign*fn(sol.sol(t)), bounds=(grid[j-1],grid[j+1]),
                                  method="bounded", options={"xatol":1e-12})
            candidates.append((float(fn(sol.sol(fit.x))), float(fit.x)))
        return min(candidates, key=lambda item: sign*item[0])

    p_values = pressure_mpa(state[2], state[1], c)
    p_min, _ = extremum(p_values, lambda y: pressure_mpa(y[2],y[1],c), 1)
    p_max, p_peak_time = extremum(p_values, lambda y: pressure_mpa(y[2],y[1],c), -1)
    t_min, _ = extremum(state[1], lambda y: y[1], 1)
    t_max, _ = extremum(state[1], lambda y: y[1], -1)
    c_min, _ = extremum(state[0], lambda y: y[0], 1)
    n_min, _ = extremum(state[2], lambda y: y[2], 1)
    if c_min < 0 or n_min < 0:
        raise InvalidCandidate("Nonphysical negative Reactor concentration or gas amount")
    if p_min < plo or p_max > phi:
        raise InvalidCandidate("Reactor pressure extrema violate safety bounds")
    if t_min < tlo or t_max > thi:
        raise InvalidCandidate("Reactor temperature extrema violate safety bounds")
    opening_times = sol.t_events[-1]
    initially_open = pressure_mpa(y0[2],y0[1],c) > c["safety_valve_pressure_mpa"]
    opened = p_max > c["safety_valve_pressure_mpa"]
    return {"duration_min":float(sol.t[-1]), "outlet_concentration_mol_l":float(sol.y[0,-1]),
            "final_temperature_k":float(sol.y[1,-1]), "final_gas_mol":float(sol.y[2,-1]),
            "final_pressure_mpa":float(pressure_mpa(sol.y[2,-1],sol.y[1,-1],c)),
            "pressure_min_mpa":p_min, "pressure_max_mpa":p_max, "pressure_peak_time_min":p_peak_time,
            "temperature_min_k":t_min, "temperature_max_k":t_max,
            "concentration_min_mol_l":c_min, "gas_min_mol":n_min,
            "safety_valve_opened":bool(opened),
            "safety_valve_first_open_min":0.0 if initially_open else (float(opening_times[0]) if len(opening_times) else None),
            "nfev":sol.nfev}


def simulate(candidate: dict, c: dict) -> dict:
    exact_keys(candidate, ("coolant_flow_l_min",))
    flow = scalar(candidate["coolant_flow_l_min"], *c["coolant_flow_bounds_l_min"], "coolant_flow_l_min")
    trajectory = _trajectory(flow, c)
    reference_flow = c["reference_coolant_flow_l_min"]
    reference = trajectory if flow == reference_flow else _trajectory(reference_flow, c)
    denominator = c["feed_concentration_mol_l"]-reference["outlet_concentration_mol_l"]
    q = 0.0 if denominator == 0.0 else (
        c["feed_concentration_mol_l"]-trajectory["outlet_concentration_mol_l"])/denominator
    bounded_q = min(1.0, max(0.0, q))
    efficiency = 1.0-flow/c["coolant_flow_bounds_l_min"][1]
    score = min(1.0, max(0.0, 0.8*bounded_q+0.2*efficiency))
    return {"native_score":score, "native_unit":"dimensionless", "model_id":c["model_id"],
            "instance_id":c["instance_id"], "coolant_flow_l_min":flow, "trajectory":trajectory,
            "score_components":{"conversion_ratio":q, "bounded_conversion_ratio":bounded_q,
                                "cooling_efficiency":efficiency,
                                "reference_coolant_flow_l_min":reference_flow,
                                "reference_outlet_concentration_mol_l":reference["outlet_concentration_mol_l"]},
            "simulator":"scipy.integrate.solve_ivp three-state CSTR with dual pressure valves"}


def main():
    """Run a default case, a specified constant flow, or the complete 12-case set."""
    import argparse
    from .registry import case_count, evaluate, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, help="Zero-based registered case ID (0 through 11)")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--coolant-flow", type=float, help="Constant coolant flow in L/min; default is per-case")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.all_cases and args.case_id is not None:
        parser.error("Choose either --all-cases or --case-id")
    ids = range(case_count("reactor", "reference")) if args.all_cases else [0 if args.case_id is None else args.case_id]
    results = []
    for case_id in ids:
        c = load_config("reactor", "reference", case_id)
        candidate = c["baseline"] if args.coolant_flow is None else {"coolant_flow_l_min":args.coolant_flow}
        results.append(evaluate("reactor", candidate, scenario_set="reference", case_id=case_id))
    payload = {"scenario_set":"reference", "model_id":"reference-cstr-dual-valve-v1", "results":results}
    encoded = json.dumps(payload, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps({"cases":len(results), "valid_cases":sum(x["valid"] for x in results), "output":str(args.output)}))
    else:
        print(encoded, end="")
    if not all(x["valid"] for x in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
