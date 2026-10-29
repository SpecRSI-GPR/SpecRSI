# Reactor: 12-instance CSTR profile

Use `scenario_set="reference"` for the three-state CSTR with the 12 instance definitions below. The implementation is in [`reactor_cstr.py`](../../src/specrsi/science/reactor_cstr.py), dispatched by [`reactor.py`](../../src/specrsi/science/reactor.py); the complete parameter file is [`configs/science/reference/reactor.json`](../../configs/science/reference/reactor.json).

Select `reference` for these 12 CSTR instances. The `selection`, `development`, and `audit` profiles select the six-state semi-batch evaluator and its candidate format.

## Run

From the repository root after installing the project dependencies:

```sh
PYTHONPATH=src python -m specrsi.science.reactor_cstr --all-cases --output reports/reactor_reference.json
PYTHONPATH=src python -m specrsi.science.reactor_cstr --case-id 10 --coolant-flow 28
python -m pytest -q tests/test_reactor_reference.py
```

Case IDs at the Python/CLI boundary are zero-based: 0 maps to `reactor_case_01`, and 11 maps to `reactor_case_12`. Each case has a default constant coolant flow. A candidate is exactly `{"coolant_flow_l_min": 28.0}`; the value must be finite and between 5 and 50 L/min. Candidate input cannot override the physical model, initial state, or safety limits.

```python
from specrsi.science.registry import evaluate, load_config

config = load_config("reactor", scenario_set="reference", case_id=10)
result = evaluate("reactor", config["baseline"], scenario_set="reference", case_id=10)
print(result["valid"], result["native_score"], result["trajectory"])
```

## Equations and units

Time is in minutes; concentration is mol/L; feed and coolant flows are L/min; gas amount is mol; temperature is K; pressure is MPa. Let `V = 100 L`, `V_head = 0.05 m³`, and `R = 8.314 J/(mol K)`.

```text
k(T) = 2.5e5 exp(-40000 / (R T))                  [1/min]
dC/dt = (F / V) (C_in - C) - k(T) C
r_gen = k(T) C V                                  [mol/min]
P = n R T / V_head × 1e-6                         [MPa]
r_out = 4 max(P - 1.2, 0) + 90 max(P - 2.8, 0)    [mol/min]
dn/dt = r_gen - r_out
C(0) = 0.2 C_in
```

The energy and coolant-outlet equations use explicit L-to-m³ conversion:

```text
V_m3 = 0.1
H = rho_liq cp_liq V_m3
T_c,out = T - (T - T_c,in) exp(-UA / ((F_c × 1e-3) rho_c cp_c))
dT/dt = (F / V) (T_in - T) - deltaH r_gen / H - UA (T - T_c,out) / H
```

| Constant | Value |
| --- | ---: |
| Reaction enthalpy `deltaH` | −45000 J/mol |
| `UA` | 1200 J/(min K) |
| Liquid density / heat capacity | 1000 kg/m³ / 4000 J/(kg K) |
| Coolant density / heat capacity | 1000 kg/m³ / 4186 J/(kg K) |
| Duration | 60 min |
| Reference coolant flow | 15 L/min |
| Solver | DOP853; `rtol=1e-10`, `atol=1e-11`, maximum step 0.1 min |

An admissible trajectory completes the horizon with finite, nonnegative physical states, temperature in [300, 400] K, and pressure in [1, 3] MPa. Safety-valve opening at 2.8 MPa is allowed. Boundary-crossing events and dense-output extrema checks assess admissibility over the complete trajectory.

## Score

`C_ref` is the outlet concentration from a 60-minute run at fixed coolant flow 15 L/min under the same instance conditions. The native dimensionless score is:

```text
q = (C_in - C_out) / (C_in - C_ref)
q = 0 if the denominator is zero
q_bounded = min(1, max(0, q))
f = 1 - F_c / 50
J = min(1, max(0, 0.8 q_bounded + 0.2 f))
```

The returned `native_score` is `J`; score components and the reference outlet concentration are included in the result. The registry's separate `specrsi_score` field is its generic bounded interface transform, not the native Reactor score. For execution through `ScienceExecutor`, `make_scope("reactor", "reference", ...)` selects native-utility aggregation by default; use `make_science_ledger()` to register this aggregation rule.

## Instance parameters

| Instance | F (L/min) | C_in (mol/L) | T_in (K) | T_c,in (K) | T(0) (K) | n(0) (mol) | Default F_c (L/min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reactor_case_01 | 1 | 1 | 298.15 | 288.15 | 320 | 24.80 | 10 |
| reactor_case_02 | 1.5 | 1.2 | 298.15 | 288.15 | 330 | 28.79 | 12 |
| reactor_case_03 | 2 | 1.5 | 300.15 | 290.15 | 340 | 32.71 | 15 |
| reactor_case_04 | 2 | 1.8 | 300.15 | 290.15 | 345 | 35.02 | 16 |
| reactor_case_05 | 2.5 | 2 | 303.15 | 293.15 | 355 | 37.87 | 18 |
| reactor_case_06 | 3 | 2.2 | 303.15 | 293.15 | 365 | 40.66 | 20 |
| reactor_case_07 | 2 | 2.5 | 305.15 | 295.15 | 350 | 37.41 | 22 |
| reactor_case_08 | 3 | 1.5 | 298.15 | 285.15 | 340 | 33.60 | 25 |
| reactor_case_09 | 1.5 | 2 | 305.15 | 298.15 | 335 | 31.62 | 14 |
| reactor_case_10 | 2.5 | 1.8 | 300.15 | 288.15 | 350 | 36.38 | 17 |
| reactor_case_11 | 3.5 | 2 | 308.15 | 298.15 | 370 | 42.97 | 28 |
| reactor_case_12 | 1 | 2.8 | 310.15 | 300.15 | 330 | 30.80 | 12 |

Initial pressures are calculated from the listed gas amounts and temperatures.

## Numerical checks

The 12 default 60-minute trajectories all satisfy the safety constraints. Their pressure range is 1.319598–2.919780 MPa; the maximum occurs in `reactor_case_11` at approximately 0.595561 min. Cases 05, 06, 07, and 11 activate the auxiliary valve. The 84 checks at constant coolant flows 5, 10, 15, 20, 30, 40, and 50 L/min also complete safely. DOP853 and an independent Radau calculation (maximum step 0.01 min) differ by less than 3e-10 MPa in default pressure peaks.

[Default trajectory checks (CSV)](data/reactor_reference_default_validation.csv) record the per-case extrema and valve activation. These trajectory checks evaluate fixed coolant controls. Multi-seed search outcomes are listed on the [manuscript results page](README.md).
