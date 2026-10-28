import copy
import math
import random

import pytest

from specrsi.core.models import thaw
from specrsi.science.common import InvalidCandidate
from specrsi.science.executor import ScienceExecutor, make_candidate, make_scope, mutate, selected_parameters
from specrsi.science.reactor import simulate
from specrsi.science.reactor_cstr import gas_outflow_mol_min, pressure_mpa
from specrsi.science.registry import case_count, evaluate, load_config, registration


# Independently computed before integration, with DOP853/Radau agreement.
EXPECTED_PEAKS = [1.473011991588305, 1.7127789549010832, 2.0964862771677795,
                  2.4068830932449417, 2.8192957116702444, 2.8977565032015793,
                  2.8322461114539252, 2.1475193622019653, 2.1750977741374404,
                  2.591187722756795, 2.9197800595490166, 2.321305861985743]


@pytest.mark.parametrize("case_id", range(12))
def test_all_author_cases_match_validated_transient_peaks(case_id):
    c = load_config("reactor", "reference", case_id)
    result = evaluate("reactor", c["baseline"], scenario_set="reference", case_id=case_id)
    assert result["valid"] and result["complete"]
    assert result["instance_id"] == f"reactor_case_{case_id+1:02d}"
    assert 0 <= result["native_score"] <= 1
    trajectory = result["trajectory"]
    assert trajectory["duration_min"] == 60
    assert trajectory["pressure_max_mpa"] == pytest.approx(EXPECTED_PEAKS[case_id], abs=2e-8)
    assert 1 <= trajectory["pressure_min_mpa"] <= trajectory["pressure_max_mpa"] < 2.95
    assert 300 <= trajectory["temperature_min_k"] <= trajectory["temperature_max_k"] <= 400
    assert trajectory["concentration_min_mol_l"] >= 0 and trajectory["gas_min_mol"] >= 0
    assert trajectory["safety_valve_opened"] == (case_id in {4,5,6,10})
    assert trajectory["pressure_max_mpa"] > trajectory["final_pressure_mpa"]


def test_pressure_units_and_dual_valve_thresholds():
    c = load_config("reactor", "reference", 10)
    assert pressure_mpa(42.97, 370.0, c) == pytest.approx(2.643669092)
    assert gas_outflow_mol_min(1.0, c) == 0
    assert gas_outflow_mol_min(1.2, c) == 0
    assert gas_outflow_mol_min(2.8, c) == pytest.approx(6.4)
    assert gas_outflow_mol_min(3.0, c) == pytest.approx(25.2)


@pytest.mark.parametrize("flow", [0, 4.99, 50.01, math.inf, math.nan, True, "15"])
def test_illegal_controls_are_rejected(flow):
    c = load_config("reactor", "reference", 0)
    with pytest.raises(InvalidCandidate):
        simulate({"coolant_flow_l_min":flow}, c)


def test_candidate_cannot_override_frozen_safety_rules():
    c = load_config("reactor", "reference", 10)
    with pytest.raises(InvalidCandidate, match="exactly"):
        simulate({"coolant_flow_l_min":28.0, "safety_valve_outflow_mol_min_mpa":900.0}, c)


def test_transient_pressure_failure_is_not_hidden_by_final_state():
    c = load_config("reactor", "reference", 10)
    c["safety_valve_outflow_mol_min_mpa"] = 30.0
    with pytest.raises(InvalidCandidate, match="pressure upper bound"):
        simulate(c["baseline"], c)


@pytest.mark.parametrize("field,value,reason", [
    ("initial_gas_mol", 1.0, "initial pressure"),
    ("initial_gas_mol", 100.0, "initial pressure"),
    ("initial_temperature_k", 299.0, "initial temperature"),
    ("initial_temperature_k", 401.0, "initial temperature"),
    ("feed_temperature_k", 1000.0, "temperature upper bound"),
])
def test_unsafe_trajectories_are_rejected(field, value, reason):
    c = load_config("reactor", "reference", 0)
    c[field] = value
    # Remove the pressure upper crossing in this targeted thermal-control test.
    if field == "feed_temperature_k":
        c["pressure_bounds_mpa"] = [0.0, 100.0]
    with pytest.raises(InvalidCandidate, match=reason):
        simulate(c["baseline"], c)


def test_fixed_flow_reference_and_score_definition():
    c = load_config("reactor", "reference", 0)
    result = simulate({"coolant_flow_l_min":15.0}, c)
    assert result["score_components"]["conversion_ratio"] == pytest.approx(1.0)
    assert result["score_components"]["cooling_efficiency"] == pytest.approx(0.7)
    assert result["native_score"] == pytest.approx(0.94)


@pytest.mark.parametrize("temperature, pressure, initial_fraction", [
    (320.0, 3.0, 0.2), (320.0, 1.0, 0.2),
    (300.0, 1.5, 0.2), (400.0, 1.5, 0.0),
])
def test_inclusive_safety_boundaries_allow_inward_initial_motion(temperature, pressure, initial_fraction):
    c = load_config("reactor", "reference", 0)
    c["initial_temperature_k"] = temperature
    c["initial_concentration_fraction"] = initial_fraction
    c["initial_gas_mol"] = pressure*1e6*c["headspace_volume_m3"] / (c["gas_constant_j_mol_k"]*temperature)
    result = simulate({"coolant_flow_l_min":15.0}, c)
    assert result["trajectory"]["duration_min"] == 60
    assert 1 <= result["trajectory"]["pressure_min_mpa"] <= result["trajectory"]["pressure_max_mpa"] <= 3
    assert 300 <= result["trajectory"]["temperature_min_k"] <= result["trajectory"]["temperature_max_k"] <= 400


@pytest.mark.parametrize("temperature", [300.0, 400.0])
def test_constant_temperature_exactly_on_inclusive_boundary(temperature):
    c = load_config("reactor", "reference", 0)
    c.update(initial_temperature_k=temperature, feed_temperature_k=temperature,
             coolant_inlet_temperature_k=temperature, reaction_enthalpy_j_mol=0.0,
             initial_concentration_fraction=0.0)
    result = simulate({"coolant_flow_l_min":15.0}, c)
    assert result["trajectory"]["temperature_min_k"] == temperature
    assert result["trajectory"]["temperature_max_k"] == temperature


@pytest.mark.parametrize("case_id", [10,11])
def test_complete_worker_path_accepts_new_registered_cases(case_id):
    scope = make_scope("reactor", "reference", (case_id,))
    assert scope.aggregation_id == "science_native_weighted_mean_v1"
    assert scope.split == "audit"
    assert scope.tasks[0].inputs["scenario_set"] == "reference"
    candidate = make_candidate("reactor", scenario_set="reference", case_id=case_id)
    result = ScienceExecutor().run(candidate, scope.tasks[0], scope, 30000)
    assert result.status == "complete", result.error
    assert result.outcome.admissible
    assert result.receipt["native_result"]["instance_id"] == f"reactor_case_{case_id+1:02d}"
    p = thaw(candidate.artifacts["parameters"])
    routed = make_candidate("reactor", parameters=p, scenario_overrides={f"case:{case_id}":p})
    assert selected_parameters(routed, scope.tasks[0], scope) == p


def test_unregistered_case_and_wrong_profile_fail_closed():
    assert case_count("reactor", "reference") == 12
    assert case_count("reactor") == 10
    for bad in [-1,12,True]:
        with pytest.raises(ValueError, match="case ID"):
            load_config("reactor", "reference", bad)
    with pytest.raises(ValueError):
        load_config("irp", "reference", 0)
    with pytest.raises(ValueError):
        case_count("reactor", "arbitrary")
    scope = make_scope("reactor", "reference", (0,))
    c = make_candidate("reactor", scenario_set="reference", scenario_overrides={"case:12":{"coolant_flow_l_min":15}})
    with pytest.raises(ValueError, match="Unregistered stable"):
        selected_parameters(c, scope.tasks[0], scope)


def test_source_and_configuration_are_part_of_evaluator_identity(monkeypatch):
    from specrsi.science import registry
    before = registration("reactor", "reference", 0)
    assert "reactor_cstr.py" in before["source_sha256"]
    original = registry.load_config
    def edited(*args, **kwargs):
        c = copy.deepcopy(original(*args, **kwargs))
        c["safety_valve_outflow_mol_min_mpa"] = 91.0
        return c
    monkeypatch.setattr(registry, "load_config", edited)
    assert registration("reactor", "reference", 0)["evaluator_identity"] != before["evaluator_identity"]


def test_new_and_legacy_mutation_schemas_remain_distinct():
    rng = random.Random(1)
    p = {"coolant_flow_l_min":15.0}
    for _ in range(100):
        p = mutate("reactor", p, rng)
        assert set(p) == {"coolant_flow_l_min"} and 5 <= p["coolant_flow_l_min"] <= 50
    legacy = load_config("reactor")["baseline"]
    assert set(mutate("reactor", legacy, rng)) == {"feed_mol_per_second", "jacket_temperature_k"}
