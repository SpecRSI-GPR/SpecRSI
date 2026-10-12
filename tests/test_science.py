from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from specrsi.core.models import Candidate,thaw
from specrsi.science import DOMAINS,evaluate,load_config,registration
from specrsi.science.common import InvalidCandidate
from specrsi.science.checker import ScienceParameterChecker
from specrsi.science.executor import ScienceExecutor,make_candidate,make_scope,selected_parameters


@pytest.mark.parametrize("domain",DOMAINS)
def test_registered_invalid_candidate_is_inadmissible(domain):
    result=evaluate(domain,load_config(domain)["invalid"])
    assert result["status"]=="invalid_candidate"
    assert result["valid"] is False and result["native_score"] is None
    assert result["specrsi_score"]==0 and result["violations"]


@pytest.mark.parametrize("domain",DOMAINS)
def test_distinct_frozen_scenario_sets(domain):
    roles=("development","selection","audit")
    manifests=[registration(domain,role) for role in roles]
    assert len({m["evaluator_identity"] for m in manifests})==3
    configurations=[]
    for role,manifest in zip(roles,manifests):
        config=manifest["config"]
        assert config["registration_version"]=="specrsi-prospective-science-v1"
        assert config["domain"]==domain and config["profile"]=="full"
        assert manifest["scenario_set"]==role
        assert config.get("scenario_set","selection")==role
        assert config["case_id"]==0
        # Distinct identities must reflect different registered numerical
        # scenarios, even after descriptive split/provenance fields are removed.
        configurations.append({key:value for key,value in config.items()
            if key not in {"scenario_set","source","author_exact","replication_scope"}})
    assert len({json.dumps(config,sort_keys=True) for config in configurations})==3
    assert all(m["source_sha256"]==manifests[0]["source_sha256"] for m in manifests)
    assert "aqm_native.cc" in manifests[0]["source_sha256"]
    assert f"{domain}.py" in manifests[0]["source_sha256"]
    assert all(len(value)==64 and all(character in "0123456789abcdef" for character in value)
               for value in manifests[0]["source_sha256"].values())


def test_reactor_material_conservation_and_thermal_failure():
    baseline=load_config("reactor")["baseline"]
    result=evaluate("reactor",baseline)
    assert result["valid"] and len(result["scenarios"])==3
    assert all(x["material_balance_residual_mol"]<1e-6 for x in result["scenarios"])
    dangerous=copy.deepcopy(baseline);dangerous["jacket_temperature_k"]=[330]*10
    rejected=evaluate("reactor",dangerous)
    assert not rejected["valid"]
    assert "temperature" in rejected["violations"][0]


def test_irp_real_demand_independent_and_conserved():
    c=load_config("irp")
    first=evaluate("irp",c["baseline"])
    second=evaluate("irp",dict(c["baseline"],target_days=0,safety_stock=0))
    assert first["valid"] and second["valid"]
    assert first["demand_sha256"]==second["demand_sha256"]
    assert first["native_score"]>second["native_score"]
    assert all(x["sold"]+x["lost"]==x["total_demand"] for x in first["episodes"])
    assert all(x["sold"]==0 for x in second["episodes"])


def test_irp_rejects_capacity_violation_from_policy(monkeypatch):
    from specrsi.science import irp
    monkeypatch.setattr(irp,"policy",lambda state,candidate:np.full(8,1000,dtype=np.int64))
    c=load_config("irp")
    with pytest.raises(InvalidCandidate,match="capacity"):
        irp.simulate(c["baseline"],c)


def test_science_projection_proof_and_real_acquisition():
    from specrsi.core.acquisition import ResultAcquirer,AcquisitionCaps
    from specrsi.core.budget import BudgetLedger
    from specrsi.core.ledger import Ledger
    scope=make_scope("reactor");task=scope.tasks[0]
    source=make_candidate("reactor")
    parameters=thaw(source.artifacts["parameters"])
    parameters["feed_mol_per_second"][0]*=.9
    unused=make_candidate("reactor",name="unused",parent_id=source.uid,scenario_overrides={"unused":parameters})
    used=make_candidate("reactor",name="used",parent_id=source.uid,scenario_overrides={task.uid:parameters})
    checker=ScienceParameterChecker();ledger=Ledger(checker);budget=BudgetLedger(90000)
    acquirer=ResultAcquirer(ledger,budget,ScienceExecutor())
    first=acquirer.acquire(source,task,scope,(),AcquisitionCaps(20000))
    reused=acquirer.acquire(unused,task,scope,(source,),AcquisitionCaps(20000,0,1000,1))
    changed=acquirer.acquire(used,task,scope,(source,),AcquisitionCaps(20000,0,1000,1))
    assert first.record.route=="U" and reused.record.route=="P" and changed.record.route=="U"
    assert reused.record.outcome==first.record.outcome
    assert changed.record.outcome.score!=first.record.outcome.score
    assert ledger.audit_provenance()==3
    assert "wall_seconds" not in first.record.outcome.fields
    # Timing is an execution receipt, so repeat native results do not conflict.
    repeat=ScienceExecutor().run(source,task,scope,20000)
    assert repeat.outcome==first.record.outcome


@pytest.mark.parametrize("mutation",["runtime","scorer","timing","arbitrary_code","nondeterministic","global_artifact","scope_replay","task_config"])
def test_science_proof_fails_closed_on_adversarial_changes(mutation):
    scope=make_scope("reactor");task=scope.tasks[0];source=make_candidate("reactor")
    child=replace(source,name="child")
    if mutation=="runtime":child=replace(child,runtime={"domain":"reactor","contract":"science-parameter-kernel-v999"})
    elif mutation=="scorer":scope=replace(scope,scorer_id="attacker-score")
    elif mutation=="timing":scope=replace(scope,scored_fields=(*scope.scored_fields,"wall_seconds"))
    elif mutation=="arbitrary_code":child=replace(child,semantic_kind="python")
    elif mutation=="nondeterministic":child=replace(child,deterministic=False)
    elif mutation=="global_artifact":child=replace(child,artifacts={**thaw(child.artifacts),"global_config":{"ignored":1}})
    elif mutation=="scope_replay":scope=replace(scope,replay={"seed":1})
    elif mutation=="task_config":
        task=replace(task,metadata={"evaluator_identity":"forged"});scope=replace(scope,tasks=(task,))
    assert ScienceParameterChecker().check(source,child,task,scope,1000).status=="U"


def test_timeout_never_supplies_a_partial_native_outcome():
    scope=make_scope("fdtd");candidate=make_candidate("fdtd")
    result=ScienceExecutor().run(candidate,scope.tasks[0],scope,1)
    assert result.status=="interrupted" and result.outcome is None and result.actual_cost==1


def test_stable_case_route_transfers_frozen_parameters_to_native_audit():
    source=make_candidate("reactor")
    parameters=thaw(source.artifacts["parameters"]);parameters["feed_mol_per_second"][0]*=.7
    child=make_candidate("reactor",name="stable-case",scenario_overrides={"case:0":parameters})
    before=child.uid
    selection=make_scope("reactor",case_ids=(0,1));audit=make_scope("reactor","audit",(0,1))
    checker=ScienceParameterChecker()
    for scope in (selection,audit):
        assert selected_parameters(child,scope.tasks[0],scope)==parameters
        assert checker.check(source,child,scope.tasks[0],scope,1000).status=="U"
        assert checker.check(source,child,scope.tasks[1],scope,1000).status=="P"
    baseline=ScienceExecutor().run(source,audit.tasks[0],audit,20000)
    changed=ScienceExecutor().run(child,audit.tasks[0],audit,20000)
    assert baseline.status==changed.status=="complete"
    assert baseline.outcome.admissible and changed.outcome.admissible
    assert baseline.outcome.score!=changed.outcome.score
    assert child.uid==before  # Audit used the exact frozen version without a remap.


@pytest.mark.parametrize("key",["case:-1","case:10","case:00","case:foo"])
def test_stable_routes_reject_unregistered_case_keys(key):
    scope=make_scope("reactor");source=make_candidate("reactor")
    child=make_candidate("reactor",scenario_overrides={key:thaw(source.artifacts["parameters"])})
    assert ScienceParameterChecker().check(source,child,scope.tasks[0],scope,1000).status=="U"
    assert ScienceExecutor().run(child,scope.tasks[0],scope,100).status=="error"


def test_stable_route_ambiguity_and_legacy_version_fail_closed():
    scope=make_scope("reactor");source=make_candidate("reactor");task=scope.tasks[0]
    parameters=thaw(source.artifacts["parameters"])
    ambiguous=make_candidate("reactor",scenario_overrides={"case:0":parameters,task.uid:parameters})
    with pytest.raises(ValueError,match="Ambiguous"):
        selected_parameters(ambiguous,task,scope)
    legacy=replace(source,runtime={"domain":"reactor","contract":"science-parameter-kernel-v1"})
    assert ScienceParameterChecker().check(legacy,source,task,scope,1000).status=="U"
