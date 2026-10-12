"""Full native U/P/U comparison and independent scientific consistency checks."""
from __future__ import annotations
from dataclasses import replace
import copy
import json
from pathlib import Path
import time

from specrsi.core.acquisition import ResultAcquirer,AcquisitionCaps
from specrsi.core.budget import BudgetLedger
from specrsi.core.ledger import Ledger
from specrsi.core.models import thaw
from .checker import ScienceParameterChecker
from .executor import ScienceExecutor,make_candidate,make_scope
from .registry import evaluate,load_config


def reuse_comparison(domain="reactor"):
    scope=make_scope(domain,case_ids=(0,1));tasks=scope.tasks
    source=make_candidate(domain)
    parameters=thaw(source.artifacts["parameters"])
    if domain=="reactor":parameters["feed_mol_per_second"][0]*=.9
    elif domain=="irp":parameters["target_days"]*=1.05
    elif domain=="aqm":parameters["max_probability"]*=1.1
    elif domain=="pybamm":parameters["currents_c"][0]*=1.05
    elif domain=="fdtd":parameters["upper_width_scale"]*=1.05
    child=make_candidate(domain,name="edit-case-zero",parent_id=source.uid,scenario_overrides={tasks[0].uid:parameters})
    checker=ScienceParameterChecker();ledger=Ledger(checker);budget=BudgetLedger(1000000)
    executor=ScienceExecutor();acquirer=ResultAcquirer(ledger,budget,executor)
    started=time.monotonic()
    source_records=[acquirer.acquire(source,t,scope,(),AcquisitionCaps(180000,0,1000,1)).record for t in tasks]
    before=budget.spent
    acquisitions=[acquirer.acquire(child,t,scope,(source,),AcquisitionCaps(180000,0,1000,1)) for t in tasks]
    verification=ledger.audit_provenance()
    specrsi_wall=time.monotonic()-started
    baseline=[];direct_start=time.monotonic()
    for task in tasks:
        result=executor.run(child,task,scope,180000)
        if result.outcome is None:raise RuntimeError(result.error)
        baseline.append({"actual_cost_ms":result.actual_cost,"outcome":result.outcome.manifest(),"receipt":thaw(result.receipt)})
    baseline_wall=time.monotonic()-direct_start
    if [a.record.route for a in acquisitions]!=["U","P"]:
        raise RuntimeError("Expected affected native case U and unaffected native case P")
    if [a.record.outcome.manifest() for a in acquisitions]!=[x["outcome"] for x in baseline]:
        raise RuntimeError("Full direct native reference disagrees with scoped reuse")
    return {"domain":domain,"case_ids":[0,1],"author_exact":False,"paper_budget_comparable":False,
            "budget_unit":"measured simulator/checker wall milliseconds; cooperative checker overruns are explicit errors",
            "controller_overhead":"provenance rechecks and bookkeeping are included in end-to-end elapsed, separately logged",
            "setup_native_records":[r.manifest() for r in source_records],"child_acquisitions":[a.record.manifest() for a in acquisitions],
            "child_charged_ms":budget.spent-before,"full_direct_charged_ms":sum(x["actual_cost_ms"] for x in baseline),
            "setup_and_scoped_elapsed_seconds":specrsi_wall,"full_direct_child_elapsed_seconds":baseline_wall,
            "physical_checker_calls":checker.physical_checks,"full_direct_child":baseline,"provenance_records":verification,
            "complete_scored_outcomes_equal":True}


def physics_controls():
    """Independent physical checks; reduced/extended discretizations are labeled."""
    import numpy as np
    from .reactor import simulate as reactor
    from .fdtd import simulate as fdtd
    results={}
    c=load_config("reactor");candidate=c["baseline"]
    regular=reactor(candidate,c)
    tighter=copy.deepcopy(c);tighter["solver"].update(rtol=1e-10,atol=1e-11,max_step_seconds=5)
    refined=reactor(candidate,tighter)
    error=abs(regular["native_score"]-refined["native_score"])
    if error>1e-5:raise RuntimeError("Reactor tolerance refinement failed")
    results["reactor_tolerance_refinement"]={"absolute_native_score_change":error,"threshold":1e-5}
    c=load_config("fdtd");symmetric=dict(c["baseline"],grating_depth=0)
    result=fdtd(symmetric,c)
    t=result["transmissions"]
    symmetry=max(abs(t["T1A"]-t["T1B"]),abs(t["T2A"]-t["T2B"]))
    if symmetry>1e-8:raise RuntimeError("Mirror-symmetric FDTD geometry breaks port symmetry")
    results["fdtd_mirror_symmetry"]={"max_transmission_difference":symmetry,"threshold":1e-8,"native":result}
    extended=copy.deepcopy(c);extended["steps"]=15000;extended["source_ramp_steps"]=1800
    original=fdtd(c["baseline"],c);longer=fdtd(c["baseline"],extended)
    delta=abs(original["native_score"]-longer["native_score"])
    if delta>0.03:raise RuntimeError("FDTD observation-time refinement changed native utility excessively")
    results["fdtd_time_refinement"]={"absolute_native_score_change":delta,"threshold":0.03,"registered_steps":10000,"control_steps":15000,
                                    "extended_run_is_diagnostic_only":True,"registered_native":original,"extended_native":longer}
    c=load_config("aqm")
    first=evaluate("aqm",c["baseline"]);second=evaluate("aqm",c["baseline"])
    if first["runs"]!=second["runs"]:raise RuntimeError("ns-3 candidate-independent seeds did not replay exactly")
    results["aqm_exact_packet_replay"]={"equal":True,"native_score":first["native_score"],"run_count":6}
    return results


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics",action="store_true")
    parser.add_argument("--domain",default="reactor")
    parser.add_argument("--output",type=Path,default=Path("reports/science/native_controls.json"))
    args=parser.parse_args()
    result=physics_controls() if args.physics else reuse_comparison(args.domain)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":True,"output":str(args.output)}))
