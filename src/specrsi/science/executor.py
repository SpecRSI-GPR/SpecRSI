"""Frozen host-owned scientific evaluator for the SpecRSI execution contract."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from specrsi.core.budget import cost_int
from specrsi.core.execution import ExecutionResult
from specrsi.core.models import Candidate,EvaluationScope,Outcome,Task,thaw
from specrsi.core.search import DraftResult
from .registry import ROOT,load_config,registration


def make_task(domain: str,scenario_set: str="selection",case_id: int=0) -> Task:
    identity=registration(domain,scenario_set,case_id)["evaluator_identity"]
    return Task(f"{domain}-complete-native-{scenario_set}-{case_id}",{"domain":domain,"scenario_set":scenario_set,"case_id":case_id},
                metadata={"evaluator_identity":identity,"unit":"one complete native scenario suite"})


def make_scope(domain: str,scenario_set: str="selection",case_ids: tuple[int,...]=(0,), *, native_utility: bool|None=None) -> EvaluationScope:
    from specrsi.core.models import digest
    if native_utility is None:
        native_utility = scenario_set == "reference"
    tasks=tuple(make_task(domain,scenario_set,i) for i in case_ids)
    identity=digest([task.metadata["evaluator_identity"] for task in tasks])
    score_identity = "native-utility-v1" if native_utility else "native-atan-v1"
    return EvaluationScope(f"science-{domain}-{scenario_set}",tasks,f"science-v2:{identity}",f"{score_identity}:{identity}",
                           scored_fields=("native_score","valid","evaluator_identity"),
                           aggregation_id="science_native_weighted_mean_v1" if native_utility else "weighted_mean_v1",
                           constraints_id="native-complete-and-all-hard-constraints-v1",
                           split={"development":"dev", "reference":"audit"}.get(scenario_set,scenario_set))


def native_science_utility(outcomes, scope):
    """Preserve scientific gain units; arctan-normalized item scores are not J."""
    return math.fsum(w * float(outcome.fields["native_score"]) for w, outcome in zip(scope.weights, outcomes))


def make_science_ledger(checker=None, *, receipt_secret=None):
    from specrsi.core.ledger import Ledger
    return Ledger(checker, receipt_secret=receipt_secret,
                  aggregators={"science_native_weighted_mean_v1": native_science_utility})


def make_candidate(domain: str,parameters: dict|None=None,name: str="baseline",parent_id=None,scenario_overrides: dict|None=None, *, scenario_set: str="selection", case_id: int=0) -> Candidate:
    return Candidate(name,{"parameters":load_config(domain,scenario_set,case_id)["baseline"] if parameters is None else parameters,
                           "scenario_overrides":scenario_overrides or {}},{"domain":domain,"contract":"science-parameter-kernel-v2"},
                     semantic_kind="scientific_parameters",deterministic=True,parent_id=parent_id)


def selected_parameters(candidate: Candidate,task: Task,scope: EvaluationScope) -> dict:
    """Trusted projection: candidate routes can select parameters, never code."""
    from collections.abc import Mapping
    from specrsi.core.models import FrozenMap
    if not candidate.deterministic or candidate.semantic_kind!="scientific_parameters":
        raise ValueError("Unsupported scientific candidate semantics")
    if set(task.inputs)!={"domain","scenario_set","case_id"} or not scope.contains(task):
        raise ValueError("Unregistered scientific task")
    domain,scenario_set=task.inputs["domain"],task.inputs["scenario_set"]
    if candidate.runtime!=FrozenMap({"domain":domain,"contract":"science-parameter-kernel-v2"}):
        raise ValueError("Unknown scientific runtime effects")
    case_ids=tuple(t.inputs["case_id"] for t in scope.tasks)
    if scope.uid!=make_scope(domain,scenario_set,case_ids,
            native_utility=scope.aggregation_id=="science_native_weighted_mean_v1").uid:
        raise ValueError("Scientific evaluator/config/source/scored-effect registry changed")
    if set(candidate.artifacts)!={"parameters","scenario_overrides"}:
        raise ValueError("Unmodeled global scientific candidate effects")
    overrides=candidate.artifacts["scenario_overrides"]
    if not isinstance(overrides,Mapping) or any(not isinstance(k,str) or not isinstance(v,Mapping) for k,v in overrides.items()):
        raise ValueError("Scientific overrides must map stable case keys or legacy task IDs to complete parameter objects")
    from .registry import case_count
    stable_keys={f"case:{i}" for i in range(case_count(domain,scenario_set))}
    if any(k.startswith("case:") and k not in stable_keys for k in overrides):
        raise ValueError("Unregistered stable scientific case route")
    stable_key=f"case:{task.inputs['case_id']}"
    if stable_key in overrides and task.uid in overrides:
        raise ValueError("Ambiguous stable-case and legacy task-UID overrides")
    # Stable case identity is fixed before evaluation and persists across role-specific
    # physical inputs. No audit observation or candidate rewrite participates here.
    parameters=overrides.get(stable_key,overrides.get(task.uid,candidate.artifacts["parameters"]))
    if not isinstance(parameters,Mapping):
        raise ValueError("Scientific parameters must be an immutable object")
    return thaw(parameters)


class ScienceExecutor:
    """Cap unit is wall milliseconds of the subprocess evaluation slice.

    Complete results charge measured elapsed time rounded up. At a timeout the
    reserved slice is exhausted; process-group cleanup overhead is separately
    receipted, and no partial native result can become a scored Outcome.
    Candidate data is never interpreted as source code or a simulator setting.
    """
    parallel_safe = True  # Each call owns a distinct native worker process.

    def run(self,candidate,task,scope,cap,resume_token=None):
        cost_int(cap)
        if resume_token is not None:
            return ExecutionResult("error",0,error="Science evaluations restart; resume is unsupported")
        if cap==0:
            return ExecutionResult("interrupted",0,error="No simulator wall-time slice available")
        try:
            parameters=selected_parameters(candidate,task,scope)
        except (KeyError,ValueError,TypeError) as exc:
            return ExecutionResult("error",0,error=str(exc))
        domain,scenario_set=task.inputs["domain"],task.inputs["scenario_set"]
        request={"domain":domain,"scenario_set":scenario_set,"case_id":task.inputs["case_id"],"candidate":parameters}
        # No inherited hosted credentials, CUDA devices or arbitrary Python paths.
        env={"PATH":"/usr/local/bin:/usr/bin:/bin","HOME":"/tmp","PYTHONPATH":str(ROOT/"src"),
             "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMBA_NUM_THREADS":"1",
             "PYBAMM_DISABLE_TELEMETRY":"true","CUDA_VISIBLE_DEVICES":"","NUMBA_CACHE_DIR":"/opt/specrsi-science/numba-cache"}
        # The worker owns a new session, so the campaign keeper needs these two
        # exact supervision tags to recover it after its parent exits. They are
        # process-ownership metadata, never evaluator inputs or score fields.
        # Keep the remaining environment whitelist closed to inherited values.
        for key in ("SPECRSI_JOB_DIR","SPECRSI_JOB_TOKEN"):
            if key in os.environ:
                env[key]=os.environ[key]
        started=time.monotonic()
        process=subprocess.Popen([sys.executable,"-m","specrsi.science.worker"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,text=True,env=env,start_new_session=True,cwd=ROOT)
        try:
            stdout,stderr=process.communicate(json.dumps(request),timeout=cap/1000)
        except subprocess.TimeoutExpired:
            stopped=time.monotonic()
            os.killpg(process.pid,signal.SIGKILL)
            process.communicate()
            return ExecutionResult("interrupted",cap,receipt={"cost_unit":"simulator-wall-milliseconds",
                  "cleanup_seconds":time.monotonic()-stopped,"wall_seconds":time.monotonic()-started},error="Simulator cap exhausted; no partial score")
        elapsed=time.monotonic()-started
        cost=math.ceil(elapsed*1000)
        if cost>cap:
            return ExecutionResult("interrupted",cap,receipt={"wall_seconds":elapsed,"boundary_overhead_ms":cost-cap},
                                   error="Complete output arrived after the reserved deadline; not admitted")
        if process.returncode:
            return ExecutionResult("error",cost,receipt={"stderr":stderr[-2000:]},error="Native simulator process failed")
        try:
            result=json.loads(stdout)
            if result["evaluator_identity"]!=task.metadata["evaluator_identity"] or not result["complete"]:
                raise ValueError("Incomplete or mismatched native evaluator output")
            fields={k:result[k] for k in ("native_score","valid","evaluator_identity")}
            outcome=Outcome(result["specrsi_score"],fields,admissible=result["valid"])
        except (ValueError,KeyError,TypeError) as exc:
            return ExecutionResult("error",cost,error=f"Malformed native result: {exc}")
        return ExecutionResult("complete",cost,outcome,receipt={"cost_unit":"simulator-wall-milliseconds","wall_seconds":elapsed,
                               "cpu_seconds":result["cpu_seconds"],"native_result":result,"stderr":stderr[-2000:]})


def mutate(domain: str,parameters: dict,rng: random.Random) -> dict:
    p=json.loads(json.dumps(parameters))
    if domain=="reactor" and set(p)=={"coolant_flow_l_min"}:
        p["coolant_flow_l_min"]=min(50.0,max(5.0,p["coolant_flow_l_min"]+rng.uniform(-4.5,4.5)))
    elif domain=="reactor":
        key=rng.choice(["feed_mol_per_second","jacket_temperature_k"])
        j=rng.randrange(10);p[key][j]*=rng.uniform(.9,1.1)
        if key=="jacket_temperature_k":p[key][j]=min(330,max(275,p[key][j]))
    elif domain=="pybamm":
        j=rng.randrange(3);p["currents_c"][j]=min(8,max(.1,p["currents_c"][j]*rng.uniform(.9,1.1)))
    elif domain=="irp" and set(p)=={"target_stock"}:
        # The frozen evaluator enforces each store's instance-specific capacity.
        j=rng.randrange(len(p["target_stock"]))
        p["target_stock"][j]=max(0,p["target_stock"][j]+rng.choice((-5,-1,1,5)))
    elif domain=="irp":
        key=rng.choice(["target_days","safety_stock","forecast_blend"])
        bounds={"target_days":(0,4),"safety_stock":(0,30),"forecast_blend":(0,1)}
        low,high=bounds[key];p[key]=min(high,max(low,p[key]+rng.uniform(-.15,.15)*(high-low)))
    elif domain=="aqm":
        p["delay_gain"]=rng.uniform(0,2)
        p["max_probability"]=min(1,max(.001,p["max_probability"]*rng.uniform(.7,1.3)))
    elif domain=="fdtd":
        key=rng.choice(["upper_width_scale","lower_width_scale","grating_depth","bend_exponent"])
        low,high={"upper_width_scale":(.6,1.4),"lower_width_scale":(.6,1.4),"grating_depth":(0,.7),"bend_exponent":(.5,3)}[key]
        p[key]=min(high,max(low,p[key]+rng.uniform(-.1,.1)*(high-low)))
    return p


class ParameterDrafter:
    """Executable seeded parameter proposal control; zero language-model calls."""
    def draft(self,parent,state,cap,rng):
        if cap<1:
            return DraftResult(None,0,status="interrupted",error="No draft slice")
        start=time.monotonic()
        domain=parent.runtime["domain"]
        child=make_candidate(domain,mutate(domain,thaw(parent.artifacts["parameters"]),rng),f"edit-{len(state.candidates)}",parent.uid)
        cost=max(1,math.ceil((time.monotonic()-start)*1000))
        if cost>cap:
            return DraftResult(None,cap,status="interrupted",error="Draft slice exhausted")
        return DraftResult(child,cost)
