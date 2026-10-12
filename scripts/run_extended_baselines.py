#!/usr/bin/env python3
"""Run registered HGM/AFlow/Janus/EMAS adaptations with native local validation."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import yaml
from specrsi.baselines.extended_common import Design,ExperimentBudget,RunContext
from specrsi.baselines.workflow_backend import WorkflowBackend
from specrsi.benchmarks import load_examples,validate_disjoint_splits


def run_one(config):
    from specrsi.baselines.hgm import HGMConfig,run_hgm
    from specrsi.baselines.aflow import AFlowConfig,run_aflow
    from specrsi.baselines.janus import JanusConfig,run_janus
    from specrsi.baselines.emas import EMASConfig,run_emas
    output=Path(config["output"])
    output.mkdir(parents=True,exist_ok=True)
    (output/"resolved_config.json").write_text(json.dumps(config,indent=2))
    features=None
    provenance={}
    if config.get("backend_factory"):
        module,function=config["backend_factory"].split(":",1)
        bundle=getattr(importlib.import_module(module),function)(config)
        backend=bundle["backend"]
        parts={split:tuple(bundle[split]) for split in ("development","selection","audit")}
        initial=bundle["initial"]
        features=bundle.get("features")
        provenance=bundle.get("provenance",{})
        identities=[{RunContext.item_id(x) for x in part} for part in parts.values()]
        if any(identities[i]&identities[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError("Backend factory supplied overlapping development/selection/audit IDs")
    else:
        backend=WorkflowBackend(config["model"])
        dataset=config["dataset"]
        limits=config.get("data",{})
        parts={split:load_examples(dataset,split,data_root=limits.get("root","data"))[:limits.get(split,4)]
               for split in ("development","selection","audit")}
        validate_disjoint_splits(parts)
        initial=Design({"workflow":config["initial_workflow"]},label="initial")
    if any(not x for x in parts.values()):raise ValueError("All explicit development/selection/audit splits must be populated")
    budget=ExperimentBudget(**config.get("budget",{}))
    context=RunContext(backend,parts["selection"],seed=config.get("seed",0),budget=budget)
    method=config["method"]
    started=time.monotonic()
    try:
        options=config.get("algorithm",{})
        if method=="hgm":result=run_hgm(context,initial,HGMConfig(**options))
        elif method=="aflow":result=run_aflow(context,initial,AFlowConfig(**options))
        elif method=="janus":result=run_janus(context,initial,JanusConfig(**options),**({"features":features} if features else {}))
        elif method=="emas":result=run_emas(context,initial,parts["development"],EMASConfig(**options))
        else:raise ValueError("Unknown method")
        selected=context.designs[result["selected"]["id"]]
        # Audit is invoked only after the final immutable candidate is fixed.
        (output/"selected.json").write_text(json.dumps(selected.to_dict(),indent=2))
        audit=backend.evaluate(selected,parts["audit"],config.get("audit_seed",10000),"audit",config.get("audit_token_cap",100000))
        result.update({"audit":asdict(audit),"audit_score":audit.score if audit.complete else None,
            "elapsed_seconds":time.monotonic()-started,"config":config,"model":getattr(backend,"health",None),"backend_provenance":provenance,
            "physical_gpu_seconds":sum(c.get("elapsed_seconds",0.)*c.get("gpu_count",1) for c in getattr(backend,"calls",[])
                if "gpu_count" in c or "request_id" in c),
            "paper_budget_comparable":False,"source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("src/specrsi/baselines").glob("*.py")}})
        (output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
        return result
    except Exception as exc:
        (output/"failure.json").write_text(json.dumps({"valid_run":False,"error_type":type(exc).__name__,"error":str(exc),
            "budget":asdict(budget),"events":context.events},indent=2,ensure_ascii=False))
        raise
    finally:
        (output/"physical_calls.json").write_text(json.dumps(getattr(backend,"calls",[]),indent=2,ensure_ascii=False))
        if hasattr(backend,"native_receipts"):
            (output/"native_receipts.json").write_text(json.dumps(backend.native_receipts,indent=2,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--matrix")
    parser.add_argument("--plan",action="store_true",help="Resolve the matrix/config without model calls")
    parser.add_argument("--output")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--ablation",choices=["aflow_uniform","hgm_no_descendants","janus_fixed_proxy","janus_no_proxy","emas_no_recurrence","emas_no_validation"])
    args=parser.parse_args()
    def apply_ablation(config,ablation):
        method,setting={"aflow_uniform":("aflow",{"uniform_mixture":1.}),
            "hgm_no_descendants":("hgm",{"use_descendants":False}),
            "janus_fixed_proxy":("janus",{"evolve_evaluators":False}),"janus_no_proxy":("janus",{"proxy_guidance":False}),
            "emas_no_recurrence":("emas",{"recurrence":False}),"emas_no_validation":("emas",{"paired_validation":False})}[ablation]
        if config["method"]!=method:parser.error("Ablation does not match configured baseline")
        config.setdefault("algorithm",{}).update(setting)
        config["prospective_ablation"]=ablation
    configs=[]
    if args.matrix:
        matrix=yaml.safe_load(Path(args.matrix).read_text())
        for job in matrix["jobs"]:
            for seed in job.get("seeds",matrix.get("seeds",[0])):
                config=yaml.safe_load(Path(job["config"]).read_text())
                config["seed"]=seed
                if job.get("ablation"):apply_ablation(config,job["ablation"])
                name=job.get("ablation",config["method"])
                config["output"]=str(Path(args.output or matrix.get("output_root","results/extended_matrix"))/name/f"seed_{seed}")
                configs.append(config)
    else:
        config=yaml.safe_load(Path(args.config).read_text())
        if args.output:config["output"]=args.output
        if args.seed is not None:config["seed"]=args.seed
        if args.ablation:apply_ablation(config,args.ablation)
        configs=[config]
    if args.plan:
        print(json.dumps(configs,indent=2));return
    for config in configs:
        result=run_one(config)
        print(json.dumps({"method":result["method"],"implementation":result["implementation"],"audit_score":result["audit_score"],
            "budget":result["budget"],"output":config["output"]},indent=2))


if __name__=="__main__":main()
