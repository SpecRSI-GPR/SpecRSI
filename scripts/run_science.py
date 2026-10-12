#!/usr/bin/env python3
"""Execute complete registered native scenarios. No surrogate or hosted API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor,as_completed

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from specrsi.science import DOMAINS,evaluate,load_config,registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain",choices=[*DOMAINS,"all"],required=True)
    parser.add_argument("--candidate",default="baseline",help="baseline, invalid, or a JSON file")
    parser.add_argument("--scenario-set",choices=["development","selection","audit"],default="selection")
    parser.add_argument("--case-id",type=int,default=0)
    parser.add_argument("--all-cases",action="store_true",help="Run every registered complete native case in this role")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--output",type=Path,default=Path("results/science"))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    domains = DOMAINS if args.domain=="all" else [args.domain]
    if args.all_cases:
        from specrsi.science.registry import case_count
        if args.workers<1 or args.workers>16:parser.error("workers must be 1..16")
        jobs=[(domain,i) for domain in domains for i in range(case_count(domain))]
        def job(pair):
            domain,i=pair
            output=args.output/f"case_{i:02d}"/domain
            command=[sys.executable,__file__,"--domain",domain,"--candidate",args.candidate,"--scenario-set",args.scenario_set,
                     "--case-id",str(i),"--output",str(output)]
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMBA_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",PYBAMM_DISABLE_TELEMETRY="true")
            result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=240)
            return {"domain":domain,"case_id":i,"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr[-2000:]}
        rows=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(job,pair) for pair in jobs]):
                row=future.result();rows.append(row);print(json.dumps(row),flush=True)
        (args.output/"suite_execution.json").write_text(json.dumps(rows,indent=2)+"\n")
        return int(any(row["exit_code"] for row in rows))
    failed = False
    for domain in domains:
        candidate = load_config(domain,args.scenario_set,args.case_id)[args.candidate] if args.candidate in {"baseline","invalid"} else json.loads(Path(args.candidate).read_text())
        label = args.candidate if args.candidate in {"baseline","invalid"} else "candidate"
        try:
            result = evaluate(domain,candidate,scenario_set=args.scenario_set,case_id=args.case_id)
        except Exception as exc:
            failed = True
            result = {"domain":domain,"status":"infrastructure_error","complete":False,"error":f"{type(exc).__name__}: {exc}"}
            import traceback
            traceback.print_exc()
        (args.output/f"{domain}_{label}.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        (args.output/f"{domain}_registration.json").write_text(json.dumps(registration(domain,args.scenario_set,args.case_id),indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("domain","status","native_score","wall_seconds") if k in result}),flush=True)
    return int(failed)


if __name__=="__main__":
    raise SystemExit(main())
