#!/usr/bin/env python3
"""Run pinned original GEPA using local Qwen and an independent frozen audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Import directly from the downloaded, revision-locked baseline checkout.
sys.path.insert(0, str(ROOT / "third_party/gepa/src"))

from specrsi.baselines.gepa import MeteredLocalModel, run_gepa
from specrsi.benchmarks import load_examples
from specrsi.models import LocalModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["gepa", "gepa-merge", "mipro-no-demos"], default="gepa")
    parser.add_argument("--dataset", choices=["game24", "aime25", "hotpotqa", "ifbench", "mbpp", "hover", "pupa", "math", "livebench_math", "planbench", "polyglot", "swe_verified", "swe_lite"], default="game24")
    parser.add_argument("--development-size", type=int, default=2)
    parser.add_argument("--selection-size", type=int, default=2)
    parser.add_argument("--audit-size", type=int, default=2)
    parser.add_argument("--max-metric-calls", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--no-data-aware-proposer", action="store_true", help="MIPRO only: skip development dataset summarization")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:18085")
    parser.add_argument("--model-lock",default="configs/models/qwen35_35b_a3b_bf16.lock.json")
    parser.add_argument("--selection-track",choices=['selection60','verified500'],default='selection60')
    parser.add_argument("--audit-track",choices=['lite_audit_filtered','lite_audit_disjoint_verified500'],default='lite_audit_filtered')
    parser.add_argument("--native-execution-cap",type=int,default=131072)
    parser.add_argument("--actor-max-steps",type=int,default=16)
    parser.add_argument("--test-timeout",type=int,default=1800)
    parser.add_argument("--harness-python",default=str(ROOT/'.envs/swe/bin/python'))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-prompt", default="Solve the task carefully. Follow its requested answer format exactly.")
    args = parser.parse_args()
    if min(args.development_size, args.selection_size, args.audit_size, args.max_metric_calls, args.max_tokens) < 1:
        parser.error("Sizes, token limits and metric-call cap must be positive")
    lock=json.loads(Path(args.model_lock).read_text())
    client = LocalModelClient(args.base_url,lock.get('served_model_name','Qwen3.5-35B-A3B'))
    health = client.health()
    model = MeteredLocalModel(client, max_tokens=args.max_tokens, seed=args.seed)
    if health['revision']!=lock['revision']: raise ValueError('Model revision differs from selected lock')
    model.model_revision=lock['revision']
    if args.dataset in {'hover','pupa','polyglot','swe_verified','swe_lite','livebench_math'}:
        from specrsi.baselines.tasks import attach_pipeline
        attach_pipeline(model,lock,dataset=args.dataset,native_config={'base_url':args.base_url,'lock':args.model_lock,
            'served_model_name':lock.get('served_model_name','Qwen3.5-35B-A3B'),'revision':lock['revision'],
            'execution_cap':args.native_execution_cap,'actor_max_steps':args.actor_max_steps,
            'test_timeout_seconds':args.test_timeout,'harness_python':args.harness_python})
    if args.dataset in {'swe_verified','swe_lite'}:
        from specrsi.baselines.native_backend import native_splits
        splits=native_splits({'dataset':args.dataset,'selection_track':args.selection_track,'audit_track':args.audit_track,
            'data':{'root':str(ROOT/'data'),'development':args.development_size,'selection':args.selection_size,'audit':args.audit_size}})
    else:
        splits = {split: load_examples(args.dataset, split, data_root=ROOT / "data",verify=split=='development')[:size]
                  for split, size in [("development", args.development_size), ("selection", args.selection_size), ("audit", args.audit_size)]}
    if args.method == "mipro-no-demos":
        from specrsi.baselines.mipro import run_mipro
        result = run_mipro(**splits, model=model, output_dir=args.output, seed_prompt=args.seed_prompt,
                           num_candidates=args.num_candidates, num_trials=args.num_trials, seed=args.seed,
                           data_aware=not args.no_data_aware_proposer)
    else:
        result = run_gepa(**splits, model=model, output_dir=args.output, seed_prompt=args.seed_prompt,
                          max_metric_calls=args.max_metric_calls, seed=args.seed, use_merge=args.method == "gepa-merge")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
