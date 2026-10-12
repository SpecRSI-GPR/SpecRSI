from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def doctor(*, base_url=None, model="Qwen3.5-35B-A3B", kimi_base_url=None):
    """Inspect installed runtimes and available inputs without starting services.

    Model health is queried only for explicitly supplied endpoints. Dataset
    contents and trained weights are downloaded/generated separately; their
    current presence is reported independently from the shipped source,
    split-membership, architecture, and deployment specifications.
    """
    from .models import LocalModelClient
    report = {"python": sys.version, "packages": {}, "model": {"status": "not_requested"},
              "models": {}, "datasets": {}, "reproduction": {}}
    for name in ["torch", "transformers", "peft", "huggingface-hub", "datasets", "pyomo", "highspy", "ifbench"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    services = {}
    if base_url is not None:
        services[model] = base_url
    if kimi_base_url is not None:
        services["Kimi-K2.6"] = kimi_base_url
    for name, endpoint in services.items():
        try:
            report["models"][name] = LocalModelClient(endpoint, name).health()
        except Exception as error:
            report["models"][name] = {"status": "unavailable", "reason": str(error)}
    if base_url is not None:
        report["model"] = report["models"][model]
    for manifest in sorted(Path("data/manifests").glob("*.json")):
        data = json.loads(manifest.read_text())
        if "dataset" not in data:
            continue
        report["datasets"][manifest.stem] = data.get("status", data.get("preparation_status", "manifest_present"))
    for name, filename in {
        "datasets": "data/reproduction/catalog.json",
        "science_scenarios": "data/reproduction/science_scenarios.json",
        "training_profiles": "configs/reproduction/training_profiles.json",
        "deployment": "configs/reproduction/deployment.json",
        "role_architecture": "configs/reproduction/role_architecture.json",
        "model_snapshot": "configs/reproduction/model_snapshot.json",
    }.items():
        path = Path(filename)
        report["reproduction"][name] = {"path": filename, "present": path.is_file()}
    report["ready_for"] = {
        "local_qwen_inference": report["model"].get("status") == "ok",
        "local_kimi_inference": report["models"].get("Kimi-K2.6", {}).get("status") == "ok",
    }
    report['native_validation'] = {}
    for name, filename in {'polyglot':'reports/polyglot_native.json','hover':'reports/hover_retrieval.json',
            'science':'reports/science/validation_summary.json','swe_images':'reports/swe_images.json',
            'swe_development_images':'reports/swe_development_images.json',
            'kimi':'reports/kimi_deployment.json'}.items():
        path=Path(filename)
        if path.is_file():
            value=json.loads(path.read_text())
            report['native_validation'][name]={'report':str(path), **{k:value[k] for k in
                ['status','total','passed','complete','complete_valid_cases','failed_cases'] if k in value}}
            if name in {'swe_images','swe_development_images'}:
                report['native_validation'][name].update(ready_images=len(value.get('images',{})),
                    requested_images=len(value.get('requested_ids',[])),failures=len(value.get('failures',[])))
    report['ready_for']['prospective_training_data']=Path('data/training/prospective_v2/warmup.jsonl').is_file()
    report['ready_for']['polyglot_native']=report['native_validation'].get('polyglot',{}).get('passed')==225
    report['ready_for']['science_native']=report['native_validation'].get('science',{}).get('complete_valid_cases')==168
    report['ready_for']['all_swe_images']=all(report['native_validation'].get(name,{}).get('complete',False)
        for name in ('swe_images','swe_development_images'))
    report['native_validation']['total_swe_images']=sum(report['native_validation'].get(name,{}).get('ready_images',0)
        for name in ('swe_images','swe_development_images'))
    report["verified_gpu_training_report"] = str(Path("artifacts/training/qwen_gpu_smoke/report.json")) if Path("artifacts/training/qwen_gpu_smoke/report.json").exists() else None
    return report


def main():
    parser = argparse.ArgumentParser(prog="specrsi")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="Inspect installed runtimes, input availability, and reproduction specifications")
    d.add_argument("--base-url", help="Explicit reference-model endpoint to check")
    d.add_argument("--model", default="Qwen3.5-35B-A3B")
    d.add_argument("--kimi-base-url", help="Optional Kimi endpoint to check")
    d.add_argument("--output", default="reports/readiness.json")
    demo = sub.add_parser("demo", help="Run pure-integer reference search (not a benchmark result)")
    demo.add_argument("--output", default="results/core_demo.json")
    run = sub.add_parser("run", help="Run an explicit prospective local-model configuration")
    run.add_argument("--config", required=True)
    arguments = parser.parse_args()
    if arguments.command == "doctor":
        result = doctor(base_url=arguments.base_url, model=arguments.model,
                        kimi_base_url=arguments.kimi_base_url)
    elif arguments.command == "demo":
        from .core.demo import run_demo
        result = run_demo()
        if hasattr(result, "to_dict"):
            result = result.to_dict()
    else:
        import yaml
        from .experiments import run_local_experiment
        result = run_local_experiment(yaml.safe_load(Path(arguments.config).read_text()))
    if hasattr(arguments, "output"):
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
