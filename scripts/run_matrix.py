#!/usr/bin/env python3
"""Run registered prospective configs by seed, with restart-safe completion markers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
import os
import uuid


def runtime_fingerprint(config, root):
    """A completion marker applies only to the same source and pinned artifacts."""
    paths = set((root / "src/specrsi").rglob("*.py"))
    paths.update((root / "scripts").glob("*.py"))
    paths.add(root / config["model"]["lock"])
    if config.get('drafter_model'): paths.add(root / config['drafter_model']['lock'])
    donors = config.get('search', {}).get('observation_donors')
    if isinstance(donors, str): paths.add(root / donors)
    if config.get('search',{}).get('controller') == 'learned' or config.get('search',{}).get('role_checkpoint'):
        checkpoint = root / config['search']['role_checkpoint']
        if not checkpoint.is_dir():
            raise ValueError(f'Missing learned role checkpoint: {checkpoint}')
        paths.update(p for p in checkpoint.rglob('*') if p.is_file())
    paths.update((root/'configs/runtimes').rglob('*lock*'))
    if config.get('task_pipeline')=='hover_three_hop': paths.add(root/'reports/hover_retrieval.json')
    if config['dataset']=='polyglot': paths.add(root/'configs/datasets/polyglot_java.lock.json')
    paths.add(root / config.get("data_root", "data") / "manifests" / (config["dataset"] + ".json"))
    for optional in ("pyproject.toml", "requirements.lock", "reports/model_download.json"):
        if (root / optional).is_file():
            paths.add(root / optional)
    hashes = {}
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(f"Missing declared experiment input: {path}")
        hashes[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def valid_completion(path, identity):
    try:
        marker = json.loads(path.read_text())
        if marker.get("config_sha256") != identity:
            return False
        hashes = marker.get("output_hashes", {})
        expected = {"search.json", "selected.json", "audit.json", "physical_calls.jsonl"}
        if set(hashes) != expected:
            return False
        return all(hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == checksum for name, checksum in hashes.items())
    except (OSError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default="results/matrix")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = []
    output_root = Path(args.output).resolve()
    for config_path in args.configs:
        config = yaml.safe_load(Path(config_path).read_text())
        if "studies" in config:
            raise ValueError("paper_matrix.yaml records study requirements; supply runnable per-study configs with frozen artifacts")
        for seed in args.seeds:
            effective = {**config, "seed": seed}
            fingerprint = runtime_fingerprint(effective, root)
            identity = hashlib.sha256(json.dumps({"config": effective, "inputs": fingerprint}, sort_keys=True).encode()).hexdigest()
            target = output_root / Path(config_path).stem / f"seed-{seed}-{identity[:10]}"
            done = target / "completed.json"
            if valid_completion(done, identity):
                print(f"already complete: {target}", flush=True)
                continue
            target.mkdir(parents=True, exist_ok=True)
            effective["output"] = str(target)
            current = target / "run.yaml"
            current.write_text(yaml.safe_dump(effective, sort_keys=False))
            with (target / "run.log").open("a") as log:
                job = subprocess.run([sys.executable, str(root / "scripts/run_experiment.py"), "--config", str(current)],
                                     cwd=root, stdout=log, stderr=subprocess.STDOUT)
            if job.returncode:
                failures.append(str(target))
            else:
                expected = ("search.json", "selected.json", "audit.json", "physical_calls.jsonl")
                if not all((target / name).is_file() for name in expected):
                    failures.append(str(target))
                    continue
                summary = json.loads((target / "search.json").read_text())
                if summary.get("source_changed_during_run") or runtime_fingerprint(effective, root) != fingerprint:
                    failures.append(str(target))
                    continue
                marker = {"config_sha256": identity, "seed": seed, "input_hashes": fingerprint,
                          "output_hashes": {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in expected}}
                temporary = target / (".completed-" + uuid.uuid4().hex)
                temporary.write_text(json.dumps(marker, sort_keys=True) + "\n")
                os.replace(temporary, done)
    if failures:
        print(json.dumps({"failed": failures}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
