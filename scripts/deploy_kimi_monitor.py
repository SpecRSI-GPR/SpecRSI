#!/usr/bin/env python3
"""Record bounded GPU resource measurements during the real Kimi deployment."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    until = time.monotonic() + args.seconds
    output = Path("reports/kimi_gpu_loading_measurements.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    while time.monotonic() < until:
        response = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        rows = []
        if response.returncode == 0:
            for line in response.stdout.splitlines():
                index, used, total, utilization = map(int, line.split(","))
                rows.append({"gpu": index, "used_mib": used, "total_mib": total,
                             "utilization_percent": utilization, "allocated_to_kimi": index != 1})
        status = "waiting_for_model_manifest"
        report_path = Path("reports/kimi_deployment.json")
        if report_path.exists():
            status = json.loads(report_path.read_text()).get("status", "unknown")
        with output.open("a") as handle:
            handle.write(json.dumps({"unix_time": time.time(), "status": status, "gpus": rows}) + "\n")
        if status == "ready_inference_verified":
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
