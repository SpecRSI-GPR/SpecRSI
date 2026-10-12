#!/usr/bin/env python3
"""Start an isolated publisher-FP8 Qwen3.5 server without touching existing services."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

import requests


def save(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    if data.get("attempt_id"):
        archived = path.with_name(path.stem + "." + data["attempt_id"] + path.suffix)
        archived.write_text(json.dumps(data, indent=2) + "\n")


def existing_health(session):
    records = {}
    for port in (18080, 18081):
        try:
            response = session.get(f"http://127.0.0.1:{port}/health", timeout=10)
            records[str(port)] = {"http_status": response.status_code, "response": response.json()}
        except (requests.RequestException, ValueError) as error:
            records[str(port)] = {"error": str(error)}
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/qwen35_35b_a3b_fp8.lock.json")
    parser.add_argument("--wait-seconds", type=int, default=1200)
    parser.add_argument("--execution-mode", choices=("eager", "decode_graph"), default="eager")
    parser.add_argument("--run-name", default="qwen35_fp8",
                        help="Separate immutable attempt archives, logs, and PID files")
    parser.add_argument("--synthetic-output-tokens", type=int, default=0,
                        help="Optional one non-task decoder throughput request after the two smokes")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Verify checkpoint and save launch commands without starting any GPU work")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.run_name):
        raise ValueError("run-name must contain only lowercase letters, digits and underscores")
    if not 0 <= args.synthetic_output_tokens <= 2048:
        raise ValueError("synthetic-output-tokens must be between 0 and 2048")
    workspace = Path(__file__).resolve().parents[1]
    os.chdir(workspace)
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text())
    runtime = lock.get("serving_runtime", {})
    if runtime and runtime.get("execution_mode") != args.execution_mode:
        raise ValueError("Explicit execution mode must match the frozen runtime lock")
    for filename, expected in runtime.get("source_sha256", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen serving source changed: {filename}")
    if lock["gpu_ids"] != [1] or lock["gpu_memory_budget_bytes"] > 58000000000:
        raise ValueError("This isolated deployment is limited to GPU 1 and a 58 GB vLLM memory target")
    manifest_path = Path(lock["download_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "publisher_verified_complete"
            or manifest.get("revision") != lock["revision"]
            or manifest.get("lock_sha256") != hashlib.sha256(lock_path.read_bytes()).hexdigest()):
        raise ValueError("The matching publisher-verified checkpoint manifest is required")
    # Every new server start checks every byte again; tokenizer and weights remain
    # unmodified copies of this exact publisher revision.
    from download_pinned_model import verify_file
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda entry: verify_file(Path(lock["local_path"]), entry), lock["files"]))
    for port in (lock["backend_port"], lock["gateway_port"]):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"Port {port} is occupied; existing service is left untouched") from error
    memory = subprocess.check_output(
        ["nvidia-smi", "--id=1", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True).strip().split(",")
    total_bytes, free_bytes = (int(value.strip()) * 1024**2 for value in memory)
    memory_target = lock["gpu_memory_budget_bytes"]
    if free_bytes < memory_target + 2 * 1024**3:
        raise RuntimeError("GPU 1 no longer has the isolated server target plus 2 GiB headroom")
    fraction = memory_target / total_bytes
    python = str(workspace / ".venv-kimi/bin/python")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                       VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_HOST_IP="127.0.0.1")
    compilation = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                   "cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": 1}
    if args.execution_mode == "decode_graph":
        environment["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
        if runtime and runtime.get("compilation_config") != compilation:
            raise ValueError("Frozen CUDA graph configuration differs from the launcher")
        execution_options = ["--compilation-config", json.dumps(compilation, separators=(",", ":"))]
    else:
        execution_options = ["--enforce-eager"]
    backend = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", lock["local_path"],
               "--served-model-name", lock["served_model_name"], "--host", "127.0.0.1",
               "--port", str(lock["backend_port"]), "--dtype", "bfloat16",
               "--gpu-memory-utilization", str(fraction), "--max-model-len", str(lock["context_length"]),
               "--max-num-seqs", "1", "--max-num-batched-tokens", "2048", *execution_options,
               "--safetensors-load-strategy", "eager", "--language-model-only",
               "--reasoning-parser", "qwen3", "--generation-config", "vllm"]
    gateway = [python, "scripts/serve_budget_gateway.py", "--lock", args.lock,
               "--backend", f"http://127.0.0.1:{lock['backend_port']}",
               "--port", str(lock["gateway_port"]), "--log", f"logs/{args.run_name}_requests.jsonl"]
    report = {"status": "ready_cpu_preparation_only", "repo_id": lock["repo_id"], "revision": lock["revision"],
              "precision_extension_not_paper_bf16": True, "format": lock["format"],
              "backend_command": backend, "gateway_command": gateway, "gpu_ids": lock["gpu_ids"],
              "gpu_memory_target_bytes": memory_target, "gpu_memory_hard_limit": False,
              "gpu_memory_utilization": fraction, "gpu_initial_free_bytes": free_bytes,
              "base_url": f"http://127.0.0.1:{lock['gateway_port']}", "prepared_at_unix": time.time(),
              "download_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    report.update(lock_path=str(lock_path), lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                  execution_mode=args.execution_mode,
                  compilation_config=compilation if args.execution_mode == "decode_graph" else None,
                  graph_memory_profiler_estimation=args.execution_mode == "decode_graph",
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ("scripts/deploy_qwen35_fp8.py", "scripts/serve_budget_gateway.py")},
                  workload_category="non_task_deployment_infrastructure", task_benchmark_calls=0,
                  previous_deployment_report=runtime.get("previous_deployment_report"))
    if args.prepare_only:
        save(Path(f"reports/{args.run_name}_preparation.json"), report)
        print(json.dumps({"status": report["status"], "base_url": report["base_url"],
                          "gpu_processes_started": False}), flush=True)
        return
    session = requests.Session()
    session.trust_env = False
    report.update(attempt_id=time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
                  old_services_health_before=existing_health(session))
    processes = {}
    for name, command in (("backend", backend), ("gateway", gateway)):
        with open(f"logs/{args.run_name}_{name}.log", "a") as output:
            output.write(f"\nDeployment attempt {report['attempt_id']}\n")
            output.flush()
            process = subprocess.Popen(command, cwd=workspace, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        Path(f"logs/{args.run_name}_{name}.pid").write_text(str(process.pid) + "\n")
    report_path = Path(f"reports/{args.run_name}_deployment.json")
    report.update(status="loading", processes={name: process.pid for name, process in processes.items()},
                  started_at_unix=time.time())
    save(report_path, report)
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        dead = [name for name, process in processes.items() if process.poll() is not None]
        if dead:
            report.update(status="failed", failed_processes=dead)
            save(report_path, report)
            raise RuntimeError(f"New Qwen3.5 processes failed: {dead}; existing services were not changed")
        try:
            response = session.get(report["base_url"] + "/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if health.get("revision") != lock["revision"] or health.get("protocol_version") != 2:
                raise ValueError("Gateway health identity does not match the pinned model")
            report["health"] = health
            break
        except requests.RequestException:
            time.sleep(5)
    else:
        report["status"] = "still_loading_after_wait"
        save(report_path, report)
        raise TimeoutError("New processes remain available to inspect; existing services were not changed")
    for thinking, max_tokens in ((False, 32), (True, 256)):
        payload = {"model": lock["served_model_name"], "messages": [{"role": "user", "content":
                   "What is 2 + 2? Return only the final answer."}], "max_tokens": max_tokens,
                   "temperature": 0., "top_p": .95, "seed": 20260928, "enable_thinking": thinking,
                   "max_seconds": 300, "total_token_cap": 512}
        key = f"smoke_thinking_{str(thinking).lower()}"
        report[key] = {"request": payload, "status": "pending", "maximum_request_tokens": 512,
                       "workload_category": "non_task_arithmetic_smoke"}
        save(report_path, report)
        try:
            response = session.post(report["base_url"] + "/v1/chat/completions", json=payload, timeout=360)
            report[key].update(http_status=response.status_code, response_text=response.text)
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            report[key].update(status="error_actual_usage_requires_receipt_reconciliation", error=str(error))
            report.update(status="smoke_request_failed", old_services_health_after=existing_health(session))
            save(report_path, report)
            raise
        report[key].update(status="metered", response=result,
            completion_tokens_per_elapsed_second=result["usage"]["completion_tokens"] /
            max(result["specrsi_meter"]["elapsed_seconds"], 1e-12))
        answer = result["choices"][0]["message"]["content"].strip()
        save(report_path, report)
        if answer != "4" or result["choices"][0]["finish_reason"] != "stop":
            report["status"] = "smoke_incomplete_or_incorrect"
            save(report_path, report)
            raise RuntimeError("Actual model smoke did not return a complete correct final answer")
    if args.synthetic_output_tokens:
        payload = {"model": lock["served_model_name"], "messages": [{"role": "user", "content":
                   "Write a numbered list from 1 to 1000. Every line must have exactly this form: "
                   "n: This is a synthetic decoder throughput check. Replace n with the line number. "
                   "Begin immediately with line 1 and continue without commentary, abbreviations, "
                   "or an early conclusion."}], "max_tokens": args.synthetic_output_tokens,
                   "temperature": 0., "top_p": .95, "seed": 20260928, "enable_thinking": False,
                   "max_seconds": 300, "total_token_cap": args.synthetic_output_tokens + 512}
        key = "synthetic_decoder_microbenchmark"
        report[key] = {"request": payload, "status": "pending",
                       "workload_category": "non_task_synthetic_decoder_microbenchmark",
                       "maximum_request_tokens": payload["total_token_cap"]}
        save(report_path, report)
        try:
            response = session.post(report["base_url"] + "/v1/chat/completions", json=payload, timeout=360)
            report[key].update(http_status=response.status_code, response_text=response.text)
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            report[key].update(status="error_actual_usage_requires_receipt_reconciliation", error=str(error))
            report.update(status="microbenchmark_request_failed", old_services_health_after=existing_health(session))
            save(report_path, report)
            raise
        report[key].update(status="metered", response=result,
            completion_tokens_per_elapsed_second=result["usage"]["completion_tokens"] /
            max(result["specrsi_meter"]["elapsed_seconds"], 1e-12))
        save(report_path, report)
        if result["choices"][0]["finish_reason"] not in ("length", "stop"):
            report["status"] = "microbenchmark_incomplete"
            save(report_path, report)
            raise RuntimeError("Synthetic throughput request violated its serving contract")
    metered = [record["response"] for record in report.values()
               if isinstance(record, dict) and record.get("status") == "metered"]
    report["infrastructure_inference_cost"] = {
        "requests": len(metered),
        "prompt_tokens": sum(item["usage"]["prompt_tokens"] for item in metered),
        "completion_tokens": sum(item["usage"]["completion_tokens"] for item in metered),
        "total_tokens": sum(item["usage"]["total_tokens"] for item in metered),
        "sum_backend_elapsed_seconds": sum(item["specrsi_meter"]["elapsed_seconds"] for item in metered),
        "task_benchmark_calls": 0,
        "gpu_seconds_note": "Per-request elapsed time; exclusive physical GPU attribution unavailable on shared GPU"}
    report.update(status="ready_inference_verified", completed_at_unix=time.time(),
                  old_services_health_after=existing_health(session),
                  gpu_memory_after=subprocess.check_output(["nvidia-smi", "--id=1",
                    "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
                    text=True).strip(),
                  throughput_scope="completion_tokens_over_full_elapsed_including_prefill; arithmetic_and_optional_synthetic_only")
    save(report_path, report)
    print(json.dumps({"status": report["status"], "base_url": report["base_url"], "revision": lock["revision"]}), flush=True)


if __name__ == "__main__":
    main()
