#!/usr/bin/env python3
"""Launch the pinned full native Kimi checkpoint and validate real local inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (OSError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/kimi_k26.lock.json")
    parser.add_argument("--wait-download", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--verification-workers", type=int, default=16)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    os.chdir(workspace)
    lock = json.loads(Path(args.lock).read_text())
    manifest = Path("reports/kimi_download.json")
    begun = time.time()
    while not manifest.exists():
        if not args.wait_download or time.time() - begun > args.wait_seconds:
            raise RuntimeError("Full publisher-verified Kimi checkpoint download is not complete")
        time.sleep(10)
    verification = json.loads(manifest.read_text())
    if verification.get("status") != "verified_complete" or verification.get("revision") != lock["revision"]:
        raise ValueError("Download verification manifest does not match pinned Kimi model")
    records = {e["path"]: e for e in verification["files"]}
    for entry in lock["files"]:
        p = Path(lock["local_path"]) / entry["path"]
        if entry["path"] not in records or not p.is_file() or p.stat().st_size != entry["bytes"]:
            raise ValueError(f"Incomplete verified checkpoint file: {entry['path']}")
        if entry.get("sha256") and records[entry["path"]]["sha256"] != entry["sha256"]:
            raise ValueError(f"Publisher hash mismatch in verification manifest: {entry['path']}")
        # Recheck publisher Git identities for code and small metadata; all file
        # bytes, including the 595 GB weights, are streamed and hashed below.
        if entry.get("git_blob_sha1"):
            data = p.read_bytes()
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            if hashlib.sha1(blob).hexdigest() != entry["git_blob_sha1"]:
                raise ValueError(f"Publisher Git blob mismatch: {entry['path']}")
    # Every future invocation re-reads and hashes every weight byte. This pass
    # also fixes the slow random mmap fault path on the shared FUSE filesystem.
    from deploy_kimi_verify import verify_checkpoint
    startup_verification = verify_checkpoint(args.lock, workers=args.verification_workers)
    if Path("reports/kimi_mapped_libraries.json").exists():
        subprocess.run([sys.executable, "scripts/deploy_kimi_prefetch.py", "--mapped-libraries", "--workers", "4"],
                       check=True, stdout=open("logs/kimi_library_prefetch.log", "a"), stderr=subprocess.STDOUT)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(map(str, lock["gpu_ids"])), "OMP_NUM_THREADS": "8",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
                "VLLM_PP_LAYER_PARTITION": ",".join(map(str, lock["pipeline_layer_partition"])), "NCCL_IB_DISABLE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_HOST_IP": "127.0.0.1"})
    python = str(workspace / ".venv-kimi/bin/python")
    backend = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", lock["local_path"],
          "--served-model-name", lock.get("served_model_name", "Kimi-K2.6"), "--host", "127.0.0.1", "--port", "18082",
          "--trust-remote-code", "--tensor-parallel-size", str(lock["tensor_parallel_size"]),
          "--pipeline-parallel-size", str(lock["pipeline_parallel_size"]), "--distributed-executor-backend", "mp",
          "--gpu-memory-utilization", "0.92", "--max-model-len", str(lock["context_length"]),
          "--max-num-seqs", "2", "--max-num-batched-tokens", "512", "--enforce-eager",
          "--mm-encoder-tp-mode", "data", "--limit-mm-per-prompt", '{"image":0,"video":0}',
          "--reasoning-parser", "kimi_k2", "--tool-call-parser", "kimi_k2", "--enable-auto-tool-choice"]
    gateway = [python, "scripts/deploy_kimi_gateway.py", "--lock", args.lock]
    processes = {}
    for name, command in (("backend", backend), ("gateway", gateway)):
        pidfile = Path(f"logs/kimi_{name}.pid")
        if pidfile.exists() and alive(int(pidfile.read_text())):
            processes[name] = int(pidfile.read_text())
            continue
        process = subprocess.Popen(command, cwd=workspace, env=env,
            stdout=open(f"logs/kimi_{name}.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
        pidfile.write_text(str(process.pid) + "\n")
        processes[name] = process.pid
    session = requests.Session()
    session.trust_env = False
    report = {"status": "loading", "repo_id": lock["repo_id"], "revision": lock["revision"],
              "processes": processes, "backend_command": backend, "gateway_command": gateway,
              "gpu_ids": lock["gpu_ids"], "context_length": lock["context_length"], "base_url": "http://127.0.0.1:18081",
              "started_at_unix": time.time(), "checkpoint_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    report["startup_verification"] = {k: v for k, v in startup_verification.items() if k != "files"}
    path = Path("reports/kimi_deployment.json")
    save(path, report)
    deadline = time.time() + args.wait_seconds
    while time.time() < deadline:
        dead = [name for name, pid in processes.items() if not alive(pid)]
        if dead:
            report.update(status="failed", failed_processes=dead)
            save(path, report)
            raise RuntimeError(f"Kimi deployment process failed: {dead}; inspect logs/kimi_backend.log and logs/kimi_gateway.log")
        try:
            response = session.get("http://127.0.0.1:18081/health", timeout=5)
            response.raise_for_status()
            report["health"] = response.json()
            health = report["health"]
            if (health.get("revision") != lock["revision"] or health.get("protocol_version") != 2
                    or not health.get("server_enforces_total_token_cap")
                    or health.get("gpu_ids") != lock["gpu_ids"]
                    or health.get("context_length") != lock["context_length"]):
                report["status"] = "health_contract_mismatch"
                save(path, report)
                raise ValueError("Tracked Kimi gateway does not match pinned revision, devices, context or SpecRSI protocol")
            break
        except requests.RequestException:
            time.sleep(10)
    else:
        report.update(status="still_loading_after_wait")
        save(path, report)
        raise TimeoutError("Kimi is still loading; processes remain available to inspect")
    if not args.skip_smoke:
        request = {"model": lock.get("served_model_name", "Kimi-K2.6"),
                   "messages": [{"role": "user", "content": "What is 2 + 2? Reply with only the answer."}],
                   "max_tokens": 32, "temperature": 0., "top_p": .95, "seed": 20260916,
                   "enable_thinking": False, "max_seconds": 600, "total_token_cap": 256}
        report["smoke_prompt"] = request
        try:
            response = session.post("http://127.0.0.1:18081/v1/chat/completions", json=request, timeout=900)
            response.raise_for_status()
        except requests.RequestException as error:
            report.update(status="inference_request_failed", error=str(error))
            if error.response is not None:
                report["response_status"] = error.response.status_code
                report["response_body"] = error.response.text
            save(path, report)
            raise
        report["inference"] = response.json()
        answer = report["inference"]["choices"][0]["message"]["content"].strip()
        if not answer or report["inference"]["choices"][0]["finish_reason"] == "timeout":
            report["status"] = "inference_incomplete"
            save(path, report)
            raise RuntimeError("Kimi health succeeded but real inference did not complete")
        report["semantic_smoke_check"] = {"expected": "4", "actual": answer, "passed": answer == "4"}
        if answer != "4":
            report["status"] = "inference_unexpected_answer"
            save(path, report)
            raise RuntimeError("Kimi returned an unexpected answer to the arithmetic sanity check")
    report.update(status="ready_inference_verified" if not args.skip_smoke else "ready_health_only", completed_at_unix=time.time())
    save(path, report)
    print(json.dumps({"status": report["status"], "base_url": report["base_url"], "model": lock["repo_id"],
                      "gpu_ids": lock["gpu_ids"]}), flush=True)


if __name__ == "__main__":
    main()
