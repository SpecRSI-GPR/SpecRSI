#!/usr/bin/env python3
"""Idempotent local deployment of the exact paper reference text backbone.

The lock binds publisher files, revision, precision, GPU, context, and ports.
This is infrastructure, not a benchmark: startup/smoke costs are recorded
separately. A cron watchdog may call this entry repeatedly; flock and Linux
process birth ticks prevent duplicate servers and accidental PID-reuse kills.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/qwen35_paper33_20261010"
STATE = REPORT / "serving.json"


def save(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(STATE)


def birth(pid):
    try:
        # comm may contain spaces/parentheses; parse fields after the final ')'.
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return record and birth(record["pid"]) == record["birth_ticks"]


def group_alive(record):
    """Retain ownership of surviving workers after their API parent dies."""
    if record.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return False
    current = birth(record["pid"])
    if current is not None and current != record["birth_ticks"]:
        return False
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != "Z" and int(fields[2]) == record["pid"] and int(fields[3]) == record["pid"]:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default="configs/models/qwen35_35b_a3b_bf16.lock.json")
    parser.add_argument("--wait-seconds", type=int, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable in the vLLM serving environment")
    args = parser.parse_args()
    os.chdir(ROOT)
    REPORT.mkdir(parents=True, exist_ok=True)
    lease = (REPORT / "serving.flock").open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('{"status":"deployment_already_running"}')
        return
    lockpath = Path(args.lock)
    lock = json.loads(lockpath.read_text())
    locksha = hashlib.sha256(lockpath.read_bytes()).hexdigest()
    if (lock["repo_id"] != "Qwen/Qwen3.5-35B-A3B"
            or lock["revision"] != "59d61f3ce65a6d9863b86d2e96597125219dc754"
            or lock["format"] != "publisher_bfloat16_unquantized"
            or lock["gpu_ids"] != [0]):
        raise ValueError("This deployment requires the registered unquantized reference model on GPU 0")
    manifest = json.loads(Path(lock["download_manifest"]).read_text())
    if (manifest.get("status") != "publisher_verified_complete"
            or manifest.get("lock_sha256") != locksha):
        raise ValueError("Matching publisher verification must finish before GPU allocation")
    session = requests.Session()
    session.trust_env = False
    base_url = f"http://127.0.0.1:{lock['gateway_port']}"

    def health():
        try:
            response = session.get(base_url + "/health", timeout=5)
            response.raise_for_status()
            value = response.json()
            return value if (value.get("revision") == lock["revision"]
                and value.get("model") == lock["served_model_name"]
                and value.get("dtype") == lock["format"]) else None
        except (requests.RequestException, ValueError):
            return None

    if value := health():
        print(json.dumps({"status": "healthy", "health": value}))
        return
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    # Finish a previous startup after a supervisor restart, or stop only our
    # own stale pair before relaunch. Never infer ownership from a port/GPU.
    records = previous.get("processes", {})
    running = [record for record in records.values() if alive(record)]
    if len(running) == 2 and time.time() - previous.get("started_at_unix", 0) < args.wait_seconds:
        print(json.dumps({"status": "loading", "processes": running}))
        return
    owned_groups = [record for record in records.values() if group_alive(record)]
    for record in owned_groups:
        os.killpg(record["pid"], signal.SIGTERM)
    if owned_groups:
        time.sleep(5)
    for record in owned_groups:
        if group_alive(record):
            os.killpg(record["pid"], signal.SIGKILL)
    for port in (lock["backend_port"], lock["gateway_port"]):
        with socket.socket() as probe:
            # A terminated HTTP server can leave TIME_WAIT connections.
            # Match the listener's reuse behavior while still rejecting a
            # live foreign listener; do not mistake TIME_WAIT for ownership.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    # Cheap size checks are repeated at every start; SHA checks are the
    # immutable download receipt. The model directory is never edited here.
    for entry in lock["files"]:
        path = Path(lock["local_path"]) / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Publisher file missing/truncated: {path}")
    python = str(Path(args.python).resolve())
    total, free = map(int, subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(","))
    target = lock["gpu_memory_budget_bytes"]
    if free * 1024**2 < target + 2 * 1024**3:
        raise RuntimeError("Registered GPU 0 target plus 2 GiB headroom is unavailable")
    backend = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", lock["local_path"],
        "--served-model-name", lock["served_model_name"], "--host", "127.0.0.1", "--port", str(lock["backend_port"]),
        "--dtype", "bfloat16", "--gpu-memory-utilization", str(target / (total * 1024**2)),
        "--max-model-len", str(lock["context_length"]), "--max-num-seqs", "1",
        "--max-num-batched-tokens", "2048", "--enforce-eager", "--safetensors-load-strategy", "eager",
        "--language-model-only", "--reasoning-parser", "qwen3", "--generation-config", "vllm",
        # vLLM 0.19.1 supports this GDN backend explicitly. The installed
        # FlashInfer wheel omits its SM90 prefill header; use vLLM's native
        # Triton/FLA implementation, retaining exact BF16 publisher weights.
        "--additional-config", json.dumps({"gdn_prefill_backend": "triton"})]
    gateway = [python, "scripts/serve_budget_gateway.py", "--lock", args.lock,
        "--backend", f"http://127.0.0.1:{lock['backend_port']}", "--port", str(lock["gateway_port"]),
        "--log", "logs/qwen35_bf16_requests.jsonl"]
    state = {"status": "prepared", "lock_sha256": locksha, "base_url": base_url,
        "commands": {"backend": backend, "gateway": gateway}, "processes": {},
        "started_at_unix": time.time(), "infrastructure_only": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.prepare_only:
        print(json.dumps(state, indent=2))
        return
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_HOST_IP="127.0.0.1")
    Path("logs").mkdir(exist_ok=True)
    for name, command in state["commands"].items():
        with Path(f"logs/qwen35_bf16_{name}.log").open("a") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)
        state["processes"][name] = {"pid": process.pid, "birth_ticks": birth(process.pid),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        state["status"] = "loading"
        save(state)
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        if not all(alive(p) for p in state["processes"].values()):
            state.update(status="failed", failed_at_unix=time.time())
            save(state)
            raise RuntimeError("Registered server process exited; inspect its deployment log")
        if value := health():
            state.update(status="healthy", health=value, ready_at_unix=time.time())
            save(state)
            print(json.dumps({"status": "healthy", "base_url": base_url}))
            return
        time.sleep(5)
    state["status"] = "loading_timeout"
    save(state)
    raise TimeoutError("Model server did not become healthy within its registered startup window")


if __name__ == "__main__":
    main()
