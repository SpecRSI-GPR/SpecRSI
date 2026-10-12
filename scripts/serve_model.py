#!/usr/bin/env python3
"""Start/status/stop only this workspace's local Qwen service."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PID = ROOT / "logs/qwen_server.pid"


def own_process():
    if not PID.exists():
        return None
    pid = int(PID.read_text())
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        cwd = Path(f"/proc/{pid}/cwd").resolve()
        return pid if b"specrsi.models.server" in cmdline and cwd == ROOT else None
    except OSError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "status", "stop"])
    p.add_argument("--gpu", default="0")
    p.add_argument("--port", type=int, default=18085)
    args = p.parse_args()
    pid = own_process()
    if args.action == "stop":
        if pid:
            os.kill(pid, signal.SIGTERM)
            for _ in range(100):
                if own_process() is None:
                    PID.unlink(missing_ok=True)
                    break
                time.sleep(.1)
            else:
                print(json.dumps({"stopping_pid": pid, "status": "SIGTERM sent; still shutting down"}))
                return
        print(json.dumps({"stopped_pid": pid}))
        return
    if args.action == "start" and not pid:
        (ROOT / "logs").mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT / "src"),
                   HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        with (ROOT / "logs/qwen_server.log").open("a") as logfile:
            process = subprocess.Popen([sys.executable, "-m", "specrsi.models.server", "--port", str(args.port)],
                                       cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        pid = process.pid
        PID.write_text(str(pid) + "\n")
    state = {"pid": pid, "url": f"http://127.0.0.1:{args.port}", "healthy": False}
    try:
        import httpx
        r = httpx.get(state["url"] + "/health", timeout=3, trust_env=False)
        state.update(healthy=r.status_code == 200, response=r.json())
    except Exception as e:
        state["status"] = "starting or unavailable: " + type(e).__name__
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
