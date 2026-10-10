"""Actual ns-3.44 TCP/RED/ECN simulations, six full eight-second runs."""
from __future__ import annotations
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

from specrsi.benchmarks.science import aqm_score
from .common import InvalidCandidate,exact_keys,scalar

def native_executable() -> Path:
    source=Path(__file__).with_name("aqm_native.cc")
    identity=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    target=Path("/opt/specrsi-science/bin")/f"aqm-{identity}"
    if target.is_file():
        return target
    spec=importlib.util.find_spec("ns3")
    roots=list(spec.submodule_search_locations or []) if spec else []
    root=next((Path(p) for p in roots if (Path(p)/"lib64/libns3.44-core.so").exists()),None)
    if root is None:
        raise RuntimeError("ns3==3.44.post0 native libraries absent; run scripts/prepare_science.py --install")
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(f".{os.getpid()}.tmp")
    command=["g++","-std=c++20","-O2","-DSPECRSI_AQM_STANDALONE",f"-I{root/'include'}",str(source),
             f"-L{root/'lib64'}",f"-Wl,-rpath,{root/'lib64'}","-Wl,--no-as-needed",
             *[f"-lns3.44-{name}" for name in ["core","network","internet","applications","flow-monitor","point-to-point","traffic-control","stats"]],
             "-o",str(temporary)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError(f"Native ns-3 compilation failed: {result.stderr[-6000:]}")
    temporary.replace(target)
    return target


def simulate(candidate: dict,c: dict) -> dict:
    exact_keys(candidate,("min_threshold_packets","max_threshold_packets","max_probability","queue_limit_packets","ecn","delay_gain","target_delay_seconds"))
    low=scalar(candidate["min_threshold_packets"],1,500,"minimum threshold")
    high=scalar(candidate["max_threshold_packets"],2,1000,"maximum threshold")
    limit=scalar(candidate["queue_limit_packets"],4,2000,"queue limit")
    if low>=high or high>=limit or not float(limit).is_integer() or type(candidate["ecn"]) is not bool:
        raise InvalidCandidate("RED requires minimum < maximum < integer queue limit and boolean ECN")
    scalar(candidate["max_probability"],0.001,1,"maximum probability")
    scalar(candidate["delay_gain"],0,2,"delay gain")
    scalar(candidate["target_delay_seconds"],0.001,0.2,"delay target")
    executable=native_executable()
    rows=[]
    for mode in c["traffic_modes"]:
        for seed in c["seeds"]:
            args=(["long","burst","mixed"].index(mode),seed,low,high,
                  candidate["max_probability"],int(limit),int(candidate["ecn"]),candidate["delay_gain"],candidate["target_delay_seconds"],
                  c["duration_seconds"],c["control_interval_seconds"],c["bottleneck_mbps"],c["bottleneck_delay_ms"],
                  c["access_mbps"],c["access_delay_ms"],c["flows"],c["packet_bytes"],c["application_start_seconds"],
                  c["application_stop_seconds"],c["delay_histogram_bin_seconds"])
            completed=subprocess.run([str(executable),*map(str,args)],capture_output=True,text=True,timeout=120)
            if completed.returncode:
                raise RuntimeError(f"Native ns-3 failed ({completed.returncode}): {completed.stderr[-2000:]}")
            row=json.loads(completed.stdout);row.update(traffic_mode=mode,seed=seed)
            if row["rx_packets"]==0 or len(row["control_trace"])!=32:
                raise InvalidCandidate("Incomplete packet delivery/control simulation")
            rows.append(row)
    score=aqm_score(rows,valid=True)
    return {"native_score":score.score,"native_unit":score.metrics["native_unit"],"runs":rows,
            "control_trace_columns":["time_seconds","queue_packets","observed_mean_delay_seconds","cumulative_queue_drops",
                                     "cumulative_queue_marks","recent_utilization","minimum_threshold_packets","maximum_threshold_packets"],
            "simulator":"ns-3.44 TCP NewReno dumbbell with RED/ECN and FlowMonitor",
            "controller_scope":"new frozen threshold feedback policy evaluated in packet-level closed loop every 0.25s"}
