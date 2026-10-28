from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import resource
import platform
import sys
from functools import lru_cache
from pathlib import Path
import time

from .common import InvalidCandidate

DOMAINS = ("pybamm", "aqm", "fdtd", "reactor", "irp")
ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=512)
def _content(path: str,mtime_ns: int,size: int) -> bytes:
    return Path(path).read_bytes()


def _read(path: Path) -> bytes:
    stat=path.stat()
    return _content(str(path),stat.st_mtime_ns,stat.st_size)


@lru_cache(maxsize=32)
def _distribution(name: str):
    try:return importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:return None


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def case_count(domain: str, scenario_set: str="selection") -> int:
    if domain not in DOMAINS:
        raise ValueError("Unknown science domain")
    if scenario_set == "reference":
        if domain != "reactor":
            raise ValueError("Reference profile is registered only for Reactor")
        return json.loads(_read(ROOT/"configs/science/reference/reactor.json"))["case_count"]
    if scenario_set not in {"development", "selection", "audit"}:
        raise ValueError("Unknown registered scenario set")
    return json.loads(_read(ROOT/"configs/science/cases"/f"{domain}.json"))["case_count"]


def load_config(domain: str, scenario_set: str="selection", case_id: int=0) -> dict:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown science domain {domain}")
    if scenario_set == "reference":
        if domain != "reactor":
            raise ValueError("Reference profile is registered only for Reactor")
        bundle=json.loads(_read(ROOT/"configs/science/reference/reactor.json"))
        if type(case_id) is not int or not 0 <= case_id < bundle["case_count"]:
            raise ValueError("Unregistered science case ID")
        case=bundle["cases"][case_id]
        if case["case_id"] != case_id:
            raise ValueError("Mismatched registered Reactor case ID")
        return {**bundle["common"], **case}
    if scenario_set not in {"development","selection","audit"}:
        raise ValueError("Unknown registered scenario set")
    folder=ROOT/"configs/science"
    if scenario_set!="selection":
        folder/=scenario_set
    c=json.loads(_read(folder / f"{domain}.json"))
    cases=json.loads(_read(ROOT/"configs/science/cases"/f"{domain}.json"))
    if type(case_id) is not int or not 0<=case_id<cases["case_count"]:
        raise ValueError("Unregistered science case ID")
    patch=cases["cases"][case_id]["patch"]
    c["case_id"]=case_id
    c["registered_case_patch"]=patch
    if domain=="pybamm":c["initial_soc"]+=patch["initial_soc_delta"]
    elif domain=="aqm":c["seeds"]=[s+patch["seed_offset"] for s in c["seeds"]]
    elif domain=="fdtd":
        c["output_offset_um"]+=patch["output_offset_um_delta"]
        c["wavelengths_um"]=[v+patch["wavelength_um_delta"] for v in c["wavelengths_um"]]
    elif domain=="reactor":
        c["initial_temperature_k"]+=patch["initial_temperature_k_delta"]
        c["feed_temperature_k"]+=patch["feed_temperature_k_delta"]
    else:
        c["seeds"]=[s+patch["seed_offset"] for s in c["seeds"]]
        c["demand_means"]=[v*patch["demand_mean_scale"] for v in c["demand_means"]]
    return c


def registration(domain: str, scenario_set: str="selection", case_id: int=0) -> dict:
    c = load_config(domain,scenario_set,case_id)
    sources = {p.name:hashlib.sha256(_read(p)).hexdigest() for p in sorted(Path(__file__).parent.iterdir())
               if p.suffix in {".py",".cc"}}
    versions = {}
    distribution_records = {}
    for name in ("numpy","scipy","numba","pybamm","pybammsolvers","casadi","ns3","cppyy"):
        distribution=_distribution(name)
        if distribution is None:
            versions[name] = None
        else:
            # Cache bytes by actual file mtime/size, not by candidate or score.
            # Editing installed metadata invalidates the fingerprint immediately.
            metadata_path=Path(distribution._path)/"METADATA"
            metadata_bytes=_read(metadata_path)
            versions[name]=next(line[9:].strip() for line in metadata_bytes.decode().splitlines() if line.startswith("Version: "))
            record_path=Path(distribution._path)/"RECORD"
            record=_read(record_path) if record_path.exists() else b""
            distribution_records[name]=hashlib.sha256(record).hexdigest()
    runtime={"python":sys.version,"machine":platform.machine(),"libc":platform.libc_ver(),
             "byteorder":sys.byteorder,"threads":1,"cuda_visible_devices":"", "numerical_precision":"float64"}
    payload = {"config":c,"scenario_set":scenario_set,"source_sha256":sources,"dependencies":versions,
               "dependency_distribution_records_sha256":distribution_records,"runtime_contract":runtime}
    return {"evaluator_identity":hashlib.sha256(canonical(payload).encode()).hexdigest(),**payload}


def evaluate(domain: str, candidate: dict, *, scenario_set: str="selection", case_id: int=0) -> dict:
    c = load_config(domain,scenario_set,case_id)
    manifest = registration(domain,scenario_set,case_id)
    started, cpu = time.monotonic(),time.process_time()
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu=children.ru_utime+children.ru_stime
    try:
        result = importlib.import_module(f"specrsi.science.{domain}").simulate(candidate,c)
        if not math.isfinite(result["native_score"]):
            raise RuntimeError("Simulator returned a nonfinite score")
        result.update(valid=True,complete=True,status="complete",violations=[])
        # Fixed monotone map for SpecRSI's bounded Outcome contract; native value is retained.
        result["specrsi_score"] = 0.5 + math.atan(result["native_score"])/math.pi
    except InvalidCandidate as exc:
        result = {"valid":False,"complete":True,"status":"invalid_candidate","violations":[str(exc)],
                  "native_score":None,"specrsi_score":0.0}
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"domain":domain,"scenario_set":scenario_set,"case_id":case_id,
            "replication_scope":c.get("replication_scope", "registered_native_scientific_scenarios"),"author_exact":False,
            "profile":"full","candidate":candidate,"candidate_sha256":hashlib.sha256(canonical(candidate).encode()).hexdigest(),
            "evaluator_identity":manifest["evaluator_identity"],**result,
            "wall_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu+children.ru_utime+children.ru_stime-child_cpu}
