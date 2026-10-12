#!/usr/bin/env python3
"""Install/check CPU simulators and freeze all prospective scenario manifests."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/science/registration.json")
    args=parser.parse_args()
    if args.install:
        subprocess.run([sys.executable,"-m","pip","install","pybamm==25.12.2","numpy==1.26.4","scipy==1.17.1","numba==0.66.0"],check=True)
        # The ns3 wheel includes many Boost headers: local disk avoids excessive
        # network-filesystem metadata operations. Native C++ needs no cppyy import.
        native=Path("/opt/specrsi-science/ns3")
        subprocess.run([sys.executable,"-m","pip","install","--target",str(native),"--no-deps","ns3==3.44.post0"],check=True)
        import sysconfig
        pth=Path(sysconfig.get_paths()["purelib"])/"specrsi_science_ns3.pth"
        pth.write_text(f"import sys; sys.path.insert(0, {str(native)!r})\n")
        sys.path.insert(0,str(native))
    from specrsi.science import DOMAINS,registration
    from specrsi.science.registry import case_count
    from specrsi.science.aqm import native_executable
    import os
    os.environ["PYBAMM_DISABLE_TELEMETRY"]="true"
    import pybamm,scipy,numba
    binary=native_executable()
    import hashlib
    result={"author_exact":False,"replication_scope":"prospective_new_scenarios",
            "native_aqm_binary":str(binary),"native_aqm_binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
            "registrations":{role:{d:[registration(d,role,i) for i in range(case_count(d))] for d in DOMAINS}
                             for role in ("development","selection","audit")}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"ready":True,"domains":list(DOMAINS),"scenario_sets":3,"manifest":str(args.output),"ns3_binary":str(binary)}))


if __name__=="__main__":
    main()
