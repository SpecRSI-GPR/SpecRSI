#!/usr/bin/env python3
"""Run paired real-simulator fixed-pool / online P-U mechanism experiments."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from specrsi.mechanisms import run_experiment

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',required=True)
p.add_argument('--output')
p.add_argument('--plan',action='store_true')
a=p.parse_args()
c=json.loads(Path(a.config).read_text())
if a.output:c['output']=a.output
if a.plan:print(json.dumps(c,indent=2))
else:run_experiment(c)
