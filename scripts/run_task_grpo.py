#!/usr/bin/env python3
"""Train/evaluate the separate task-model GRPO baseline; planning is default."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import yaml
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/training/task_grpo/game24.yaml')
    p.add_argument('--execute',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    config=yaml.safe_load(Path(a.config).read_text())
    if not a.execute:print(json.dumps({'kind':'task_model_GRPO_not_SpecRSI_training','config':config,'execution_requested':False},indent=2));return
    from specrsi.learning.task_grpo import run_task_grpo
    print(json.dumps(run_task_grpo(config),indent=2))
if __name__=='__main__':main()
