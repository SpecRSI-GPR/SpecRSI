#!/usr/bin/env python3
"""Run local coding agents or frozen patch files through official native grading."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml
from specrsi.benchmarks.swe_native import client, load_rows, public_problem, ensure_image, evaluate_patch, registered_tracks
from specrsi.coding_agent import RepositoryAgent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/coding/qwen_swe_smoke.yaml')
    p.add_argument('--ids',nargs='+')
    p.add_argument('--track',choices=['selection60','verified500','lite_audit_filtered','lite_audit_disjoint_verified500'])
    p.add_argument('--gold-control',action='store_true')
    p.add_argument('--predictions',help='Official JSONL with instance_id and model_patch')
    p.add_argument('--output')
    a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text())
    output=Path(a.output or config['output']);output.mkdir(parents=True,exist_ok=True)
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2))
    ids=a.ids or (registered_tracks()[a.track] if a.track else config['instance_ids'])
    rows=load_rows()
    if config.get('frozen_instance'):
        frozen=config['frozen_instance']
        if ids != [frozen['instance_id']]:raise ValueError('Frozen native snapshot must match the sole requested task')
        rows={frozen['instance_id']:frozen}
    c=client(config.get('docker_host','unix:///tmp/specrsi-podman-api/podman.sock'))
    predictions={}
    if a.predictions:
        predictions={x['instance_id']:x for x in map(json.loads,Path(a.predictions).read_text().splitlines())}
    model=dict(config['model']);lock=json.loads(Path(model['lock']).read_text());model.update(lock)
    agent=RepositoryAgent(model,**config.get('agent',{}))
    results=[]
    for i,key in enumerate(ids):
        row=rows[key];folder=output/key;folder.mkdir(exist_ok=True)
        try:
            receipt=ensure_image(row,c)
            (folder/'image.json').write_text(json.dumps(receipt,indent=2))
            if a.gold_control:
                patch=row['patch'];model_name='gold-control'
            elif a.predictions:
                patch=predictions[key]['model_patch'];model_name=predictions[key].get('model_name_or_path','submitted')
            else:
                container=c.containers.run(receipt['image_id'],command='tail -f /dev/null',detach=True,network_mode='none')
                try:
                    actor=agent.solve(public_problem(row),container,seed=config.get('seed',0)+i*1000,
                                      trace_path=folder/'actor_progress.json')
                finally:
                    container.remove(force=True)
                (folder/'agent.json').write_text(json.dumps(actor,indent=2,ensure_ascii=False))
                patch=actor['patch'];model_name=model.get('served_model_name','Qwen3.5-35B-A3B')
            (folder/'patch.diff').write_text(patch)
            prediction={'instance_id':key,'model_name_or_path':model_name,'model_patch':patch}
            (folder/'prediction.json').write_text(json.dumps(prediction,indent=2))
            result=evaluate_patch(row,patch,run_id=config.get('run_id','specrsi-native')+'-'+str(config.get('seed',0)),
                model_name=model_name,timeout=config.get('test_timeout',1800),docker_client=c,image_receipt=receipt)
            result['gold_control']=a.gold_control
            (folder/'evaluation.json').write_text(json.dumps(result,indent=2))
            results.append(result)
            print(json.dumps({'instance_id':key,'completed':result['completed'],'resolved':result['resolved']}),flush=True)
        except Exception as exc:
            failure={'instance_id':key,'completed':False,'resolved':False,'score':None,'error':str(exc),'type':type(exc).__name__}
            (folder/'failure.json').write_text(json.dumps(failure,indent=2));results.append(failure)
            print(json.dumps(failure),flush=True)
        summary={'requested':len(ids),'finished':len(results),'completed':sum(r['completed'] for r in results),
                 'resolved':sum(r['resolved'] for r in results),'results':results,'updated_unix':time.time(),
                 'implementation':'official_swebench4.1.0_native_grading_prospective_actor',
                 'gold_control':a.gold_control,'all_complete':len(results)==len(ids) and all(r['completed'] for r in results)}
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
    if not summary['all_complete']:raise SystemExit(1)


if __name__=='__main__':main()
