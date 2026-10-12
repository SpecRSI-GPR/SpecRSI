#!/usr/bin/env python3
"""Register prospective tracks and pre-pull/verify native task images (resumable)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from specrsi.benchmarks.swe_native import load_rows, registered_tracks, ensure_image, client, DOCKER_HOST


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--track', choices=['selection60','verified500','lite_audit_filtered','lite_audit_disjoint_verified500','development23','all'],default='selection60')
    p.add_argument('--ids', nargs='+')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--pull',action='store_true')
    p.add_argument('--docker-host',default=DOCKER_HOST)
    p.add_argument('--min-free-gb',type=float,default=300)
    p.add_argument('--storage-root',default='/tmp/specrsi-podman-overlay-root')
    p.add_argument('--output',default='reports/swe_images.json')
    p.add_argument('--bootstrap-fixtures',action='store_true')
    a=p.parse_args()
    if a.bootstrap_fixtures:
        subprocess.run([sys.executable,'-m','venv','/tmp/specrsi-httpbin-venv'],check=True)
        subprocess.run(['/tmp/specrsi-httpbin-venv/bin/python','-m','pip','install','-r','configs/coding/requirements-httpbin.txt'],check=True)
        return
    tracks=registered_tracks()
    Path('configs/coding').mkdir(parents=True,exist_ok=True)
    Path('configs/coding/prospective_swe_tracks.json').write_text(json.dumps(tracks,indent=2)+'\n')
    rows=load_rows()
    ids=a.ids or (sorted(set(tracks['verified500'])|set(tracks['lite_audit_filtered'])) if a.track=='all' else tracks[a.track])
    if any(x not in rows for x in ids):raise ValueError('Unknown instance ID')
    if not a.pull:
        print(json.dumps({'track':a.track,'count':len(ids),'ids':ids,'pull':False},indent=2));return
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    report=json.loads(output.read_text()) if output.exists() else {'images':{},'failures':{}}
    report.update(track=a.track,requested_ids=ids,docker_host=a.docker_host,complete=False)
    def one(key):
        if shutil.disk_usage(a.storage_root).free < a.min_free_gb*1024**3:
            raise RuntimeError('Configured free disk reserve reached; resume after adding space')
        receipt=report['images'].get(key)
        c=client(a.docker_host)
        if receipt:
            try:
                image=c.images.get(receipt['alias'])
                if image.id==receipt['image_id']:return receipt
            except Exception:pass
        return ensure_image(rows[key],c)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(one,key):key for key in ids}
        for f in as_completed(pending):
            key=pending[f]
            try:
                report['images'][key]=f.result();report['failures'].pop(key,None)
                print(f'READY {key} ({len(report["images"])} images)',flush=True)
            except Exception as exc:
                report['failures'][key]={'error':str(exc),'type':type(exc).__name__}
                print(f'FAILED {key}: {exc}',flush=True)
            report['updated_unix']=time.time()
            temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(output)
    report['complete']=all(key in report['images'] and key not in report['failures'] for key in ids)
    output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
