#!/usr/bin/env python3
"""Resolve trusted native dependencies and exercise all six language graders."""
from pathlib import Path
import sys,json,os,subprocess,shutil,tempfile,argparse,hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from specrsi.benchmarks.data import load_examples,sha256_file
from specrsi.benchmarks.coding import gold_solution,polyglot_score

def main():
    p=argparse.ArgumentParser(); p.add_argument('--verify-only',action='store_true'); p.add_argument('--workers',type=int,default=6); p.add_argument('--limit',type=int); a=p.parse_args()
    examples=[e for split in ['development','selection','audit'] for e in load_examples('polyglot',split,verify=split=='development')]
    runtime=ROOT/'runtimes/polyglot'; runtime.mkdir(parents=True,exist_ok=True)
    cargo=runtime/'cargo'; cargo.mkdir(exist_ok=True)
    if not a.verify_only:
        import requests
        for artifact in json.loads((ROOT/'configs/datasets/polyglot_java.lock.json').read_text()):
            path=ROOT/artifact['path']; path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                response=requests.get(artifact['url'],timeout=60); response.raise_for_status(); path.write_bytes(response.content)
            if sha256_file(path)!=artifact['sha256']: raise ValueError('Java runtime hash changed: '+str(path))
        npm=Path('/opt/specrsi-polyglot/npm'); npm.mkdir(parents=True,exist_ok=True)
        for name in ['package.json','package-lock.json']: shutil.copy(ROOT/'configs/runtimes/polyglot'/name,npm/name)
        if not (npm/'node_modules/jest/bin/jest.js').exists():
            subprocess.run(['npm','ci','--ignore-scripts'],cwd=npm,check=True)
        seen=set()
        for e in examples:
            lang=e.metadata['language']; src=ROOT/e.reference['exercise_path']
            if lang != 'rust': continue
            recipe=src/('Cargo.toml' if lang=='rust' else 'build.gradle')
            key=(lang,recipe.read_text())
            if key in seen: continue
            seen.add(key)
            with tempfile.TemporaryDirectory(prefix='specrsi-polyglot-deps-') as temporary:
                work=Path(temporary)/'exercise'; shutil.copytree(src,work)
                for name,text in gold_solution(e.reference).items(): (work/name).write_text(text)
                env=os.environ.copy()
                if lang=='rust':
                    env['CARGO_HOME']=str(cargo); cmd=['cargo','fetch']
                else:
                    env['GRADLE_USER_HOME']=str(runtime/'gradle-cache')
                    cmd=[str(runtime/'gradle/gradle-8.7/bin/gradle'),'--no-daemon','--max-workers=2','compileTestJava']
                result=subprocess.run(cmd,cwd=work,env=env,capture_output=True,text=True,timeout=300)
                logfile=ROOT/'logs'/f'polyglot-deps-{lang}-{src.name}.log'; logfile.write_text(result.stdout+result.stderr)
                print(json.dumps({'dependency':e.id,'exit_code':result.returncode}),flush=True)
                if result.returncode: raise RuntimeError(f'Dependency preparation failed: {logfile}')
                if lang=='rust' and (work/'Cargo.lock').exists():
                    locks=runtime/'cargo-locks'; locks.mkdir(exist_ok=True); shutil.copy(work/'Cargo.lock',locks/(src.name+'.lock'))
    if a.limit:
        examples=[next(e for e in examples if e.metadata['language']==lang) for lang in ['python','javascript','go','rust','cpp','java']][:a.limit]
    def verify(e):
        try: score=polyglot_score(gold_solution(e.reference),e.reference,timeout=240); row={'id':e.id,'language':e.metadata['language'],'score':score.score,'metrics':dict(score.metrics)}
        except Exception as exc: row={'id':e.id,'language':e.metadata['language'],'error':str(exc),'score':None}
        print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True); return row
    with ThreadPoolExecutor(max_workers=a.workers) as pool: rows=list(pool.map(verify,examples))
    report={'purpose':'native_gold_controls_not_model_results','total':len(rows),'passed':sum(r['score']==1 for r in rows),'rows':rows,
            'runtime':'offline bubblewrap; JUnit5.10.0/AssertJ3.25.1 direct launcher (same native tests as Gradle)',
            'enabled_tests':'Java @Disabled removed, JS xtest enabled, Rust ignored tests included, C++ EXERCISM_RUN_ALL_TESTS=1'}
    (ROOT/'reports/polyglot_native.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    if report['passed']!=len(rows): raise SystemExit(1)
if __name__=='__main__': main()
