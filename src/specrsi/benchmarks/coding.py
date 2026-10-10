"""Native Polyglot language tests with candidate edits restricted to solutions."""
from __future__ import annotations
import json,re,shutil,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
from .base import ScoreResult,MissingArtifactError
from .sandbox import ROOT,run_isolated

def solution_context(reference: dict) -> dict[str,str]:
    exercise=(ROOT/reference['exercise_path']).resolve()
    if exercise.name=='counter' and '/go/' in str(exercise):
        return {'counter_test.go':(exercise/'counter_test.go').read_text(),'interface.go':(exercise/'interface.go').read_text()}
    return {name:(exercise/name).read_text() for name in reference['files']['solution'] if (exercise/name).is_file()}

def gold_solution(reference:dict) -> dict[str,str]:
    exercise=ROOT/reference['exercise_path']; files=reference['files']
    result=solution_context(reference)
    if exercise.name=='counter' and '/go/' in str(exercise):
        return {'counter_test.go':(exercise/'.meta/example.go').read_text()}
    for example in files.get('example',[]):
        suffix=Path(example).suffix
        choices=[s for s in files['solution'] if Path(s).suffix==suffix]
        same_name=[s for s in choices if Path(s).name==Path(example).name]
        if suffix=='.java': target='src/main/java/'+Path(example).name
        elif same_name: target=same_name[0]
        elif len(choices)==1: target=choices[0]
        else: raise ValueError(f'Cannot map native gold source {example}: {choices}')
        result[target]=(exercise/example).read_text()
    if (exercise/'.meta/Cargo-example.toml').exists():
        result['Cargo.toml']=(exercise/'.meta/Cargo-example.toml').read_text()
    return result

def polyglot_score(solution: dict[str,str], reference: dict, *, timeout=180) -> ScoreResult:
    exercise=(ROOT/reference['exercise_path']).resolve()
    if not exercise.is_relative_to(ROOT/'data/raw/polyglot'): raise ValueError('Exercise outside pinned corpus')
    allowed=set(reference['files']['solution'])
    language=exercise.relative_to(ROOT/'data/raw/polyglot/source').parts[1]
    counter=language=='go' and exercise.name=='counter'
    if counter: allowed={'counter_test.go'}
    if not solution: return ScoreResult(0.,{'reason':'Empty solution'})
    for name,text in solution.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not isinstance(text,str) or len(text)>1000000:
            return ScoreResult(0.,{'reason':'Invalid solution path or content'})
        helper=language=='java' and p.parts[:3]==('src','main','java') and p.suffix=='.java'
        if name not in allowed and not helper: return ScoreResult(0.,{'reason':'Edits must be solution files only'})
    with tempfile.TemporaryDirectory(prefix='specrsi-polyglot-') as directory:
        work=Path(directory)/'exercise'; shutil.copytree(exercise,work)
        # Gold reference implementations are excluded from candidate execution.
        shutil.rmtree(work/'.meta',ignore_errors=True)
        for name,text in solution.items():
            target=work/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(text)
        mounts={}; env={}; timeout=float(timeout); memory_mb=16384
        if language=='python':
            command=['/usr/local/bin/python','-m','pytest','-q','--junitxml=/work/result.xml']
        elif language=='javascript':
            local=Path('/opt/specrsi-polyglot/npm/node_modules')
            dependency=local if local.is_dir() else ROOT/'runtimes/polyglot/npm/node_modules'
            mounts={str(dependency):'/deps/node_modules','/opt/node':'/opt/node'}
            env={'PATH':'/opt/node/bin:/usr/local/bin:/usr/bin:/bin','NODE_PATH':'/deps/node_modules'}
            env['NODE_OPTIONS']='--max-old-space-size=256'
            memory_mb=262144  # V8 workers reserve large virtual cages; heaps are capped above.
            (work/'node_modules').symlink_to('/deps/node_modules')
            for test in work.glob('*.spec.js'): test.write_text(re.sub(r'\bxtest\(', 'test(',test.read_text()))
            command=['/opt/node/bin/node','/deps/node_modules/jest/bin/jest.js','--runInBand','--json','--outputFile=/work/result.json']
        elif language=='go':
            env={'GOPATH':'/tmp/go','GOCACHE':'/tmp/go-build','GOPROXY':'off','GOTOOLCHAIN':'local','GOROOT':'/usr/lib/golang','GOTELEMETRY':'off'}
            command=['go','test','-json','./...']
            if counter:
                rows=[]
                for implementation in ['1','2','3','4']:
                    run=run_isolated(command,work,timeout=timeout/4,memory_mb=memory_mb,
                                     environment={**env,'COUNTER_IMPL':implementation})
                    events=[]
                    for line in run['stdout'].splitlines():
                        try: events.append(json.loads(line))
                        except ValueError: pass
                    completed=[e for e in events if e.get('Test') and e.get('Action') in {'pass','fail'}]
                    failures=sum(e.get('Action')=='fail' for e in completed)
                    ok=bool(completed) and not run['timeout'] and ((run['exit_code']==0 and failures==0) if implementation=='4' else (run['exit_code']==1 and failures>0))
                    rows.append({'implementation':implementation,'correct_classification':ok,'tests':len(completed),'failures':failures,'wall_seconds':run['wall_seconds']})
                return ScoreResult(float(all(r['correct_classification'] for r in rows)),{'language':'go','metric_variant':'native_counter_test_suite_mutation_classification','tests':sum(r['tests'] for r in rows),'implementations':rows})
        elif language=='rust':
            mounts={str(ROOT/'runtimes/polyglot/cargo'):'/cargo'}
            env={'CARGO_HOME':'/cargo','CARGO_NET_OFFLINE':'true'}
            lock=ROOT/'runtimes/polyglot/cargo-locks'/f'{exercise.name}.lock'
            if lock.is_file(): shutil.copy(lock,work/'Cargo.lock')
            command=['cargo','test','--offline','--','--include-ignored']
        elif language=='cpp':
            if not re.fullmatch(r'[a-z0-9-]+',exercise.name): raise ValueError('Invalid exercise directory')
            command=['/bin/bash','-c',f'set -e\nmkdir /tmp/{exercise.name}\ncp -a . /tmp/{exercise.name}/\ncd /tmp/{exercise.name}\ncmake -S . -B build -DEXERCISM_RUN_ALL_TESTS=1\ncmake --build build -j 2\n./build/{exercise.name} -r junit -o /work/result.xml']
        elif language=='java':
            for test in (work/'src/test').rglob('*.java'):
                test.write_text(re.sub(r'@Disabled(?:\([^)]*\))?\s*\n', '', test.read_text()))
            mounts={str(ROOT/'runtimes/polyglot/java-jars'):'/jars'}
            env={'JAVA_TOOL_OPTIONS':'-XX:ActiveProcessorCount=2 -Xmx512m -XX:CompressedClassSpaceSize=128m'}
            # The original Gradle files specify exactly JUnit5.10.0 and
            # AssertJ3.25.1. Invoke those same native tests directly offline.
            command=['/bin/bash','-c','set -e\nmkdir -p build/classes\nfind src -name "*.java" -print0 | xargs -0 javac -encoding UTF-8 -cp "/jars/*" -d build/classes\njava -cp "/jars/*:build/classes" org.junit.platform.console.ConsoleLauncher execute --scan-classpath --disable-banner --reports-dir=build/test-results/test --fail-if-no-tests']
        else: raise ValueError(f'Unknown Polyglot language {language}')
        result=run_isolated(command,work,timeout=timeout,memory_mb=memory_mb,mounts=mounts,environment=env)
        tests=failures=0
        reports=[work/'result.xml'] if language in {'python','cpp'} else list((work/'build/test-results/test').glob('TEST-*.xml')) if language=='java' else []
        for report in reports:
            if report.exists():
                root=ET.parse(report).getroot()
                suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
                tests+=sum(int(s.attrib.get('tests',0))-int(s.attrib.get('skipped',0)) for s in suites)
                failures+=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
        if language=='javascript' and (work/'result.json').exists():
            report=json.loads((work/'result.json').read_text()); tests=report['numPassedTests']+report['numFailedTests']; failures=report['numFailedTests']
        elif language=='go':
            rows=[]
            for line in result['stdout'].splitlines():
                try: rows.append(json.loads(line))
                except ValueError: pass
            tests=sum(r.get('Action')=='pass' and bool(r.get('Test')) for r in rows)
            failures=sum(r.get('Action')=='fail' for r in rows)
        elif language=='rust':
            matches=re.findall(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed',result['stdout'])
            tests=sum(int(p)+int(f) for p,f in matches); failures=sum(int(f) for _,f in matches)
        passed=result['exit_code']==0 and not result['timeout'] and tests>0 and failures==0
        return ScoreResult(float(passed),{'language':language,'passed':passed,'tests':tests,'failures':failures,
            **{k:v for k,v in result.items() if k not in {'stdout','stderr'}},'stdout':result['stdout'][-8000:],'stderr':result['stderr'][-4000:]})
