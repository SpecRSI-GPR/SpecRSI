"""Bounded local-model repository editor with a recorded tool/usage transcript.

Only public issue text and the base repository are supplied. Gold patches and
official FAIL_TO_PASS/PASS_TO_PASS lists are never passed to this actor.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import PurePosixPath,Path
import re
import shlex
import time
from specrsi.models.client import LocalModelClient

SYSTEM = '''You fix a real software issue inside a repository at /testbed.
Reply with exactly one JSON object for one tool action per turn:
{"action":"search","query":"literal text"} searches tracked source files.
{"action":"read","path":"relative/file.py","start":1,"end":180} reads numbered lines.
{"action":"replace","path":"relative/file.py","old":"exact existing text","new":"replacement"} replaces exactly one occurrence.
{"action":"write","path":"relative/file.py","content":"full file"} creates or replaces a file.
{"action":"run","command":"python -m pytest path/to/test.py -q"} runs a short check in the repository.
{"action":"finish"} submits the current repository diff.
Inspect relevant source, make the smallest correct fix, and check it if practical.
Do not merely explain a patch: use replace/write to apply it. Do not change .git.
The evaluation tests are separate and unavailable. Your final diff is evaluated in a fresh container.'''


def parse_action(text):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    value=json.loads(text)
    if not isinstance(value,dict) or value.get('action') not in {'search','read','replace','write','run','finish'}:
        raise ValueError('Expected one supported JSON action')
    return value


def safe_path(path):
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or '.git' in p.parts or not p.parts:
        raise ValueError('Path must be relative within the repository, excluding .git')
    return str(p)


class RepositoryAgent:
    def __init__(self, model_config, *, max_steps=16,max_tokens=50000,tool_timeout=90,max_observation_chars=12000):
        self.client=LocalModelClient(model_config.get('base_url','http://127.0.0.1:18085'),
                                     model_config.get('served_model_name','Qwen3.5-35B-A3B'))
        self.model=model_config;self.max_steps=max_steps;self.max_tokens=max_tokens
        self.tool_timeout=tool_timeout;self.max_observation_chars=max_observation_chars

    def _exec(self,container,argv):
        result=container.exec_run(argv,workdir='/testbed')
        return {'exit_code':result.exit_code,'output':result.output.decode(errors='replace')[:self.max_observation_chars]}

    def tool(self,container,action):
        kind=action['action']
        if kind=='search':
            query=str(action['query'])
            if not query or len(query)>300:raise ValueError('Search query must be 1–300 characters')
            return self._exec(container,['git','grep','-n','-i','-F','--',query])
        if kind=='run':
            command=str(action['command'])
            return self._exec(container,['timeout','-k','5',str(self.tool_timeout),'bash','-lc',
                                         'source /opt/miniconda3/bin/activate 2>/dev/null || true; conda activate testbed 2>/dev/null || true; cd /testbed; '+command])
        path=safe_path(str(action['path']))
        # Resolve symlinks in the container before file operations.
        script='''import pathlib,sys,json
p=pathlib.Path('/testbed')/sys.argv[1]
assert str(p.resolve()).startswith('/testbed/') and '.git' not in p.resolve().parts
a=json.loads(sys.argv[2]); k=a['action']
if k=='read':
 start=max(1,int(a.get('start',1)));end=min(start+249,int(a.get('end',start+179)))
 print(''.join('%d: %s'%(i,line) for i,line in enumerate(p.read_text().splitlines(True),1) if start<=i<=end))
elif k=='replace':
 text=p.read_text();old=a['old'];assert old and text.count(old)==1,'old text must match exactly once'
 p.write_text(text.replace(old,a['new'],1));print('Replacement applied')
elif k=='write':
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(a['content']);print('File written')
'''
        result=self._exec(container,['python','-c',script,path,json.dumps(action)])
        if kind=='write' and result['exit_code']==0:
            staged=self._exec(container,['git','add','--intent-to-add','--',path])
            if staged['exit_code']!=0:return staged
        return result

    def solve(self,problem,container,*,seed=0,trace_path=None):
        forbidden={'patch','test_patch','FAIL_TO_PASS','PASS_TO_PASS','hints_text'}
        if forbidden & problem.keys():raise ValueError('Actor received evaluator-only fields')
        health=self.client.health()
        if health.get('protocol_version',0)<2:raise ValueError('Requires metered protocol2 local model')
        if self.model.get('revision') and health.get('revision')!=self.model['revision']:
            raise ValueError('Local model revision mismatch')
        from .benchmarks.swe_native import verified_image_base
        base_identity=verified_image_base(container,problem['base_commit'])
        if base_identity['empty_setup_commit']:
            reset=self._exec(container,['git','reset','--hard',problem['base_commit']])
            if reset['exit_code']:raise ValueError('Unable to reset the empty image setup commit')
        tree=self._exec(container,['git','ls-files'])
        issue='Repository: '+problem['repo']+'\nIssue:\n'+problem['problem_statement']+'\nTracked files:\n'+tree['output'][:6000]
        history=[];transcript=[];tokens=0;started=time.monotonic();reason='max_steps'
        for step in range(self.max_steps):
            system=SYSTEM+'\nRepository-solving strategy:\n'+str(self.model.get('strategy','Inspect the relevant source and apply a correct minimal fix.'))
            messages=[{'role':'system','content':system},{'role':'user','content':issue}]+history[-8:]
            max_output=int(self.model.get('max_tokens',1536))
            # UTF-8 byte count is a conservative tokenizer-independent bound.
            if self.max_tokens-tokens < sum(len(m['content'].encode())+64 for m in messages)+max_output:
                reason='remaining_token_budget';break
            reply=self.client.generate(messages,max_tokens=max_output,temperature=self.model.get('temperature',.2),
                seed=seed+step,max_seconds=self.model.get('max_seconds',120),total_token_cap=self.max_tokens-tokens)
            tokens+=reply.prompt_tokens+reply.completion_tokens
            record={'step':step,'reply':asdict(reply)}
            if trace_path:
                Path(trace_path).write_text(json.dumps({'tokens':tokens,'transcript':transcript+[record],
                    'model_health':health},indent=2,ensure_ascii=False))
            try:
                action=parse_action(reply.text);record['action']=action
                if action['action']=='finish':
                    transcript.append(record);reason='finish';break
                observation=self.tool(container,action)
            except Exception as exc:
                observation={'error':f'{type(exc).__name__}: {exc}'}
            record['observation']=observation;transcript.append(record)
            history.extend([{'role':'assistant','content':reply.text},
                            {'role':'user','content':'Tool result:\n'+json.dumps(observation,ensure_ascii=False)}])
        # A submitted patch must be complete even when tool observations truncate.
        diff=container.exec_run(['git','-c','core.fileMode=false','diff',problem['base_commit']],workdir='/testbed')
        patch=diff.output.decode(errors='strict')
        if diff.exit_code!=0:raise RuntimeError('Unable to obtain final patch')
        return {'instance_id':problem['instance_id'],'patch':patch,'tokens':tokens,'stop_reason':reason,
                'image_base_identity':base_identity,
                'steps':len(transcript),'transcript':transcript,'model_health':health,
                'wall_seconds':time.monotonic()-started,
                'physical_gpu_seconds':sum(x['reply']['elapsed_seconds']*x['reply'].get('gpu_count',1) for x in transcript),
                'implementation':'prospective_local_repository_tool_agent_v1'}


from .experiments import QwenExecutor


class SWERepositoryExecutor(QwenExecutor):
    """Opaque SpecRSI executor; candidate.system_prompt controls the repository actor.

The native harness runs in its pinned interpreter, so a GPU training process
does not need the harness's separate dependency stack. Interrupted native
evaluations never become reusable SpecRSI outcomes.
"""
    def validate_prompt_candidate(self,candidate,tasks,execution_cap):
        if 'workflow' in candidate.artifacts:raise ValueError('SWE currently optimizes repository strategy text')
        if not str(candidate.artifacts.get('system_prompt','')).strip():raise ValueError('Missing repository strategy')
        if execution_cap<2048:raise ValueError('Repository actor execution cap must be at least 2048 tokens')
        for task in tasks:
            if task.metadata.get('dataset') not in ('swe_verified','swe_lite'):
                raise ValueError('SWE executor received another task family')

    def run(self,candidate,task,scope,cap,resume_token=None):
        import subprocess,uuid
        import yaml
        from .core.models import thaw,Outcome
        from .core.execution import ExecutionResult
        if not scope.contains(task):raise ValueError('Task outside frozen scope')
        self.validate_prompt_candidate(candidate,[task],cap)
        generation=thaw(candidate.runtime['generation'])
        reference=thaw(task.expected)
        if reference.get('instance_id')!=task.name or reference.get('problem_statement')!=task.inputs['prompt']:
            raise ValueError('SWE task identity/input mismatch')
        root=Path(__file__).resolve().parents[2]
        output=root/'artifacts/coding/specrsi_executions'/uuid.uuid4().hex
        output.mkdir(parents=True)
        model={**generation,'strategy':str(candidate.artifacts['system_prompt'])}
        config={'instance_ids':[task.name],'seed':int(candidate.runtime['seed']),
            'run_id':'specrsi-search-'+uuid.uuid4().hex[:12],'output':str(output),
            'model':model,'agent':{'max_steps':generation.get('actor_max_steps',16),'max_tokens':cap,
                                 'tool_timeout':generation.get('tool_timeout',90)},
            'test_timeout':generation.get('test_timeout_seconds',1800),'frozen_instance':reference}
        config_path=output/'config.yaml';config_path.write_text(yaml.safe_dump(config))
        interpreter=generation.get('harness_python',str(root/'.envs/swe/bin/python'))
        started=time.monotonic()
        with (output/'runner.log').open('w') as log:
            process=subprocess.run([interpreter,str(root/'scripts/run_swe.py'),'--config',str(config_path)],
                                   cwd=root,stdout=log,stderr=log)
        folder=output/task.name
        actor_file=folder/'agent.json'
        progress_file=folder/'actor_progress.json'
        actor=json.loads((actor_file if actor_file.exists() else progress_file).read_text()) if actor_file.exists() or progress_file.exists() else {}
        actual=int(actor.get('tokens',0))
        records=actor.get('transcript',[])
        for record in records:
            self.physical_records.append({'candidate':candidate.uid,'task':task.uid,'scope':scope.uid,
                                          'kind':'repository_actor',**record['reply']})
        receipt={'executor':'swebench4.1.0-local-repository-agent-v1','cost_unit':'tokens',
                 'measured_tokens':actual,'output':str(output),'runner_exit_code':process.returncode,
                 'gpu_seconds':sum(x['reply']['elapsed_seconds']*x['reply'].get('gpu_count',1) for x in records),
                 'wall_seconds':time.monotonic()-started}
        if actual>cap:raise RuntimeError('Repository actor violated enforced token cap')
        evaluation=folder/'evaluation.json'
        if not evaluation.exists():
            return ExecutionResult('error',actual,receipt=receipt,error='Repository actor or native harness failed; see runner log')
        result=json.loads(evaluation.read_text())
        receipt['native_evaluation']=result
        if not result['completed']:
            return ExecutionResult('interrupted',actual,receipt=receipt,error='Native SWE evaluation incomplete')
        if any(x['reply']['model_revision']!=candidate.runtime['model_revision'] for x in records):
            return ExecutionResult('error',actual,receipt=receipt,error='Frozen model revision mismatch')
        return ExecutionResult('complete',actual,Outcome(result['score'],{'answer':actor.get('patch',''),
            'metrics':{'resolved':result['resolved'],'native_log_dir':result['log_dir']},
            'finish_reason':actor.get('stop_reason','unknown')}),receipt=receipt)
