from dataclasses import replace
import json
from pathlib import Path
import sys
import pytest

from specrsi.baselines.extended_common import Design,Evaluation,Generated,ItemResult,ExperimentBudget,RunContext
from specrsi.baselines.native_backend import NativeInstructionBackend,native_splits
from specrsi.baselines.emas import Hypothesis,EMASConfig,run_emas,validate_revision,OPERATIONS
from specrsi.benchmarks.base import BenchmarkExample


def test_native_swe_full_registry_disjoint_before_limiting():
    if not all(Path(path).is_file() for path in (
        'data/raw/swe_verified/test-00000-of-00001.parquet',
        'data/raw/swe_lite/test-00000-of-00001.parquet',
        'data/raw/swe_lite/dev-00000-of-00001.parquet',
    )):
        pytest.skip('Download pinned SWE-bench inputs with scripts/prepare_data.py')
    config={'dataset':'swe_verified','selection_track':'verified500','audit_track':'lite_audit_disjoint_verified500'}
    parts=native_splits(config)
    assert [len(parts[k]) for k in ('development','selection','audit')]==[23,500,207]
    assert {e.dataset for values in parts.values() for e in values}=={'swe_verified'}
    groups=[{x.id for x in p} for p in parts.values()]
    assert not any(a&b for i,a in enumerate(groups) for b in groups[i+1:])
    config.update(audit_track='lite_audit_filtered',data={'selection':1,'audit':1,'development':1})
    with pytest.raises(ValueError):native_splits(config)


def test_native_optimizer_feedback_cannot_expose_gold_or_sealed_tests():
    from specrsi.baselines.gepa import optimization_reference
    raw={'instance_id':'case','repo':'x/y','base_commit':'abc','problem_statement':'public issue',
         'patch':'SECRET_GOLD','test_patch':'SECRET_TEST','FAIL_TO_PASS':['SEALED_CASE'],'hints_text':'SEALED_HINT'}
    item=BenchmarkExample('case','swe_verified','development','public issue',raw)
    text=json.dumps(optimization_reference(item))
    assert 'public issue' in text and not any(s in text for s in ('SECRET','SEALED','test_patch','hints_text'))


def test_native_emas_requires_recurrence_and_real_paired_gain():
    class Backend:
        def evaluate(self,design,items,seed,stage,token_cap):
            return Evaluation(design.uid,tuple(ItemResult(e.id,float(design.payload['system_prompt']=='better'),2) for e in items),seed,stage)
        def diagnose(self,design,item,result,seed,token_cap):
            return Generated([{'objective':'accuracy','defect':'missing-check','operation':'prompt-only','location':'strategy'}],1)
        def propose(self,parent,feedback,seed,token_cap):
            return Generated(Design({'system_prompt':'better'},parent.uid),1)
    initial=Design({'system_prompt':'initial'})
    examples=[BenchmarkExample(str(i),'polyglot','development','task'+str(i),{}) for i in range(3)]
    ctx=RunContext(Backend(),examples[:1],budget=ExperimentBudget(1000,20))
    result=run_emas(ctx,initial,examples[1:],EMASConfig(epochs=1,thresholds={op:2 for op in OPERATIONS}))
    assert result['metadata']['accepted_versions']==1
    pairs=[e for e in ctx.evaluations if e.stage=='paired_validation']
    assert len(pairs)==2 and pairs[0].seed==pairs[1].seed and pairs[1].score>pairs[0].score
    with pytest.raises(ValueError,match='prompt-only'):
        validate_revision(initial,Design({'system_prompt':'better'},initial.uid),Hypothesis('accuracy','x','add-node','strategy'))


@pytest.mark.skipif(sys.platform != 'linux', reason='Native Polyglot execution requires Linux bubblewrap namespaces')
def test_native_polyglot_adapter_runs_actual_python_tests():
    if not Path('data/processed/polyglot/selection.jsonl').is_file():
        pytest.skip('Download pinned Polyglot inputs with scripts/prepare_data.py')
    from specrsi.benchmarks.coding import gold_solution
    from specrsi.task_pipelines import TaskPipelineExecutor
    from specrsi.models.client import ModelReply
    rows=[BenchmarkExample.from_dict(json.loads(line)) for line in Path('data/processed/polyglot/selection.jsonl').read_text().splitlines()]
    example=next(x for x in rows if '/python/' in x.reference['exercise_path'])
    gold=gold_solution(example.reference)
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [1]*5
    class GoldControl:
        def generate(self,*args,**kwargs):
            return ModelReply(json.dumps({'files':gold}),5,2,.1,'stop','control','native-control')
    backend=object.__new__(NativeInstructionBackend)
    backend.dataset='polyglot';backend.lock={'revision':'control'}
    backend.generation={'max_tokens':10,'test_timeout_seconds':60}
    backend.client=GoldControl();backend.tokenizer=Tokenizer()
    backend.execution_cap=100;backend.calls=[];backend.native_receipts=[]
    backend.executor=TaskPipelineExecutor(backend.client,backend.tokenizer,backend.generation)
    design=Design({'system_prompt':'Implement requested behavior'})
    result=backend.evaluate(design,[example],0,'validation',100)
    assert result.complete and result.score==1 and result.tokens==7
    metrics=result.items[0].trace[0]['output']['metrics']
    assert metrics['tests']>0 and metrics['failures']==0
    assert backend.calls[0]['stage']=='solution'
    interrupted=backend.evaluate(design,[example],0,'audit',1)
    assert not interrupted.complete and interrupted.tokens==0


@pytest.mark.parametrize('platform,available,message', [
    ('darwin', '/usr/bin/bwrap', 'requires Linux'),
    ('linux', None, 'requires bubblewrap'),
])
def test_native_sandbox_rejects_unsupported_runtime_before_execution(monkeypatch,tmp_path,platform,available,message):
    from specrsi.benchmarks import sandbox
    from specrsi.benchmarks.base import MissingArtifactError
    monkeypatch.setattr(sandbox.sys,'platform',platform)
    monkeypatch.setattr(sandbox.shutil,'which',lambda name:available)
    def forbidden(*args,**kwargs):
        pytest.fail('An unsupported sandbox must not launch candidate code')
    monkeypatch.setattr(sandbox.subprocess,'Popen',forbidden)
    with pytest.raises(MissingArtifactError,match=message):
        sandbox.run_isolated(['/usr/local/bin/python','untrusted.py'],tmp_path)
