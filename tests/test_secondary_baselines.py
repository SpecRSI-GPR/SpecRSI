import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from specrsi.baselines.secondary_common import CappedClient
from specrsi.baselines.secondary_gradients import run_gradients
from specrsi.baselines.secondary_mipro import labeled_answer, render_demonstrations, run_mipro_demos
from specrsi.benchmarks.base import BenchmarkExample, ScoreResult
from specrsi.core.budget import BudgetExceeded
from specrsi.models import ModelReply


class FakeModel:
    def __init__(self):
        self.records=[]; self.journal=None; self.client=SimpleNamespace(model='fixture')
        self.model_revision='fixture'; self.max_tokens=1024; self.temperature=0.; self.inputs=[]
    def append(self, row):
        self.records.append(row)
        if self.journal:
            with Path(self.journal).open('a') as file: file.write(json.dumps(row)+'\n')
    def evaluate_example(self, instruction, example):
        self.inputs.append((instruction, example.split))
        self.append({'role':example.split,'example_id':example.id})
        return '42', ScoreResult(.9 if 'new instruction' in instruction else .2,{})
    def call(self, messages, *, role, **kwargs):
        text=json.dumps(messages)
        assert 'AUDIT-SECRET' not in text
        self.append({'role':role})
        if role=='textgrad_backward': return 'Use a clearer general instruction.'
        if role=='textgrad_update': return '<IMPROVED_VARIABLE>new instruction</IMPROVED_VARIABLE>'
        if role=='optoprime_update':
            user=messages[-1]['content']
            variable=re.search(r'\(str\) (system_prompt\w*)=',user).group(1)
            return json.dumps({'reasoning':'Improve strategy.','answer':'','suggestion':{variable:'new instruction'}})
        raise AssertionError(role)
    def summary(self): return {'calls':len(self.records)}


def parts():
    return {role:[BenchmarkExample(role,'aime25',role,'AUDIT-SECRET' if role=='audit' else role+' public question',42)]
            for role in ('development','selection','audit')}


@pytest.mark.parametrize('method',['textgrad','trace-optoprime'])
def test_original_gradient_optimizer_uses_development_graph_and_frozen_audit(tmp_path, method):
    source_name = 'secondary_textgrad' if method == 'textgrad' else 'secondary_trace'
    if not (Path(__file__).resolve().parents[1] / 'third_party' / source_name).is_dir():
        pytest.skip('Download pinned optimizer sources with scripts/prepare_upstream.py')
    model=FakeModel()
    config={'method':method,'dataset':'aime25','output':str(tmp_path/method),'seed_prompt':'old instruction',
            'seed':0,'algorithm':{'steps':1,'batch_size':1}}
    result=run_gradients(config,parts(),model)
    assert result['status']=='complete' and result['selection_score']==.9 and result['audit_score']==.9
    assert model.records[-1]['role']=='audit'
    history=json.loads((Path(config['output'])/'optimization.json').read_text())
    assert history[1]['development_update']['native_records'][0]['example_id']=='development'
    assert json.loads((Path(config['output'])/'selected_candidate.json').read_text())=={'system_prompt':'new instruction'}
    roles=[row['role'] for row in model.records]
    assert ('textgrad_backward' in roles and 'textgrad_update' in roles) if method=='textgrad' else 'optoprime_update' in roles


def test_mipro_demos_reach_native_agent_and_audit_cannot_be_a_demo(tmp_path, monkeypatch):
    dspy = pytest.importorskip('dspy', reason='Optional DSPy optimizer runtime required')
    class Optimizer:
        def __init__(self,**kwargs): assert kwargs['max_labeled_demos']==1
        def compile(self,student,*,trainset,valset,**kwargs):
            student.predictor.demos=[trainset[0]]
            with dspy.context(trace=[]):
                student(question=trainset[0].question)
                assert dspy.settings.trace and dspy.settings.trace[0][0] is student.predictor
            student(question=valset[0].question)
            return student
    monkeypatch.setattr(dspy,'MIPROv2',Optimizer)
    model=FakeModel()
    config={'method':'mipro-demos','dataset':'aime25','output':str(tmp_path/'mipro'),'seed_prompt':'old instruction',
            'algorithm':{'max_labeled_demos':1,'max_bootstrapped_demos':1}}
    result=run_mipro_demos(config,parts(),model)
    assert result['demonstrations']==1 and result['executions_with_demonstrations']==3
    assert all('Development demonstrations' in text for text,role in model.inputs)
    assert all('AUDIT-SECRET' not in text for text,role in model.inputs)
    with pytest.raises(ValueError,match='development'):
        render_demonstrations('instruction',[{'question':'AUDIT-SECRET','answer':'42'}],{'development public question'})
    with pytest.raises(ValueError,match='development'):
        labeled_answer(parts()['audit'][0])


def test_capped_client_charges_native_calls_and_separates_audit_before_gpu():
    physical=[]
    def generate(messages,**kwargs):
        physical.append(kwargs)
        return ModelReply('answer',10,5,.1,'stop','revision',str(len(physical)))
    client=CappedClient(SimpleNamespace(model='fixture',generate=generate),
        SimpleNamespace(apply_chat_template=lambda *a,**kw:[0]*10),
        revision='revision',search_tokens=20,audit_tokens=30)
    client.generate([],max_tokens=8)
    assert client.receipt()['optimization']['spent']==15
    with pytest.raises(BudgetExceeded): client.generate([],max_tokens=8)
    assert len(physical)==1
    client.phase='audit';client.generate([],max_tokens=8)
    assert client.receipt()['audit']['spent']==15 and client.receipt()['optimization']['spent']==15


def test_coding_sealed_answers_are_not_labeled_demonstrations():
    example=BenchmarkExample('dev','swe_verified','development','public issue',{'patch':'SECRET GOLD','test_patch':'SECRET TESTS'})
    assert labeled_answer(example) is None


def test_prepare_secondary_source_checks_archive_and_file_bytes(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import io
    import tarfile
    script = Path(__file__).resolve().parents[1] / 'scripts/prepare_secondary_baselines.py'
    spec = importlib.util.spec_from_file_location('prepare_secondary_fixture', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b'original source'
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode='w:gz') as archive:
        item = tarfile.TarInfo('repo-commit/package.py'); item.size = len(payload)
        archive.addfile(item, io.BytesIO(payload))
    archive_bytes = archive_buffer.getvalue()
    lock = {'commit':'commit','archive_url':'https://fixture.invalid/source.tar.gz',
            'archive_sha256':hashlib.sha256(archive_bytes).hexdigest(),
            'local_path':'third_party/secondary_textgrad',
            'files':{'package.py':hashlib.sha256(payload).hexdigest()}}
    lock_path = tmp_path / 'configs/baselines/secondary_textgrad.lock.json'
    lock_path.parent.mkdir(parents=True); lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(module.requests, 'get', lambda *a, **kw:
                        SimpleNamespace(content=archive_bytes, raise_for_status=lambda:None))
    result = module.prepare('textgrad', tmp_path)
    assert result['action'] == 'downloaded_and_verified'
    assert module.prepare('textgrad', tmp_path, verify_only=True)['action'] == 'verified_existing'
    (tmp_path / lock['local_path'] / 'package.py').write_bytes(b'x' * len(payload))
    with pytest.raises(ValueError, match='changed pinned source'):
        module.prepare('textgrad', tmp_path)
    fresh = tmp_path / 'fresh'
    other_lock = fresh / 'configs/baselines/secondary_textgrad.lock.json'
    other_lock.parent.mkdir(parents=True)
    lock['archive_sha256'] = '0' * 64; other_lock.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='archive checksum'):
        module.prepare('textgrad', fresh)
