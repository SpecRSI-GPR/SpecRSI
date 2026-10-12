from types import SimpleNamespace
import json
import pytest
from specrsi.models import ModelReply
from specrsi.core.models import Candidate, Task, EvaluationScope
from specrsi.task_pipelines import TaskPipelineExecutor
from specrsi.benchmarks.coding import polyglot_score


def executor_case(outputs,cap=10000,*,mapping_output=False):
    calls=[]
    def generate(messages,**kwargs):
        calls.append(kwargs)
        return ModelReply(outputs[len(calls)-1],10,5,.25,'stop','revision',str(len(calls)),7)
    tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:
        {'input_ids':[list(range(10))],'attention_mask':[[1]*10]} if mapping_output else list(range(10)))
    executor=TaskPipelineExecutor(SimpleNamespace(generate=generate),tokenizer,{'max_tokens':20})
    candidate=Candidate('initial',{'system_prompt':'preserve privacy'},
         {'model_revision':'revision','seed':0,'generation':{'max_tokens':20}},semantic_kind='opaque',deterministic=False)
    task=Task('one',{'prompt':'private question'},{'target_response':'gold','pii_units':'name||address'},
              {'dataset':'pupa'})
    scope=EvaluationScope('test',(task,),'pipeline','pupa')
    return executor.run(candidate,task,scope,cap),calls,executor.physical_records


@pytest.mark.parametrize('mapping_output', [False, True])
def test_pipeline_meter_includes_all_judges_and_gpu_allocation(mapping_output):
    result,calls,records=executor_case(['redacted','untrusted','answer','{"judgment":true}',
                                      '{"judgment":false}','{"num_pii_leaked":1}'],mapping_output=mapping_output)
    assert result.status=='complete' and result.outcome.score==.75
    assert result.actual_cost==90 and len(calls)==len(records)==6
    assert result.receipt['gpu_seconds']==6*.25*7
    assert [c['total_token_cap'] for c in calls]==[10000-i*15 for i in range(6)]


def test_incomplete_pipeline_keeps_paid_prefix_without_outcome():
    result,calls,_=executor_case(['redacted','untrusted'],cap=40)
    assert result.status=='interrupted' and result.outcome is None
    assert result.actual_cost==15 and len(calls)==1


def test_invalid_judge_output_never_becomes_zero_score():
    result,calls,_=executor_case(['redacted','untrusted','answer','{"judgment":"true"}'])
    assert result.status=='error' and result.outcome is None and result.actual_cost==60


def test_polyglot_disallows_test_edits_before_execution(tmp_path, monkeypatch):
    # A temporary source fixture keeps this access-control test independent of
    # downloaded benchmark contents. Invalid edits must fail before execution.
    exercise=tmp_path/'data/raw/polyglot/source/fixture/python/exercises/practice/pig-latin'
    exercise.mkdir(parents=True)
    monkeypatch.setattr('specrsi.benchmarks.coding.ROOT', tmp_path)
    (exercise/'pig_latin.py').write_text('def translate(text): return text\n')
    (exercise/'pig_latin_test.py').write_text('assert False\n')
    ref={'exercise_path':str(exercise),'files':{'solution':['pig_latin.py']}}
    assert polyglot_score({'pig_latin_test.py':'assert True'},ref).score==0
    assert polyglot_score({'../../escape.py':'x=1'},ref).score==0


def test_gpu_count_is_validated():
    with pytest.raises(ValueError,match='GPU'):
        ModelReply('x',1,1,.1,'stop','r','id',0)
