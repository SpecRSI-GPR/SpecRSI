import importlib
import hashlib
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from specrsi.benchmarks.base import MissingArtifactError, ScoreResult
from specrsi.benchmarks.livebench import LOCAL_FALLBACK_METRIC, livebench_math_score
from specrsi.core.models import Candidate, EvaluationScope, Task
from specrsi.models import ModelReply
from specrsi.task_pipelines import TaskPipelineExecutor


REFERENCE = {'subtask': 'amps_hard_gcd', 'ground_truth': '2'}


@pytest.fixture
def native_livebench_harness():
    if (not Path('data/manifests/livebench_harness.json').is_file()
            or not list(Path('data/raw/livebench_harness/source').glob('LiveBench-*'))):
        pytest.skip('Download the pinned LiveBench harness with scripts/prepare_data.py')


@pytest.fixture
def fixture_amps_scorer(monkeypatch):
    """The pipeline tests exercise billing and judge validation with fake scoring.

    The separate native integration test above checks the pinned symbolic
    implementation. No production manifest/hash verification is changed.
    """
    def score(prediction, reference, *, equivalence_judge, judge_identity):
        assert prediction == r'\boxed{3}'
        decision = equivalence_judge(reference['ground_truth'], '3')
        return ScoreResult(float(decision), {'metric_variant': LOCAL_FALLBACK_METRIC,
            'local_fallback_calls': 1, 'judge_identity': judge_identity})

    monkeypatch.setattr('specrsi.benchmarks.livebench.livebench_math_score', score)


@pytest.mark.skipif(
    sys.platform != 'linux' or multiprocessing.get_context().get_start_method() != 'fork',
    reason='The pinned LiveBench symbolic timeout helper requires Linux fork multiprocessing',
)
def test_real_native_symbolic_path_and_required_fallback_are_isolated(native_livebench_harness):
    calls=[]
    environment=hashlib.sha256(repr(sorted(os.environ.items())).encode()).hexdigest()
    exact=livebench_math_score(r'\boxed{2}',REFERENCE,equivalence_judge=lambda a,b:calls.append((a,b)),
                              judge_identity='test-judge')
    assert exact.score == 1 and not calls
    module=importlib.import_module('livebench.process_results.math.AMPS_Hard.utils')
    official_callback=module.is_equiv_llm
    def judge(a,b):
        calls.append((a,b))
        return False
    result=livebench_math_score(r'\boxed{3}',REFERENCE,equivalence_judge=judge,judge_identity='test-judge')
    assert result.score == 0 and calls == [('2','3')]
    assert result.metrics['local_fallback_calls'] == 1
    assert result.metrics['metric_variant'] == LOCAL_FALLBACK_METRIC
    assert module.is_equiv_llm is official_callback
    assert hashlib.sha256(repr(sorted(os.environ.items())).encode()).hexdigest() == environment
    with pytest.raises(MissingArtifactError,match='registered judge'):
        livebench_math_score(r'\boxed{3}',REFERENCE)


def pipeline(judge_text,cap=10000):
    calls=[]
    outputs=[r'\boxed{3}',judge_text]
    def generate(messages,**kwargs):
        calls.append((messages,kwargs))
        return ModelReply(outputs[len(calls)-1],10,5,.25,'stop','revision',str(len(calls)))
    tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:list(range(10)))
    executor=TaskPipelineExecutor(SimpleNamespace(generate=generate),tokenizer,{'max_tokens':20})
    candidate=Candidate('agent',{'system_prompt':'Solve the problem.'},
        {'model_revision':'revision','seed':0,'generation':{'max_tokens':20}},semantic_kind='opaque',deterministic=False)
    task=Task('fixture',{'prompt':'Compute 1+1.'},REFERENCE,{'dataset':'livebench_math'})
    scope=EvaluationScope('fixture',(task,),'local-test','prospective-AMPS')
    return executor.run(candidate,task,scope,cap),calls,executor.physical_records


def test_local_judge_tokens_are_charged_and_gold_is_only_in_judge_input(fixture_amps_scorer):
    result,calls,records=pipeline('{"equivalent_or_negation":false}')
    assert result.status == 'complete' and result.outcome.score == 0
    assert result.actual_cost == 30 and len(calls) == 2
    assert [r['stage'] for r in records] == ['answer','amps_equivalence_judge']
    assert [kw['total_token_cap'] for _,kw in calls] == [10000,9985]
    assert 'expression_A' not in str(calls[0][0]) and 'expression_A' in str(calls[1][0])
    assert result.outcome.fields['metrics']['metric_variant'] == LOCAL_FALLBACK_METRIC


def test_malformed_judge_and_unaffordable_fallback_do_not_create_outcomes(fixture_amps_scorer):
    malformed,calls,_=pipeline('{"equivalent_or_negation":"false"}')
    assert malformed.status == 'error' and malformed.outcome is None
    assert malformed.actual_cost == 30 and len(calls) == 2
    interrupted,calls,_=pipeline('',cap=100)
    assert interrupted.status == 'interrupted' and interrupted.outcome is None
    assert interrupted.actual_cost == 15 and len(calls) == 1
