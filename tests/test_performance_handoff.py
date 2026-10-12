"""Artifact-only performance handoff and native AIME alias compatibility."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from specrsi.answer_strategies import AnswerStrategy
from specrsi.benchmarks import BenchmarkExample
from specrsi.core.models import Candidate, Outcome, thaw
from specrsi.core.execution import ExecutionResult


def module(name,filename):
 spec=importlib.util.spec_from_file_location(name,ROOT/filename)
 value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
 return value

from specrsi import experiments
from specrsi import performance_candidates as candidates
runner=module('handoff_proposed_runner','scripts/run_performance_campaign.py')
comparison=module('comparison_handoff_proposal','scripts/run_registered_comparison.py')


def test_prompt_extension_carry_preserves_only_artifacts_and_requires_exact_hash(tmp_path):
 recipe=AnswerStrategy(version='reasoning_extract_v1',mode='reasoning_extract',answer_format='boxed_integer').to_dict()
 source=Candidate('selected',{'system_prompt':'Solve.','answer_strategy':recipe},
  {'model_revision':'old-model','seed':7,'generation':{'max_tokens':8}},semantic_kind='prompt_inference_extension')
 path=tmp_path/'selected.json';path.write_text(json.dumps(source.manifest()))
 sha=hashlib.sha256(path.read_bytes()).hexdigest()
 artifacts,receipt=experiments.carried_initial_artifacts({'initial_candidate_manifest':str(path),'initial_candidate_sha256':sha},{})
 assert artifacts==thaw(source.artifacts)
 assert receipt['carried_fields']==['artifacts']
 assert receipt['source_candidate_id']==source.uid
 assert receipt['ledger_policy']=='fresh_full_initial_evaluation_no_previous_outcomes_imported'
 target=Candidate('new-root',artifacts,{'model_revision':'new-model','seed':13,'generation':{'max_tokens':16}},semantic_kind='opaque')
 assert target.uid!=source.uid and target.runtime['model_revision']=='new-model'
 with pytest.raises(ValueError,match='hash'):
  experiments.carried_initial_artifacts({'initial_candidate_manifest':str(path),'initial_candidate_sha256':'wrong'},{})


def test_extension_kind_does_not_admit_workflow_or_unknown_artifact(tmp_path):
 for kind,artifacts in [('unknown',{'system_prompt':'x'}),('prompt_inference_extension',{'workflow':{}}),('prompt_inference_extension',{'source_code':'x'})]:
  c=Candidate('x',artifacts,{'model_revision':'r','seed':1,'generation':{}},semantic_kind=kind)
  path=tmp_path/'manifest.json';path.write_text(json.dumps(c.manifest()))
  with pytest.raises(ValueError):experiments.carried_initial_artifacts({'initial_candidate_manifest':str(path)},{})


def test_aime_alias_builds_same_registered_math_recipes_and_integer_metric():
 runtime={'model_revision':'r','seed':1,'generation':{'max_tokens':128}}
 pool=candidates.build_performance_candidates([],runtime=runtime,dataset='aime_2025_2026')
 assert len(pool)==4 and {x.recipe_id for x in pool}=={'direct','careful_derivation','verify','independent_alternative'}
 assert all(x.provenance['supervision']=='none' for x in pool)
 assert all('0 to 999' in x.candidate.artifacts['system_prompt'] for x in pool)
 assert experiments.score_answer('aime_2025_2026',r'\boxed{42}','42').score==1.
 assert experiments.score_answer('aime_2025_2026',r'\boxed{41}','42').score==0.


def test_comparison_alias_passes_dataset_guard_and_reaches_checkpoint_validation(tmp_path,monkeypatch):
 lock=tmp_path/'lock.json';lock.write_text('{}')
 cfg={'dataset':'aime_2025_2026','methods':['specrsi','initial'],'training_replicas':[{'seed':0,'checkpoint':'fake'}],
  'search_seeds':[1],'engineering_smoke_only':True,'budget':{k:10 for k in ('search_tokens','search_calls','execution_cap','draft_cap')},
  'selection_limit':1,'audit_limit':1,'resource_profile':'test','initial_prompt':'Solve.',
  'search':{'max_candidates':2,'max_actions':2},'model':{'lock':str(lock)}}
 def guard(*args):raise RuntimeError('reached frozen role checkpoint validation')
 monkeypatch.setattr(comparison,'require_role_checkpoint',guard)
 with pytest.raises(RuntimeError,match='reached frozen role'):
  comparison.register(cfg,tmp_path/'out')


class FakeExecutor:
 def __init__(self):self.physical_records=[];self.calls=[]
 def validate_prompt_candidate(self,*args):pass
 def run(self,candidate,task,scope,cap):
  self.calls.append((scope.split,task.name))
  self.physical_records.append({'request_id':scope.split+task.name,'prompt_tokens':1,'completion_tokens':1,'elapsed_seconds':.01,'gpu_count':0})
  return ExecutionResult('complete',2,Outcome(1.,{'answer':'42'}))


def test_selected_manifest_published_before_audit_hash_checked_and_ready_to_carry(tmp_path,monkeypatch):
 lock=tmp_path/'lock.json';lock.write_text(json.dumps({'revision':'r'}))
 config={'schema':'inference_performance_v1','output':str(tmp_path/'campaign'),'model_lock':str(lock),
  'dataset':'aime_2025_2026','sample_seed':1,'decode_seed':1,'execution_cap':10,'total_token_budget':100,
  'counts':{'development':1,'selection':1,'audit':1},'baseline':'base','finalists':1,
  'candidates':[{'name':'base','artifacts':{'system_prompt':'Solve.'},'generation':{'max_tokens':4}}]}
 examples={s:[BenchmarkExample(s,'aime_2025_2026',s,'Question '+s,'42')] for s in config['counts']}
 monkeypatch.setattr(runner,'source_hashes',lambda:{'frozen':'source'})
 executor=FakeExecutor();frozen=runner.run(config,executor=executor,loaded=examples,stop_after='selection')
 assert all(split!='audit' for split,_ in executor.calls)
 path=Path(config['output'])/'selected_candidate.json'
 assert Candidate(**json.loads(path.read_text())).uid==frozen['candidate_uid']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['selected_candidate_sha256']
 artifacts,receipt=experiments.carried_initial_artifacts({'initial_candidate_manifest':str(path),
  'initial_candidate_sha256':frozen['selected_candidate_sha256']},{})
 assert artifacts=={'system_prompt':'Solve.'} and receipt['source_candidate_id']==frozen['candidate_uid']
 path.write_text(path.read_text()+' ')
 executor=FakeExecutor()
 with pytest.raises(ValueError,match='bytes differ'):
  runner.run(config,executor=executor,loaded=examples)
 assert not executor.calls
