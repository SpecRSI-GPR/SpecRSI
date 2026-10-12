"""Nested experiments count trained replicas, and donors require real frozen evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('specrsi_nested_launcher',ROOT/'scripts/run_ablations.py')
nested=importlib.util.module_from_spec(spec);spec.loader.exec_module(nested)


@pytest.fixture
def config(monkeypatch,tmp_path):
    monkeypatch.setattr(nested.studies,'source_fingerprint',lambda root=ROOT:{'sha256':'frozen-test-source','files':{}})
    c=nested.studies.read(ROOT/'configs/experiments/prospective_ablations.json')
    c['output_root']=str(tmp_path/'results');c['plan_root']=str(tmp_path/'plan')
    c['paired_states']['inputs']=[str(tmp_path/'observed_*'/'search.json')]
    c['paired_states']['output']=str(tmp_path/'pairs')
    return c


def test_nested480_has40_independent_training_replicas_and12_searches(config):
    plan=nested.build_nested(config)
    assert len(plan['jobs'])==480
    commands={tuple(j['required_artifacts'][0]['preparation_command']) for j in plan['jobs']}
    assert len(commands)==40
    seeds=set();search_seeds={}
    for method in config['methods']:
        for replica in range(8):
            jobs=[j for j in plan['jobs'] if j['method']==method and j['training_replica']==replica]
            assert len(jobs)==12 and len({j['config']['role_checkpoint'] for j in jobs})==1
            seed=jobs[0]['config']['training_replica'];assert seed not in seeds;seeds.add(seed)
            assert all(j['reported_n']==8 and j['config']['study_provenance']['nested_searches_are_not_independent_replicas'] for j in jobs)
            search_seeds.setdefault(replica,[]).append([j['seed'] for j in jobs])
    assert all(all(s==rows[0] for s in rows) for rows in search_seeds.values())
    sft=next(j for j in plan['jobs'] if j['method']=='sft')
    assert '/warmup/checkpoint' in sft['config']['role_checkpoint']
    assert sft['required_artifacts'][0]['preparation_command'][-1]=='--execute'


def test_inference_controls_share_gvo_but_retrained_controls_have_new_weights(config):
    plan=nested.build_nested(config,methods=['gvo','outer_blind','trained_outer_blind','quality_rule','no_exploration'],replicas=[0],searches=[0])
    jobs={j['method']:j for j in plan['jobs']}
    ckpt=lambda method:jobs[method]['config']['role_checkpoint']
    assert ckpt('gvo')==ckpt('outer_blind')==ckpt('quality_rule')==ckpt('no_exploration')
    assert ckpt('trained_outer_blind')!=ckpt('gvo')
    assert jobs['trained_outer_blind']['config']['training_method']=='outer_blind'
    assert jobs['quality_rule']['config']['controller']=='quality_rule'
    assert jobs['gvo']['config']['draft_exploration']==.1 and jobs['no_exploration']['config']['draft_exploration']==0
    with pytest.raises(ValueError,match='Duplicate'):nested.build_nested(config,replicas=[0,0])


def test_default_nested_only_materializes_and_does_not_execute(config,tmp_path,monkeypatch,capsys):
    def forbid(*a,**kw):raise AssertionError('Planning invoked training or experiment')
    monkeypatch.setattr(nested.studies,'execute_plan',forbid)
    path=tmp_path/'recipe.json';path.write_text(json.dumps(config))
    assert nested.main(['--config',str(path),'--methods','gvo','--replicas','0','--searches','0'])==0
    plan=json.loads((Path(config['plan_root'])/'plan.json').read_text())
    assert len(plan['jobs'])==1 and not Path(config['output_root']).exists()
    assert '"formal_jobs_started": false' in capsys.readouterr().out


def test_statistics_use_within_replica_means_then_sample_sd(config,tmp_path,monkeypatch):
    jobs=[]
    for replica,values in enumerate(([0.,2.],[10.,12.])):
        for index,value in enumerate(values):
            path=tmp_path/f'r{replica}s{index}';path.mkdir()
            (path/'search.json').write_text(json.dumps({'final_f':value,'total_cost':1,'cost_unit':'wall-ms'}))
            (path/'audit.json').write_text('[{"score": 0.5}]')
            jobs.append({'method':'gvo','training_replica':replica,'nested_search':index,'output':str(path)})
    monkeypatch.setattr(nested.studies,'valid_completion',lambda job:True)
    plan={'jobs':jobs,'declared_training_replicas':2,'declared_nested_searches_per_replica':2}
    summary=nested.summarize_nested(plan);endpoint=summary['methods'][0]
    assert endpoint['n_training_replicas']==2 and endpoint['mean_replica_return']==6
    assert endpoint['sample_sd_across_replicas']==pytest.approx(50**.5)
    plan['jobs']=jobs[:-1]
    assert nested.summarize_nested(plan)['methods'][0]['mean_replica_return'] is None


def observed_source(path,*,evidence=1,kind='outer',scope='science-reactor-selection',legal=True,changed=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    features=[.8,2.,evidence,.5,.2,1.,1.,3.]
    body={'scope':scope,'budget':{'available':80,'total':100},'tasks':[{'id':'t'}],
          'candidates':[{'id':'c','complete_score':.5,'missing':['t'] if evidence else []}],
          'recent_history':[{'status':'U' if evidence else 'P'}]}
    options=[[0.,.2,evidence,1.,0.,0.,0.,0.],[1.,.1,0.,0.,1.,0.,0.,0.]]
    records=[{'key':f'1:{kind}:a{i}','kind':kind,'mode':'full','features':features,
              'context_text':json.dumps(body),'options':[row]} for i,row in enumerate(options)]
    decisions=[{'kind':kind,'state_version':1,'support':['a0','a1'],'selected_id':'a0','values':[.5,.2]}] if legal else []
    path.write_text(json.dumps({'search_id':path.parent.name,'observation_records':records,'decisions':decisions,'source_changed_during_run':changed}))


def test_pairs_require_matching_observed_legal_supports_and_distinct_runs(config,tmp_path):
    observed_source(tmp_path/'observed_a/search.json',evidence=1)
    observed_source(tmp_path/'observed_b/search.json',evidence=0)
    observed_source(tmp_path/'observed_wrong_scope/search.json',evidence=0,scope='another-domain')
    observed_source(tmp_path/'observed_no_decision/search.json',legal=False)
    observed_source(tmp_path/'observed_changed/search.json',changed=True)
    prepared=nested.prepare_pairs(config)
    assert prepared['counts']=={'outer':1,'batch':0}
    assert prepared['missing_pairs']=={'outer':119,'batch':120}
    assert not prepared['complete_registered_cohorts']
    pair=prepared['pairs'][0]
    assert pair['receiver']['source_file']!=pair['donor']['source_file']
    assert pair['receiver']['support']==['a0','a1']
    assert pair['receiver']['cohort']==pair['donor']['cohort']
    assert len(prepared['rejected_sources'])==2
    with pytest.raises(ValueError,match='Insufficient observed pairs'):
        nested.execute_pairs(config,prepared)


def test_no_duplicate_states_or_identical_evidence_as_donors(config,tmp_path):
    for name in 'abc':observed_source(tmp_path/f'observed_{name}/search.json',evidence=1)
    assert not nested.prepare_pairs(config)['pairs']
    observed_source(tmp_path/'observed_d/search.json',evidence=0)
    prepared=nested.prepare_pairs(config)
    assert len(prepared['pairs'])==1
    ids=[x['id'] for pair in prepared['pairs'] for x in (pair['receiver'],pair['donor'])]
    assert len(ids)==len(set(ids))


def test_copied_search_is_not_an_independent_donor(config,tmp_path):
    a=tmp_path/'observed_a/search.json';observed_source(a,evidence=1)
    b=tmp_path/'observed_b/search.json';observed_source(b,evidence=0)
    copied=json.loads(b.read_text());copied['search_id']='observed_a';b.write_text(json.dumps(copied))
    prepared=nested.prepare_pairs(config)
    assert prepared['observed_eligible_states']==1 and not prepared['pairs']


def test_changed_frozen_state_rejected_before_model_load(config,tmp_path):
    a=tmp_path/'observed_a/search.json';observed_source(a,evidence=1)
    observed_source(tmp_path/'observed_b/search.json',evidence=0)
    prepared=nested.prepare_pairs(config)
    a.write_text('{}')
    with pytest.raises(ValueError,match='Frozen state source changed'):
        nested.execute_pairs(config,prepared,allow_partial=True)


def test_changed_receiver_checkpoint_rejected_before_cached_result(config,tmp_path):
    observed_source(tmp_path/'observed_a/search.json',evidence=1)
    observed_source(tmp_path/'observed_b/search.json',evidence=0)
    prepared=nested.prepare_pairs(config)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();weight=checkpoint/'heads.pt';weight.write_bytes(b'original')
    state=prepared['pairs'][0]['receiver'];state['role_checkpoint']=str(checkpoint)
    state['role_checkpoint_sha256']={'heads.pt':nested.studies.sha(weight)}
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):
        nested.execute_pairs(config,prepared,allow_partial=True)
