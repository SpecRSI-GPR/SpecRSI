import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import yaml
from specrsi.core.models import Candidate
from specrsi.experiments import carried_initial_artifacts

SPEC=importlib.util.spec_from_file_location('specrsi_transfer_runner',Path(__file__).parents[1]/'scripts/run_transfer.py')
transfer=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(transfer)


def manifest(instruction='initial',revision='old'):
    return Candidate('selected',{'system_prompt':instruction},{'model_revision':revision,'seed':9,'generation':{'max_tokens':7}},semantic_kind='opaque').manifest()


def test_carry_hash_and_artifact_only_boundary(tmp_path):
    path=tmp_path/'selected.json';path.write_text(json.dumps(manifest()))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts,receipt=carried_initial_artifacts({'initial_candidate_manifest':str(path),'initial_candidate_sha256':sha},{})
    assert artifacts=={'system_prompt':'initial'}
    assert receipt['carried_fields']==['artifacts'] and receipt['source_model_revision']=='old'
    path.write_text(json.dumps(manifest('tampered')))
    with pytest.raises(ValueError,match='hash'):carried_initial_artifacts({'initial_candidate_manifest':str(path),'initial_candidate_sha256':sha},{})


def test_carry_rejects_non_executable_artifact_and_incomplete_runtime(tmp_path):
    path=tmp_path/'selected.json';value=manifest();value['artifacts']={'unsupported':'code'}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Only a single'):carried_initial_artifacts({'initial_candidate_manifest':str(path)},{})
    value=manifest();value['runtime']={};path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='runtime'):carried_initial_artifacts({'initial_candidate_manifest':str(path)},{})


def test_paired_recursion_resets_or_carries_without_audit_or_ledger(tmp_path):
    base={'dataset':'game24','model':{'lock':'fresh'},'search':{'controller':'heuristic_bootstrap','initial_prompt':'start'}}
    stage=tmp_path/'stage.yaml';stage.write_text(yaml.safe_dump(base))
    config={'study':'control','output':str(tmp_path/'run'),'seeds':[4],'rounds':3,'stages':[{'config':str(stage)}]}
    observed=[]
    def runner(cfg):
        artifacts,receipt=carried_initial_artifacts(cfg,cfg['search'])
        observed.append((cfg['seed'],str(cfg['output']),artifacts['system_prompt'],receipt))
        # Fresh evaluation of the supplied artifact; deliberately distinct audit
        # sentinel must never appear in the selected artifact or next run config.
        selected=manifest(artifacts['system_prompt']+'|improved',revision='fresh')
        path=Path(cfg['output']);(path/'selected.json').write_text(json.dumps(selected))
        (path/'ledger.json').write_text('OLD_LEDGER_DO_NOT_TRANSFER')
        return {'search':{'initial_f':0.,'final_f':.5,'initial_acquisition_tokens':3,'total_cost':4},
                'audit':[{'status':'complete','score':.2,'secret':'AUDIT_DO_NOT_TRANSFER'}], 'physical_gpu_seconds':0.}
    result=transfer.execute(config,runner)
    assert len(observed)==6
    resets=[x for x in observed if '/reset/' in x[1]]
    retained=[x for x in observed if '/retain/' in x[1]]
    assert [x[2] for x in resets]==['start']*3
    assert [x[2] for x in retained]==['start','start|improved','start|improved|improved']
    assert [x[0] for x in resets]==[x[0] for x in retained]
    assert all('AUDIT' not in x[2] and 'LEDGER' not in x[2] for x in observed)
    assert result['improver_update_policy']=='fixed_improver_no_learning_updates'


def test_cross_family_schedule_and_explicit_checkpoint_identity(tmp_path):
    stages=[]
    for family in ('game24','hotpotqa'):
        path=tmp_path/(family+'.yaml')
        path.write_text(yaml.safe_dump({'dataset':family,'search':{'controller':'heuristic_bootstrap','initial_prompt':family}}))
        stages.append({'config':str(path)})
    config={'output':str(tmp_path/'cross'),'seeds':[0],'stages':stages,'rounds':2}
    planned=transfer.execute(config,plan=True)['runs']
    assert [r['config']['dataset'] for r in planned]==['game24','game24','hotpotqa','hotpotqa']
    assert planned[-1]['carry_source']=='previous_selected_artifact'
    config['checkpoint_sequence']=['only-one']
    with pytest.raises(ValueError,match='every round'):transfer.validate(config)
