#!/usr/bin/env python3
"""Paired reset/retain recursion and cross-family transfer with fresh SpecRSI ledgers."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml
from specrsi.experiments import run_local_experiment


def merge(base,overrides):
    out=copy.deepcopy(base)
    for key,value in overrides.items():
        out[key]=merge(out[key],value) if isinstance(value,dict) and isinstance(out.get(key),dict) else copy.deepcopy(value)
    return out


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_hash(path):
    path=Path(path)
    if not (path/'model_config.json').exists():raise ValueError(f'Missing actual role checkpoint: {path}')
    entries={str(p.relative_to(path)):file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}
    return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()


def validate(config):
    rounds=int(config.get('rounds',len(config.get('stages',[]))))
    if rounds<1 or len(config.get('stages',[])) not in {1,rounds}:
        raise ValueError('Supply one repeated stage or exactly one explicit stage per round')
    modes=config.get('modes',['reset','retain'])
    if not modes or len(set(modes))!=len(modes) or set(modes)-{'reset','retain'}:
        raise ValueError('Modes must be distinct reset and/or retain')
    if not config.get('seeds') or any(type(seed) is not int for seed in config['seeds']):
        raise ValueError('Supply explicit integer seeds')
    sequence=config.get('checkpoint_sequence')
    if sequence is not None and len(sequence)!=rounds:
        raise ValueError('Checkpoint sequence must identify every round')
    jobs=config.get('training_schedule',[])
    if sequence is not None and jobs:raise ValueError('Choose supplied checkpoints or an executed training schedule, not both')
    indices=[job['before_round'] for job in jobs]
    if len(set(indices))!=len(indices) or any(type(i) is not int or i<0 or i>=rounds for i in indices):
        raise ValueError('At most one training job per explicit zero-based round')
    return rounds


def stage_config(config,round_index,seed,mode,output,previous=None,checkpoint=None):
    stage=config['stages'][0 if len(config['stages'])==1 else round_index]
    base=yaml.safe_load(Path(stage['config']).read_text())
    run=merge(base,stage.get('overrides',{}))
    # Same per-round model/controller RNG for both paired arms.
    run['seed']=(seed+round_index*1000003)%(2**32)
    run['output']=str(output)
    run['purpose']='prospective_'+config.get('study','recursion_reset_retain')+'_round_'+str(round_index)
    if mode=='retain' and previous is not None:
        run['initial_candidate_manifest']=str(Path(previous).resolve())
        run['initial_candidate_sha256']=file_hash(previous)
    if checkpoint is not None:
        run['search']['controller']='learned'
        run['search']['role_checkpoint']=str(Path(checkpoint).resolve())
    return run


def execute(config,runner=run_local_experiment,*,plan=False):
    rounds=validate(config);root=Path(config['output'])
    if not plan:root.mkdir(parents=True,exist_ok=True)
    records=[];training=[];plans=[]
    jobs={job['before_round']:job for job in config.get('training_schedule',[])}
    for seed in config['seeds']:
        previous={mode:None for mode in config.get('modes',['reset','retain'])}
        trained_checkpoint=config.get('initial_role_checkpoint')
        for round_index in range(rounds):
            if round_index in jobs and not plan:
                job=jobs[round_index];train_output=root/f'seed_{seed}'/f'training_round_{round_index}'
                train_output.mkdir(parents=True,exist_ok=True)
                training_config=merge(yaml.safe_load(Path(job['config']).read_text()),job.get('overrides',{}))
                training_config.setdefault('training',{})['seed']=(seed+round_index*1000003)%(2**32)
                used_config=train_output/'supplied_training_config.yaml'
                used_config.write_text(yaml.safe_dump(training_config,sort_keys=False))
                started=time.monotonic()
                command=[sys.executable,str(ROOT/'scripts/train.py'),'--config',str(used_config),'--output',str(train_output)]
                if job.get('resume_from_previous',False):
                    if trained_checkpoint is None:raise ValueError('Training resume requested without a preceding role checkpoint')
                    command.extend(['--resume',trained_checkpoint])
                with (train_output/'runner.log').open('w') as log:
                    completed=subprocess.run(command,stdout=log,stderr=log,cwd=ROOT)
                record={'seed':seed,'before_round':round_index,'config':job['config'],'config_sha256':file_hash(job['config']),
                        'used_config':str(used_config),'used_config_sha256':file_hash(used_config),
                        'command':command,'exit_code':completed.returncode,'wall_seconds':time.monotonic()-started}
                training.append(record)
                (root/'training.json').write_text(json.dumps(training,indent=2))
                if completed.returncode:raise RuntimeError(f'Trajectory training failed; see {train_output}')
                trained_checkpoint=str(train_output/'checkpoint')
                record['checkpoint_sha256']=checkpoint_hash(trained_checkpoint)
                (root/'training.json').write_text(json.dumps(training,indent=2))
            checkpoint=(config['checkpoint_sequence'][round_index] if config.get('checkpoint_sequence') is not None else trained_checkpoint)
            checkpoint_identity=checkpoint_hash(checkpoint) if checkpoint and not plan else None
            for mode in config.get('modes',['reset','retain']):
                output=root/f'seed_{seed}'/mode/f'round_{round_index}'
                run=stage_config(config,round_index,seed,mode,output,previous[mode],checkpoint)
                if plan:
                    plans.append({'seed':seed,'round':round_index,'mode':mode,'config':run,
                                  'carry_source':'previous_selected_artifact' if mode=='retain' and round_index else 'stage_initial_artifact'})
                    continue
                output.mkdir(parents=True,exist_ok=True)
                (output/'registered_config.json').write_text(json.dumps(run,indent=2))
                try:
                    result=runner(run)
                    selected=output/'selected.json'
                    if not selected.exists():raise RuntimeError('Experiment did not save its selected immutable artifact')
                    # Only the selected artifact is passed to a following round.
                    # Audits and all previous ledger records stay in their run.
                    previous[mode]=str(selected)
                    audit=result.get('audit',[])
                    complete_audit=[row['score'] for row in audit if row.get('status')=='complete']
                    record={'seed':seed,'round':round_index,'mode':mode,'dataset':run['dataset'],
                        'execution_seed':run['seed'],'output':str(output),'selected_sha256':file_hash(selected),
                        'carried_manifest_sha256':run.get('initial_candidate_sha256'),
                        'initial_f':result['search']['initial_f'],'final_f':result['search']['final_f'],
                        'initial_acquisition_tokens':result['search']['initial_acquisition_tokens'],
                        'search_tokens':result['search']['total_cost'],
                        'physical_gpu_seconds':result.get('physical_gpu_seconds'),
                        'audit_requested':len(audit),'audit_completed':len(complete_audit),
                        'audit_mean':sum(complete_audit)/len(complete_audit) if audit and len(complete_audit)==len(audit) else None,
                        'checkpoint':checkpoint,'checkpoint_sha256':checkpoint_identity,
                        'controller':run['search']['controller'],
                        'source_changed_during_run':result['search'].get('source_changed_during_run',False)}
                    records.append(record)
                    (root/'rounds.json').write_text(json.dumps(records,indent=2))
                except Exception as exc:
                    (output/'failure.json').write_text(json.dumps({'completed':False,'error':str(exc),'type':type(exc).__name__},indent=2))
                    raise
    if plan:return {'study':config.get('study'),'runs':plans,'training_schedule':config.get('training_schedule',[])}
    paired=[]
    for seed in config['seeds']:
        for index in range(rounds):
            pair={r['mode']:r for r in records if r['seed']==seed and r['round']==index}
            if set(pair)=={'reset','retain'}:
                a,b=pair['reset'],pair['retain']
                paired.append({'seed':seed,'round':index,'dataset':a['dataset'],
                    'retained_minus_reset_selection':b['final_f']-a['final_f'],
                    'retained_minus_reset_audit':b['audit_mean']-a['audit_mean'] if a['audit_mean'] is not None and b['audit_mean'] is not None else None})
    summary={'study':config.get('study'),'implementation':'prospective_registered_transfer_controls_v1',
        'completed':True,'rounds':records,'paired_differences':paired,'training_jobs':training,
        'improver_update_policy':'executed_supplied_training_schedule' if training else
            ('supplied_checkpoint_sequence_no_training_in_this_runner' if config.get('checkpoint_sequence') else 'fixed_improver_no_learning_updates'),
        'carried_state':'selected_candidate_artifacts_only','evidence_policy':'fresh_full_initial_evaluation_every_round_no_ledger_transfer',
        'audit_policy':'read_after_selection_never_passed_to_later_rounds'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--plan',action='store_true')
    a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text());result=execute(config,plan=a.plan)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
