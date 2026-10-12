#!/usr/bin/env python3
"""Plan or launch concrete prospective S01-S13 experiments; planning is default."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml

ROLE_CONTROLLERS={'learned','quality_rule','completion_cost_rule'}
SPECRSI_INTERVENTIONS={
    'specrsi':{},'specrsi-heuristic':{},
    'specrsi-quality-rule':{'controller':'quality_rule'},
    'specrsi-completion-cost-rule':{'controller':'completion_cost_rule'},
    'specrsi-outer-blind':{'observation_mode':'outer_blind'},
    'specrsi-inner-blind':{'observation_mode':'inner_blind'},
    'specrsi-evidence-blind':{'observation_mode':'evidence_blind'},
    'specrsi-coarse-counts':{'observation_mode':'coarse_counts'},
    'specrsi-additive':{'batch_mode':'additive'},
    'specrsi-sham-relabel':{'observation_mode':'sham_relabel'},
    'specrsi-blind-additive':{'observation_mode':'outer_blind','batch_mode':'additive'},
    'specrsi-no-exploration':{'draft_exploration':0.},
    'specrsi-trained-outer-blind':{'observation_mode':'outer_blind'},
    'specrsi-trained-inner-blind':{'observation_mode':'inner_blind'},
    'specrsi-trained-evidence-blind':{'observation_mode':'evidence_blind'},
    'specrsi-trained-additive':{'batch_mode':'additive'},
    'specrsi-trained-blind-additive':{'observation_mode':'outer_blind','batch_mode':'additive'},
}
TRAINING_INTERVENTIONS={name:name.removeprefix('specrsi-trained-').replace('-','_') for name in SPECRSI_INTERVENTIONS if name.startswith('specrsi-trained-')}


def read(path):
    path=Path(path);text=path.read_text()
    return json.loads(text) if path.suffix=='.json' else yaml.load(text,Loader=getattr(yaml,'CSafeLoader',yaml.SafeLoader))


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def source_fingerprint(root=ROOT):
    """Source/config/locked dataset identities, without reading model tensor files."""
    paths=set((root/'src/specrsi').rglob('*.py'))|set((root/'src/specrsi').rglob('*.cc'))|set((root/'scripts').glob('*.py'))
    for folder in ('third_party/gepa/src','third_party/dspy/dspy','third_party/secondary_textgrad','third_party/secondary_trace'):
        paths.update((root/folder).rglob('*.py'))
    for folder in ('configs','data/manifests','data/reproduction'):
        paths.update(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in {'.json','.yaml','.yml','.txt'})
    for name in ('pyproject.toml','requirements.lock'):
        if (root/name).is_file():paths.add(root/name)
    hashes={str(p.relative_to(root)):sha(p) for p in sorted(paths)}
    return {'sha256':hashlib.sha256(canonical(hashes).encode()).hexdigest(),'files':hashes}


def sizes(dataset,profile,root):
    manifest=root/'data/manifests'/f'{dataset}.json'
    if not manifest.exists():
        manifest=root/'data/reproduction/manifests'/f'{dataset}.json'
    if not manifest.exists():return None
    counts=read(manifest).get('counts',{})
    if not all(counts.get(role,0)>0 for role in ('development','selection','audit')):return None
    return {role:min(profile[role],counts[role]) for role in ('development','selection','audit')}


def split_contracts(study,root):
    """Describe recorded split identities without loading benchmark contents."""
    contracts=[]
    for dataset in study.get('datasets',[study.get('dataset')]):
        if not dataset:
            continue
        if study['family']=='science':
            case_path=root/'configs/science/cases'/f'{dataset}.json'
            if case_path.is_file():
                cases=read(case_path)
                contracts.append({'dataset':dataset,'case_manifest':str(case_path.relative_to(root)),
                    'case_ids':list(range(cases['case_count'])),
                    'scenario_sets':['development','selection','audit']})
            continue
        if study['family'] in {'coding','audit'} and dataset!='polyglot':
            track_path=root/'configs/coding/prospective_swe_tracks.json'
            if track_path.is_file():
                tracks=read(track_path)
                selection='verified500' if study['id']=='S08' else 'selection60'
                audit='lite_audit_disjoint_verified500' if study['id']=='S08' else 'lite_audit_filtered'
                names={'development':'development23','selection':selection,'audit':audit}
                contracts.append({'dataset':dataset,'manifest':str(track_path.relative_to(root)),
                    'tracks':names,'counts':{role:len(tracks[name]) for role,name in names.items()}})
            continue
        manifest=root/'data/manifests'/f'{dataset}.json'
        if not manifest.is_file():
            manifest=root/'data/reproduction/manifests'/f'{dataset}.json'
        if manifest.is_file():
            counts=read(manifest).get('counts',{})
            contracts.append({'dataset':dataset,'manifest':str(manifest.relative_to(root)),
                'counts':{role:counts[role] for role in ('development','selection','audit') if role in counts}})
    return contracts


def workflow_of(config):
    if 'initial_workflow' in config['search']:return copy.deepcopy(config['search']['initial_workflow'])
    instruction=config['search']['initial_prompt']
    return {'nodes':[{'id':'solve','instruction':instruction,'dependencies':[]},
                     {'id':'review','instruction':'Check the prior answer and return the corrected final answer. '+instruction,'dependencies':['solve']}],
            'output_node':'review'}


def science_config(study,method,seed,profile,replica,training_revision):
    domain=study['dataset']
    from specrsi.science.registry import case_count
    ids=list(range(case_count(domain)))
    if method=='janus':
        c=read(ROOT/'configs/baselines/prospective_science_janus_reactor.yaml')
        c.update(science_domain=domain,science_case_ids=ids,science_proposer='local_model',science_proposal_space='global_and_case_routes',seed=seed,audit_seed=seed+1000000,
                 model={'base_url':'http://127.0.0.1:18085','name':'Qwen3.5-35B-A3B','lock':'configs/models/qwen35_35b_a3b_bf16.lock.json'})
        c['algorithm'].update(real_evaluations=20,islands=2,proposals_per_promotion=3)
        c['budget']={'max_tokens':profile['search_token_budget'],'max_real_items':case_count(domain)*30}
        return 'extended',c,[]
    if method=='native':return 'science-native',{'domain':domain,'case_ids':ids,'scenario_sets':['selection','audit'],'seed':seed},[]
    learned=method not in {'specrsi-heuristic','specrsi-all-u-heuristic'}
    training_root=f'artifacts/training/replicas/studies-{training_revision[:12]}/science_{domain}'
    training_method=TRAINING_INTERVENTIONS.get(method,'gvo')
    checkpoint=f'{training_root}/{training_method}/replica-{replica}/learned/checkpoint'
    return 'science-specrsi',{'domain':domain,'case_ids':ids,'seed':seed,'all_u':method in {'specrsi-all-u','specrsi-all-u-heuristic'},
        'budget_ms':profile['science_budget_ms'],'execution_cap_ms':profile['science_execution_cap_ms'],
        'checker_cap_ms':profile['science_checker_cap_ms'],'max_candidates':profile['science_candidates'],
        'max_actions':profile['science_max_actions'],'eta':0.,'controller':'learned' if learned else 'heuristic_bootstrap',
        'proposal':'trained_HF_parameter_drafter' if learned else 'seeded_routed_parameter_edits',
        'role_checkpoint':checkpoint if learned else None,'role_device':'cuda:1',
        'head_cap_ms':5000,'draft_cap_ms':30000,'draft_max_tokens':384,'draft_exploration':.1,
        'training_replica':replica,'training_output_root':training_root,
        'training_config':f'configs/training/prospective/science_{domain}.yaml','training_method':training_method,'author_checkpoint_used':False,
        **SPECRSI_INTERVENTIONS.get(method,{})},[
            'Independent development-trained role checkpoint shared by the registered search interventions.' if learned else
            'Explicit validation control: heuristic heads and seeded parameter proposals; no learned performance claim.',
            'Scientific outcomes retain native utility and use 0.5+atan(utility)/pi for the bounded search score; expenses are measured in wall milliseconds.']


def required_artifacts(config,kind,root=ROOT):
    """Bind actual checkpoint contents when present; never bless a missing model."""
    settings=config if kind=='science-specrsi' else config.get('search',{})
    if kind not in {'science-specrsi','specrsi'} or settings.get('controller') not in ROLE_CONTROLLERS:return []
    checkpoint=Path(settings['role_checkpoint']);absolute=root/checkpoint
    files={str(p.relative_to(absolute)):sha(p) for p in sorted(absolute.rglob('*')) if p.is_file()}
    necessary={'model_config.json','heads.pt','adapters/draft/adapter_model.safetensors','adapters/verify/adapter_model.safetensors'}
    ready=necessary<=set(files)
    if config.get('checkpoint_provenance')=='explicit_validated_pilot':
        try:
            metrics=read(absolute.parent/'metrics.json')
            ready=ready and read(absolute/'model_config.json').get('backend')=='hf' and bool(metrics['history']) and metrics['history'][-1].get('steps',0)>0
            files['../metrics.json']=sha(absolute.parent/'metrics.json')
        except (OSError,ValueError,TypeError,KeyError):ready=False
        return [{'kind':'explicit_validation_only_checkpoint','path':str(checkpoint),'status':'present' if ready else 'requires_preparation',
                 'sha256':files,'formal_replica':False}]
    # Formal replicas come from the actual training launcher, including an optimizer update.
    try:
        marker=read(absolute.parent/'completed.json');metrics=read(absolute.parent/'metrics.json')
        ready=ready and read(absolute/'model_config.json').get('backend')=='hf'
        ready=ready and bool(marker['outputs']) and bool(metrics['history']) and metrics['history'][-1].get('steps',0)>0
        ready=ready and all((root/p).is_file() and sha(root/p)==h for p,h in marker['outputs'].items())
        ready=ready and all((root/('scripts/train.py' if p=='train.py' else p)).is_file() and
            sha(root/('scripts/train.py' if p=='train.py' else p))==h for p,h in marker['sources'].items())
        if ready:
            files['../completed.json']=sha(absolute.parent/'completed.json')
            files['../metrics.json']=sha(absolute.parent/'metrics.json')
    except (OSError,ValueError,TypeError,KeyError):ready=False
    command=[sys.executable,str(root/'scripts/run_training_matrix.py'),'--config',
        config['training_config'],'--methods',config.get('training_method','gvo'),'--seeds',str(config['training_replica']),
        '--output',config['training_output_root'],'--execute']
    return [{'kind':'trained_role_checkpoint','path':str(checkpoint),'status':'present' if ready else 'requires_preparation',
             'sha256':files,'preparation_command':command,'preparation_shell':shlex.join(command)}]


def configure_specrsi_roles(config,registry,study,seed,stem,method,root):
    """A formal method always has executable training; controls stay explicit."""
    if method=='specrsi-heuristic':
        config['search']['controller']='heuristic_bootstrap'
        config['search'].pop('role_checkpoint',None)
        return ['Explicit heuristic-head validation control, not trained SpecRSI performance.']
    recipe=f'configs/training/prospective/{stem}.yaml'
    if not (root/recipe).is_file():raise ValueError('Missing executable native development training recipe: '+recipe)
    replica=seed-registry['prospective_seed_base']-10000*int(study['id'][1:])
    training_root=f"artifacts/training/replicas/studies-{registry['_source_fingerprint'][:12]}/{stem}"
    training_method=TRAINING_INTERVENTIONS.get(method,'gvo')
    config.update(training_config=recipe,training_replica=replica,training_output_root=training_root,training_method=training_method,
        intervention_training='development_retrained' if method in TRAINING_INTERVENTIONS else 'same_base_GVO_checkpoint_inference_intervention')
    config['search'].update(controller='learned',role_device='cuda:1',draft_exploration=.1,
        role_checkpoint=f'{training_root}/{training_method}/replica-{replica}/learned/checkpoint')
    config['search'].update(SPECRSI_INTERVENTIONS[method])
    return ['Actual independently trained HF SpecRSI roles; opaque LLM execution uses U because no universal equality proof is registered.']


def native_optimizer_config(registry,study,dataset,method,seed,root,base,splits,selection_track=None,audit_track=None):
    p=registry['profile'];swe=dataset.startswith('swe_')
    if method in {'gepa','gepa-merge','mipro-no-demos'}:
        c={'method':method,'dataset':'swe_verified' if swe else dataset,'seed':seed,'sizes':splits,'model':base['model'],
            'seed_prompt':base['search']['initial_prompt'],'max_metric_calls':max(p['gepa_metric_calls'],3*splits['selection']),
            'num_candidates':p['mipro_candidates'],'num_trials':p['mipro_trials']}
        if swe:c['native_args']={'selection_track':selection_track,'audit_track':audit_track,
            'execution_cap':60000,'actor_max_steps':24,'test_timeout':1800}
        return 'prompt-baseline',c,['Official optimizer with real repository/tool-agent native grading; prospective adaptation and optimizer-specific cost caps.']
    c=read(root/'configs/baselines'/f'prospective_native_{method}_{"swe" if swe else "polyglot"}.yaml')
    c.update(seed=seed,audit_seed=seed+1000000,initial_prompt=base['search']['initial_prompt'])
    c['data']={'root':'data',**splits}
    c['model']={**c.get('model',{}),**base['model'],'execution_cap':60000 if swe else 32768}
    if swe:c.update(selection_track=selection_track,audit_track=audit_track)
    c['budget']={'max_tokens':p.get('coding_search_token_budget',p['search_token_budget']) if swe else p['search_token_budget'],
                 'max_real_items':max(5000,splits['selection']*(p['max_candidates']+1))}
    c['audit_token_cap']=splits['audit']*c['model']['execution_cap']
    return 'extended',c,['Executable native instruction-adapter optimizer; candidates change agent instructions and native repository grading remains fixed.']


def task_config(registry,study,dataset,method,seed,root):
    p=registry['profile'];family=study['family'];notes=[]
    if family=='science':return science_config(study,method,seed,p,seed-registry['prospective_seed_base']-10000*int(study['id'][1:]),registry['_source_fingerprint'])
    if family=='mechanisms':
        name='fixed_pool_50pairs_50candidates' if method=='fixed-pool' else 'online_30pairs'
        c=read(root/'configs/mechanisms'/f'{name}.json');c['seeds']=[seed]
        return 'mechanisms',c,['Native Reactor mechanism control using the registered candidate pool and paired seeds.']
    if study['id']=='S09' and method!='repository-agent':
        return 'audit-artifact',{'source_study':'S06','source_method':'specrsi' if method=='specrsi-audit' else method,
            'source_task':'swe_verified_60','seed':seed},[
            'S09 reuses the frozen S06 method policy and its disjoint Lite audit; this is a shared audit receipt, not a second independent run.']
    if family in {'coding','audit'} and dataset!='polyglot':
        track={'S06':'selection60','S08':'verified500','S09':'lite_audit_filtered'}[study['id']]
        if method=='repository-agent':
            c=read(root/registry['swe_templates'][track]);c['seed']=seed
            c['run_id']=f"study-{study['id']}-{registry['_source_fingerprint'][:12]}"
            return 'swe-agent',c,['Prospective fixed repository-agent protocol, not an optimizer baseline.']
        c=read(root/registry['swe_templates']['specrsi']);c.update(seed=seed,selection_track=track)
        c.pop('selection_ids',None);c.pop('audit_ids',None)
        c['search'].update(max_candidates=p['max_candidates'],max_actions=p['max_actions'],budget_tokens=p.get('coding_search_token_budget',p['search_token_budget']))
        if track=='verified500':
            c['audit_track']='lite_audit_disjoint_verified500'
            notes.append('Verified500 overlaps selection60/calibration; its separate audit excludes the whole Verified500 set.')
        if method not in SPECRSI_INTERVENTIONS:
            tracks=read(root/'configs/coding/prospective_swe_tracks.json')
            audit_track=c.get('audit_track','lite_audit_filtered')
            splits={'development':len(tracks['development23']),'selection':len(tracks[track]),'audit':len(tracks[audit_track])}
            return native_optimizer_config(registry,study,'swe_verified',method,seed,root,c,splits,track,audit_track)
        notes.extend(configure_specrsi_roles(c,registry,study,seed,'swe',method,root))
        return 'specrsi',c,notes
    workflow=family=='workflows'
    key=registry['workflow_templates']['kimi' if study['id']=='S11' else 'qwen'][dataset] if workflow else registry['prompt_templates'][dataset]
    c=read(root/key);n=sizes(dataset,p,root)
    if n is None:raise ValueError('Missing nonempty registered development/selection/audit data manifest')
    c.update(seed=seed,selection_limit=n['selection'],audit_limit=n['audit'],purpose='prospective_registered_S01_S13_new_protocol')
    c['search'].update(budget_tokens=p['search_token_budget'],max_candidates=p['max_candidates'],max_actions=p['max_actions'])
    if workflow:
        w=workflow_of(c);c['search'].pop('initial_prompt',None);c['search']['initial_workflow']=w
    if method in SPECRSI_INTERVENTIONS:
        notes.extend(configure_specrsi_roles(c,registry,study,seed,Path(key).stem,method,root))
        return 'specrsi',c,notes
    if dataset=='polyglot' and method in {'hgm','aflow','janus','emas'}:
        return native_optimizer_config(registry,study,dataset,method,seed,root,c,n)
    if method in {'textgrad','trace-optoprime','mipro-demos'}:
        key={'textgrad':'textgrad','trace-optoprime':'trace_optoprime','mipro-demos':'mipro_demos'}[method]
        secondary=read(root/'configs/baselines'/f'secondary_{key}.yaml')
        secondary.update(dataset=dataset,seed=seed,seed_prompt=c['search']['initial_prompt'],sizes=n)
        secondary['data']={'root':c.get('data_root','data'),**n}
        secondary['model']={**secondary['model'],**c['model']}
        secondary['budget']={'search_tokens':p['search_token_budget'],
            'audit_tokens':n['audit']*secondary['model'].get('execution_cap',131072)}
        return 'secondary-baseline',secondary,['Pinned original secondary optimizer with local native scoring; demos are confined to registered development examples.']
    if method in {'gepa','gepa-merge','mipro-no-demos'}:
        return 'prompt-baseline',{'method':method,'dataset':dataset,'seed':seed,'sizes':n,'model':c['model'],
            'seed_prompt':c['search']['initial_prompt'],'max_metric_calls':p['gepa_metric_calls'],
            'num_candidates':p['mipro_candidates'],'num_trials':p['mipro_trials']},[
            'Pinned original optimizer with local native-task adapter; budget units are optimizer-specific, not calibrated paper B.']
    ext=read(root/'configs/baselines'/f'prospective_{method}.yaml')
    ext.update(dataset=dataset,seed=seed,audit_seed=seed+1000000,initial_workflow=c['search']['initial_workflow'])
    ext['data']={'root':c.get('data_root','data'),**n}
    ext['model']={**ext['model'],**c['model'],'execution_cap':c['search'].get('execution_cap_tokens',32768)}
    ext['budget']={'max_tokens':p['search_token_budget'],'max_real_items':5000}
    ext['audit_token_cap']=max(100000,n['audit']*ext['model']['execution_cap'])
    return 'extended',ext,['Executable workflow adaptation with the registered program space, native scorer, and enforced budgets.']


def build_plan(registry_path=None,*,studies=None,methods=None,replicates=None,output_root='results/studies',plan_root='artifacts/study_plan',root=ROOT,_fingerprint=None):
    registry_path=Path(registry_path or root/'configs/experiments/prospective_studies.json')
    registry=read(registry_path);paper=read(root/registry['paper_matrix'])
    known={s['id'] for s in paper['studies']}
    if studies and not set(studies)<=known:raise ValueError('Unknown study IDs: '+str(set(studies)-known))
    fingerprint=_fingerprint or source_fingerprint(root)
    registry['_source_fingerprint']=fingerprint['sha256']
    jobs=[];unavailable=[];study_records=[]
    for study in paper['studies']:
        sid=study['id']
        if studies and sid not in studies:continue
        group='polyglot' if study.get('dataset')=='polyglot' else study['family']
        defaults=registry['method_families'][group]
        supported=defaults+registry.get('optional_method_families',{}).get(group,[])
        requested=methods or defaults
        study_records.append({'paper':study,'supported_methods':supported,'default_methods':defaults,
                              'split_contracts':split_contracts(study,root)})
        for method in requested:
            if method not in supported:
                reasons=[x['reason'] for x in registry.get('explicit_unavailable_tracks',[]) if sid in x['studies'] and method in x['methods']]
                unavailable.append({'study':sid,'method':method,'status':'unavailable','reason':reasons[0] if reasons else 'No registered executable method adapter for this study/program space.'})
                continue
            count=(study['fixed_pool_pairs'] if method=='fixed-pool' else study['online_pairs']) if group=='mechanisms' else study['reported_n']
            indices=list(range(count)) if replicates is None else list(replicates)
            if any(type(i) is not int or i<0 or i>=count for i in indices):raise ValueError(f'{sid}/{method}: replicate outside reported count{count}')
            for dataset in study.get('datasets',[study.get('dataset','mechanisms')]):
                for i in indices:
                    seed_study=6 if sid=='S09' and method!='repository-agent' else int(sid[1:])
                    seed=registry['prospective_seed_base']+10000*seed_study+i
                    job_id=f'{sid}/{dataset}/{method}/pair_{i:03d}'
                    try:kind,c,notes=task_config(registry,study,dataset,method,seed,root)
                    except (OSError,ValueError,KeyError) as exc:
                        unavailable.append({'study':sid,'method':method,'task':dataset,'replicate':i,'status':'unavailable','reason':str(exc)});continue
                    pairing=f'{sid}/{method if group=="mechanisms" else dataset}/pair_{i:03d}'
                    c['study_provenance']={'study':sid,'reported_settings':study,'pair_id':pairing,'replicate':i,
                        'prospective_seed':seed,'original_seed_known':False,'author_exact':False,'paper_budget_comparable':False}
                    prerequisites=required_artifacts(c,kind,root)
                    dependencies=[]
                    if kind=='audit-artifact':
                        source_plan=build_plan(registry_path,studies=['S06'],methods=[c['source_method']],replicates=[i],
                            output_root=output_root,plan_root=plan_root,root=root,_fingerprint=fingerprint)
                        if not source_plan['jobs']:
                            unavailable.append({'study':sid,'method':method,'status':'unavailable','reason':'S06 dependency is unavailable'});continue
                        dependencies=source_plan['jobs']
                    identity=hashlib.sha256(canonical({'kind':kind,'config':c,'source':fingerprint['sha256'],
                        'required_artifacts':prerequisites,'dependencies':[d['fingerprint'] for d in dependencies]}).encode()).hexdigest()
                    folder=Path(output_root)/job_id/identity[:12]
                    c['output']=str(folder)
                    config_path=Path(plan_root)/'jobs'/job_id/(identity[:12]+'.json')
                    job={'id':job_id,'study':sid,'task':dataset,'method':method,'replicate':i,'seed':seed,'pair_id':pairing,
                         'reported_n':count,'status':'requires_preparation' if any(a['status']!='present' for a in prerequisites) or any(d['status']=='requires_preparation' for d in dependencies) else 'runnable_prospective',
                         'required_artifacts':prerequisites,'runner':kind,'config':c,'config_path':str(config_path),
                         'dependencies':dependencies,
                         'output':str(folder),'fingerprint':identity,'source_fingerprint':fingerprint['sha256'],'notes':notes,
                         'completion_contract':kind,'gpu_required':kind not in {'science-native','science-specrsi','mechanisms','audit-artifact'} or
                             (kind=='science-specrsi' and c['controller'] in ROLE_CONTROLLERS)}
                    job['command']=command_for(job,root)
                    job['shell_display']=shlex.join(job['command'])
                    jobs.append(job)
    # Explicit unavailable options remain visible even when not selected for launch.
    catalog=[x for x in registry.get('explicit_unavailable_tracks',[]) if not studies or set(x['studies'])&set(studies)]
    return {'schema_version':1,'author_exact':False,'default_action':'plan_only_no_model_calls','registry':str(registry_path),
            'request':{'studies':studies,'methods':methods,'replicates':replicates,'output_root':str(output_root),'plan_root':str(plan_root)},
            'study_records':study_records,'source_fingerprint':fingerprint,'jobs':jobs,'unavailable_requested_tracks':unavailable,
            'unavailable_catalog':catalog,'pairing_policy':registry['pairing_rule'],'job_count':len(jobs),
            'supplemental_protocols':registry.get('supplemental_protocols',[])}


def command_for(job,root=ROOT):
    python=sys.executable;kind=job['runner'];path=job['config_path'];c=job['config']
    if kind=='specrsi':return [python,str(root/'scripts/run_experiment.py'),'--config',path]
    if kind=='extended':return [python,str(root/'scripts/run_extended_baselines.py'),'--config',path]
    if kind=='swe-agent':return [python,str(root/'scripts/run_swe.py'),'--config',path]
    if kind=='mechanisms':return [python,str(root/'scripts/run_mechanisms.py'),'--config',path]
    if kind=='secondary-baseline':return [python,str(root/'scripts/run_secondary_baselines.py'),'--config',path]
    if kind=='prompt-baseline':
        command=[python,str(root/'scripts/run_baseline.py'),'--method',c['method'],'--dataset',c['dataset'],'--seed',str(c['seed']),
            '--development-size',str(c['sizes']['development']),'--selection-size',str(c['sizes']['selection']),
            '--audit-size',str(c['sizes']['audit']),'--max-metric-calls',str(c['max_metric_calls']),
            '--num-candidates',str(c['num_candidates']),'--num-trials',str(c['num_trials']),
            '--max-tokens',str(c['model']['max_tokens']),'--base-url',c['model']['base_url'],'--model-lock',c['model']['lock'],
            '--seed-prompt',c['seed_prompt'],'--output',c['output']]
        if c.get('native_args'):
            native=c['native_args']
            command.extend(['--selection-track',native['selection_track'],'--audit-track',native['audit_track'],
                '--native-execution-cap',str(native['execution_cap']),'--actor-max-steps',str(native['actor_max_steps']),
                '--test-timeout',str(native['test_timeout'])])
        return command
    return [python,str(root/'scripts/run_studies.py'),'--worker',kind,'--worker-config',path]


def completion_outputs(job):
    """Return exactly the evidence files whose complete semantics were checked."""
    folder=Path(job['output']);kind=job['runner'];c=job['config']
    def j(name):return json.loads((folder/name).read_text())
    files=[]
    if kind in {'specrsi','science-specrsi'}:
        search=j('search.json');selected=j('selected.json');audit=j('audit.json')
        if search.get('source_changed_during_run') or search.get('budget_breached'):raise ValueError('Runtime source/cost contract failed')
        a=search['acceptance']
        if a['selected_id'] not in a['complete_candidates'] or selected!=a['selected']:raise ValueError('Selected candidate is not frozen and complete')
        if not audit or any(x.get('status')!='complete' or x.get('score') is None for x in audit):raise ValueError('Native audit did not finish completely')
        expected=len(c['case_ids']) if kind=='science-specrsi' else c.get('audit_limit',len(audit))
        if len(audit)!=expected:raise ValueError('Wrong SpecRSI audit registry size')
        files=['search.json','selected.json','audit.json','physical_calls.jsonl']
    elif kind in {'prompt-baseline','secondary-baseline'}:
        summary=j('summary.json')
        if summary.get('status')!='complete':raise ValueError('Optimizer did not complete')
        rows=[json.loads(line) for line in (folder/'audit.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)!=c['sizes']['audit']:raise ValueError('Missing native audit examples')
        manifest=j('manifest.json')
        if [x['id'] for x in rows]!=manifest['splits']['audit']:raise ValueError('Wrong native audit example IDs')
        if any(type(x.get('score')) not in (int,float) or not math.isfinite(x['score']) for x in rows):raise ValueError('Missing finite native audit score')
        selected_text=(folder/'selected_candidate.json').read_text().rstrip('\n')
        if hashlib.sha256(selected_text.encode()).hexdigest()!=summary['selected_candidate_sha256']:raise ValueError('Selected baseline candidate changed')
        files=['summary.json','manifest.json','selected_candidate.json','audit.jsonl','model_calls.jsonl']
        if c.get('dataset') in {'hover','pupa','polyglot','swe_verified','swe_lite','livebench_math'}:files.append('native_receipts.jsonl')
        if kind=='secondary-baseline':files.extend(['resolved_config.json','physical_budget_calls.jsonl'])
    elif kind=='extended':
        result=j('result.json');selected=j('selected.json')
        if result['selected']!=selected or not result['audit']['items'] or not all(x['complete'] for x in result['audit']['items']):
            raise ValueError('Extended baseline has incomplete audit/selection artifact')
        expected=len(c.get('science_case_ids',[])) if c.get('backend_factory')=='specrsi.science.baselines:make_backend' else c['data']['audit']
        if len(result['audit']['items'])!=expected:raise ValueError('Wrong extended audit registry size')
        files=['result.json','selected.json','physical_calls.json','resolved_config.json']
        if c.get('backend_factory')=='specrsi.baselines.native_backend:make_backend':files.append('native_receipts.json')
    elif kind=='swe-agent':
        result=j('summary.json');expected=c['instance_ids']
        if not result.get('all_complete') or result['requested']!=len(expected) or result['completed']!=len(expected):raise ValueError('SWE native grading incomplete')
        if {x['instance_id'] for x in result['results']}!=set(expected):raise ValueError('Wrong SWE native task IDs')
        files=['summary.json','resolved_config.json']
        for iid in expected:files.extend(f'{iid}/{name}' for name in ('evaluation.json','prediction.json','patch.diff','image.json','agent.json'))
    elif kind=='science-native':
        result=j('summary.json');expected=len(c['case_ids'])*len(c['scenario_sets'])
        if not result['all_complete'] or len(result['results'])!=expected:raise ValueError('Incomplete native science suite')
        files=['summary.json']+[r['file'] for r in result['results']]
    elif kind=='mechanisms':
        summary=j('summary.json')
        if summary['pairs']!=len(c['seeds']):raise ValueError('Missing paired mechanism units')
        files=['summary.json','resolved_config.json']+[f'pair_{seed}.json' for seed in c['seeds']]
        for name in files[2:]:
            result=j(name)
            if not result['comparison']['common_complete_outcomes_equal']:raise ValueError('Mechanism native comparison disagrees')
            if any(v['accounting']['budget_breached'] for v in result['variants'].values()):raise ValueError('Mechanism cap contract breached')
    elif kind=='audit-artifact':
        result=j('audit_summary.json')
        if not result.get('all_complete') or not result.get('source_completed_fingerprint'):raise ValueError('Unproven frozen source audit')
        source=Path(result['source_output'])
        if sha(source/'completed.json')!=result['source_completed_sha256']:raise ValueError('Referenced frozen source completion changed')
        source_marker=read(source/'completed.json')
        if any(sha(source/name)!=h for name,h in source_marker['output_sha256'].items()):raise ValueError('Referenced frozen source outputs changed')
        files=['audit_summary.json','selected.json','audit.json']
    else:raise ValueError('Unknown completion contract')
    if not all((folder/name).is_file() for name in files):raise ValueError('Required output artifact missing')
    return files


def valid_completion(job):
    folder=Path(job['output'])
    try:
        marker=json.loads((folder/'completed.json').read_text())
        if marker.get('fingerprint')!=job['fingerprint'] or marker.get('source_fingerprint')!=job['source_fingerprint']:return False
        if marker.get('config_sha256')!=hashlib.sha256(canonical(job['config']).encode()).hexdigest():return False
        if required_artifacts(job['config'],job['runner'])!=job.get('required_artifacts',[]):return False
        files=completion_outputs(job)
        if set(marker.get('output_sha256',{}))!=set(files):return False
        return all(sha(folder/name)==marker['output_sha256'][name] for name in files)
    except (OSError,ValueError,KeyError,TypeError):return False


def materialize(plan):
    for job in plan['jobs']:
        path=Path(job['config_path']);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(job['config'],indent=2,ensure_ascii=False)+'\n')


def execute_plan(plan,*,root=ROOT,rebuild=None):
    if plan['unavailable_requested_tracks']:
        raise ValueError('Requested unavailable tracks: '+json.dumps(plan['unavailable_requested_tracks'],ensure_ascii=False))
    if source_fingerprint(root)['sha256']!=plan['source_fingerprint']['sha256']:raise RuntimeError('Source/config inputs changed since plan')
    all_jobs={j['id']:j for item in plan['jobs'] for j in [item,*item.get('dependencies',[])]}
    preparations={tuple(a['preparation_command']):a for j in all_jobs.values() for a in j.get('required_artifacts',[])
                  if a['status']!='present'}
    if preparations:
        preparation_root=Path(plan['request']['plan_root'])/'preparations';preparation_root.mkdir(parents=True,exist_ok=True)
        for command,artifact in preparations.items():
            name=hashlib.sha256(canonical(command).encode()).hexdigest()[:16]
            print(json.dumps({'stage':'training_prerequisite','path':artifact['path'],'command':list(command)}),flush=True)
            replica_directory=(root/artifact['path']).parents[1]
            if replica_directory.exists():
                replica_directory.rename(replica_directory.with_name(replica_directory.name+'.incomplete-'+uuid.uuid4().hex[:10]))
            with (preparation_root/(name+'.log')).open('w') as log:
                process=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError('Training prerequisite failed; see '+str(preparation_root/(name+'.log')))
            if source_fingerprint(root)['sha256']!=plan['source_fingerprint']['sha256']:raise RuntimeError('Sources changed during prerequisite training')
        # Bind the newly measured checkpoint bytes before freezing any formal job identity.
        plan=rebuild(plan) if rebuild else build_plan(plan['registry'],root=root,**plan['request'])
        if any(a['status']!='present' for item in plan['jobs'] for j in [item,*item.get('dependencies',[])] for a in j.get('required_artifacts',[])):
            raise RuntimeError('Training returned without complete, hash-verified checkpoint artifacts')
        (Path(plan['request']['plan_root'])/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    materialize(plan);results=[]
    by_id={j['id']:j for j in plan['jobs']}
    for job in plan['jobs']:
        if valid_completion(job):
            results.append({'id':job['id'],'status':'resumed_complete','output':job['output']});continue
        if source_fingerprint(root)['sha256']!=job['source_fingerprint']:raise RuntimeError('Source/config inputs changed since plan; regenerate the plan')
        if required_artifacts(job['config'],job['runner'],root)!=job.get('required_artifacts',[]):raise RuntimeError('Trained checkpoint inputs changed; regenerate the plan')
        c=copy.deepcopy(job['config'])
        if job['runner']=='audit-artifact':
            source_id=f"S06/swe_verified_60/{c['source_method']}/pair_{job['replicate']:03d}"
            source_job=by_id.get(source_id)
            if source_job is None or not valid_completion(source_job):
                dependency=build_plan(plan['registry'],studies=['S06'],methods=[c['source_method']],replicates=[job['replicate']],
                    output_root=plan['request']['output_root'],plan_root=plan['request']['plan_root'],root=root,
                    _fingerprint=plan['source_fingerprint'])
                source_job=dependency['jobs'][0]
                if not valid_completion(source_job):execute_plan(dependency,root=root)
            if not valid_completion(source_job):raise RuntimeError('Required frozen S06 SpecRSI run/audit is not complete: '+source_id)
            c['source_output']=source_job['output'];c['source_fingerprint']=source_job['fingerprint']
            c['source_runner']=source_job['runner']
            Path(job['config_path']).write_text(json.dumps(c,indent=2)+'\n')
        folder=Path(job['output'])
        if folder.exists() and any(folder.iterdir()):
            archived=folder.with_name(folder.name+'.incomplete-'+uuid.uuid4().hex[:10]);folder.rename(archived)
        # GEPA/MIPRO require an empty payload directory. Keep launcher evidence beside it.
        launch_folder=folder.with_name(folder.name+'.launcher')
        if launch_folder.exists():launch_folder.rename(launch_folder.with_name(launch_folder.name+'.previous-'+uuid.uuid4().hex[:10]))
        launch_folder.mkdir(parents=True,exist_ok=True)
        (launch_folder/'launch.json').write_text(json.dumps({'job_id':job['id'],'fingerprint':job['fingerprint'],'command':job['command'],
                                                   'started_unix':time.time()},indent=2)+'\n')
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYBAMM_DISABLE_TELEMETRY='true')
        if not job['gpu_required']:env['CUDA_VISIBLE_DEVICES']=''
        started=time.monotonic()
        print(json.dumps({'id':job['id'],'formal_job_started':True,'output':job['output']}),flush=True)
        with (launch_folder/'launcher.log').open('w') as log:
            process=subprocess.run(job['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        failure=None
        if process.returncode:failure=f'Process exited {process.returncode}'
        elif source_fingerprint(root)['sha256']!=job['source_fingerprint']:failure='Source/config inputs changed during execution'
        elif read(job['config_path'])!=c:failure='Resolved job configuration changed during execution'
        elif required_artifacts(job['config'],job['runner'],root)!=job.get('required_artifacts',[]):failure='Trained checkpoint changed during execution'
        else:
            try:files=completion_outputs(job)
            except (OSError,ValueError,KeyError,TypeError) as exc:failure=f'Incomplete output contract: {exc}'
        if failure:
            record={'id':job['id'],'status':'failed','reason':failure,'elapsed_seconds':time.monotonic()-started}
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'failed.json').write_text(json.dumps(record,indent=2)+'\n');results.append(record)
            continue
        marker={'fingerprint':job['fingerprint'],'source_fingerprint':job['source_fingerprint'],'job_id':job['id'],
                'config_sha256':hashlib.sha256(canonical(job['config']).encode()).hexdigest(),
                'output_sha256':{name:sha(folder/name) for name in files},'finished_unix':time.time(),
                'elapsed_seconds':time.monotonic()-started}
        temporary=folder/('.completed-'+uuid.uuid4().hex);temporary.write_text(json.dumps(marker,indent=2)+'\n');os.replace(temporary,folder/'completed.json')
        record={'id':job['id'],'status':'completed','output':str(folder),'elapsed_seconds':marker['elapsed_seconds']}
        results.append(record);print(json.dumps(record),flush=True)
    return results


def run_science_specrsi(c):
    initial_sources=source_fingerprint(ROOT)
    checkpoint_inputs=required_artifacts(c,'science-specrsi')
    if any(a['status']!='present' for a in checkpoint_inputs):raise ValueError('Missing independent trained science role checkpoint')
    from specrsi.core.acquisition import ResultAcquirer,AcquisitionCaps
    from specrsi.core.budget import BudgetLedger
    from specrsi.core.ledger import Ledger
    from specrsi.core.search import SearchEngine,SearchConfig,DraftResult
    from specrsi.core.models import thaw
    from specrsi.science.executor import make_candidate,make_scope,ScienceExecutor
    from specrsi.science.checker import ScienceParameterChecker
    from specrsi.mechanisms import child_edit
    scope=make_scope(c['domain'],case_ids=tuple(c['case_ids']));initial=make_candidate(c['domain'])
    checker=ScienceParameterChecker();ledger=Ledger(checker);executor=ScienceExecutor()
    caps=AcquisitionCaps(c['execution_cap_ms'],0,c['checker_cap_ms'],1)
    setup=BudgetLedger(max(10000000,c['execution_cap_ms']*len(scope.tasks)*2));acquirer=ResultAcquirer(ledger,setup,executor)
    started=time.monotonic()
    for task in scope.tasks:
        result=acquirer.acquire(initial,task,scope,(),caps)
        if result.record is None or not result.record.outcome.admissible:raise RuntimeError('Invalid/incomplete initial native science incumbent')
    class RoutedDrafter:
        def draft(self,parent,state,cap,rng):
            t=time.monotonic();child=child_edit(parent,scope,rng,len(state.candidates))
            cost=max(1,math.ceil((time.monotonic()-t)*1000))
            if cost>cap:return DraftResult(None,cost,status='interrupted',error='Measured parameter drafting overrun')
            return DraftResult(child,cost)
    bridge=None;rules=None;hooks={};drafter=RoutedDrafter()
    if c.get('controller','heuristic_bootstrap') in ROLE_CONTROLLERS:
        from specrsi.learning.hf import HFRoleModel
        from specrsi.learning.bridge import NeuralSearchBridge
        from specrsi.learning.science_bridge import HFParameterDrafter
        model=HFRoleModel.load(c['role_checkpoint'],device=c.get('role_device','cuda:1'));model.eval()
        bridge=NeuralSearchBridge(model,cost_unit='wall-ms',batch_mode=c.get('batch_mode','joint'),
            observation_mode=c.get('observation_mode','full'),observation_donors=c.get('observation_donors'))
        drafter=HFParameterDrafter(bridge,max_new_tokens=c.get('draft_max_tokens',384),
            check_cap_ms=c['checker_cap_ms'],exploration=c.get('draft_exploration',.1),
            snapshot=hashlib.sha256(canonical(checkpoint_inputs).encode()).hexdigest())
        hooks={'outer_value':bridge.outer_value,'batch_value':bridge.batch_value}
    elif c.get('controller','heuristic_bootstrap')!='heuristic_bootstrap':raise ValueError('Unregistered science controller')
    config=SearchConfig(execution_cap=c['execution_cap_ms'],draft_cap=c.get('draft_cap_ms',30000) if bridge else 1000,check_cap=c['checker_cap_ms'],
        max_sources=0 if c['all_u'] else 1,max_actions=c['max_actions'],max_candidates=c['max_candidates'],max_children=2,
        seed=c['seed'],margin=c['eta'],all_u=c['all_u'],cost_unit='measured-native-and-role-wall-ms' if bridge else 'native/checker wall-ms + explicitly registered host-controller units',
        planning_cost=0 if bridge else 1,outer_value_cap=c.get('head_cap_ms',5000) if bridge else 1,
        batch_value_cap=c.get('head_cap_ms',5000) if bridge else 1)
    if c.get('controller') in {'quality_rule','completion_cost_rule'}:
        from specrsi.learning.observations import make_rule_controller
        rules=make_rule_controller(config,rule='quality_only' if c['controller']=='quality_rule' else 'completion_cost',
            initial_fraction=c.get('rule_initial_fraction',.05),draft_fraction=c.get('rule_draft_fraction',.35),
            cost_meter=lambda elapsed,kind:max(1,math.ceil(elapsed*1000)))
        hooks={'outer_value':rules.outer_value,'batch_value':rules.batch_value}
    result=SearchEngine(scope,initial,ledger,BudgetLedger(c['budget_ms']),executor,drafter,config,**hooks).run()
    output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    data=result.to_dict();data.update(author_checkpoint_used=False,paper_budget_comparable=False,
        source_changed_during_run=initial_sources['sha256']!=source_fingerprint(ROOT)['sha256'] or checkpoint_inputs!=required_artifacts(c,'science-specrsi'),
        source_fingerprint=initial_sources,initial_acquisition_ms=setup.spent,elapsed_seconds=time.monotonic()-started,
        physical_checker_calls=checker.physical_checks,controller=c.get('controller','heuristic_bootstrap'),
        role_checkpoint_inputs=checkpoint_inputs,
        observation_records=bridge.observation_records if bridge else [],
        rule_variant=rules.variant if rules else None,
        audit_parameter_policy='Stable case:N routes persist unchanged into audit; legacy task-UID routes remain role-specific. No candidate remapping or audit feedback.',
        wall_cap_contract='Cooperative measured slices; actual overruns fail without clamping, not a hard OS wall budget')
    (output/'search.json').write_text(json.dumps(data,indent=2)+'\n')
    (output/'selected.json').write_text(json.dumps(result.selected.manifest(),indent=2)+'\n')
    from specrsi.core.models import Candidate
    # Stable case routes transfer unchanged; legacy task-UID controls use common parameters.
    audit_scope=make_scope(c['domain'],'audit',tuple(c['case_ids']));rows=[];physical=[]
    for task in audit_scope.tasks:
        r=executor.run(result.selected,task,audit_scope,c['execution_cap_ms'])
        rows.append({'task':task.uid,'status':r.status,'score':r.outcome.score if r.outcome else None,'cost_ms':r.actual_cost})
        physical.append(thaw(r.receipt))
    physical.extend(thaw(r.receipt) for r in ledger.records.values() if r.route=='U')
    if bridge:physical.extend(bridge.physical_inferences)
    if rules:physical.extend(rules.physical_inferences)
    (output/'audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'physical_calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in physical))


def run_worker(kind,c):
    output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    if kind=='science-specrsi':return run_science_specrsi(c)
    if kind=='science-native':
        from specrsi.science import evaluate,load_config
        rows=[]
        for role in c['scenario_sets']:
            for case_id in c['case_ids']:
                result=evaluate(c['domain'],load_config(c['domain'],role,case_id)['baseline'],scenario_set=role,case_id=case_id)
                name=f'{role}_case_{case_id:02d}.json';(output/name).write_text(json.dumps(result,indent=2)+'\n')
                rows.append({'role':role,'case_id':case_id,'complete':result['complete'],'valid':result['valid'],'file':name})
        (output/'summary.json').write_text(json.dumps({'all_complete':all(x['complete'] for x in rows),'results':rows},indent=2)+'\n');return
    if kind=='audit-artifact':
        source=Path(c['source_output']);marker=read(source/'completed.json')
        if marker['fingerprint']!=c['source_fingerprint']:raise ValueError('Source run identity changed')
        if any(sha(source/name)!=h for name,h in marker['output_sha256'].items()):raise ValueError('Frozen source artifact changed')
        if c['source_runner']=='prompt-baseline':
            selected_name='selected_candidate.json'
            rows=[{**json.loads(line),'status':'complete'} for line in (source/'audit.jsonl').read_text().splitlines() if line.strip()]
        elif c['source_runner']=='extended':
            selected_name='selected.json'
            rows=[{**row,'status':'complete' if row['complete'] else 'incomplete'} for row in read(source/'result.json')['audit']['items']]
        else:selected_name='selected.json';rows=read(source/'audit.json')
        (output/'selected.json').write_bytes((source/selected_name).read_bytes())
        (output/'audit.json').write_text(json.dumps(rows,indent=2)+'\n')
        (output/'audit_summary.json').write_text(json.dumps({'source_completed_fingerprint':marker['fingerprint'],
            'source_completed_sha256':sha(source/'completed.json'),
            'source_output':str(source),'all_complete':bool(rows) and all(x['status']=='complete' for x in rows),
            'rows':rows,'new_model_calls':0,'shared_audit_receipt_not_independent_run':True},indent=2)+'\n');return
    raise ValueError('Unknown internal worker')


def main(argv=None):
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',default=str(ROOT/'configs/experiments/prospective_studies.json'))
    parser.add_argument('--study',nargs='+',help='S01 ... S13; default all')
    parser.add_argument('--method',nargs='+',help='Filter explicit registered methods')
    parser.add_argument('--replicate',nargs='+',type=int,help='Zero-based paired replicate indices; default all reported counts')
    parser.add_argument('--output-root',default='results/studies')
    parser.add_argument('--plan-dir',default='artifacts/study_plan')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--plan',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',choices=['science-specrsi','science-native','audit-artifact'],help=argparse.SUPPRESS)
    parser.add_argument('--worker-config',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        if not args.worker_config:parser.error('--worker requires --worker-config')
        run_worker(args.worker,read(args.worker_config));return 0
    plan=build_plan(args.registry,studies=args.study,methods=args.method,replicates=args.replicate,
                    output_root=args.output_root,plan_root=args.plan_dir)
    materialize(plan)
    location=Path(args.plan_dir);location.mkdir(parents=True,exist_ok=True)
    (location/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'action':'execute' if args.execute else 'plan_only','jobs':len(plan['jobs']),
        'requested_unavailable_tracks':plan['unavailable_requested_tracks'],'historical_author_reconstruction':'unavailable',
        'plan':str(location/'plan.json'),'formal_jobs_started':False,'execution_requested':args.execute,
        'jobs_requiring_training':sum(j['status']=='requires_preparation' for j in plan['jobs'])},indent=2,ensure_ascii=False),flush=True)
    if args.execute:
        results=execute_plan(plan)
        (location/'execution_summary.json').write_text(json.dumps(results,indent=2)+'\n')
        return int(any(x['status']=='failed' for x in results))
    return 0


if __name__=='__main__':raise SystemExit(main())
