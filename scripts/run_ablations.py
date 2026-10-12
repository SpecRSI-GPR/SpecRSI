#!/usr/bin/env python3
"""Actual nested role-replica searches and frozen paired-state interventions.

Planning and donor preparation use no GPUs. Execution must be explicit.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import glob
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('specrsi_study_launcher',ROOT/'scripts/run_studies.py')
studies=importlib.util.module_from_spec(spec);spec.loader.exec_module(studies)

PRIMARY=('sft','rloo','gvo','no_p','item_reward','grpo')
INFERENCE={
    'outer_blind':{'observation_mode':'outer_blind'},'inner_blind':{'observation_mode':'inner_blind'},
    'evidence_blind':{'observation_mode':'evidence_blind'},'coarse_counts':{'observation_mode':'coarse_counts'},
    'additive':{'batch_mode':'additive'},'blind_additive':{'observation_mode':'outer_blind','batch_mode':'additive'},
    'sham_relabel':{'observation_mode':'sham_relabel'},'quality_rule':{'controller':'quality_rule'},
    'completion_cost_rule':{'controller':'completion_cost_rule'},'no_exploration':{'draft_exploration':0.}}


def merge(base,overrides):
    result=copy.deepcopy(base)
    for key,value in overrides.items():
        result[key]=merge(result.get(key,{}),value) if isinstance(value,dict) else copy.deepcopy(value)
    return result


def build_nested(config,*,methods=None,replicas=None,searches=None):
    source=studies.source_fingerprint(ROOT)
    methods=methods or config['methods']
    allowed=set(config['methods'])|set(config.get('optional_methods',[]))
    if not set(methods)<=allowed:raise ValueError('Unknown registered nested method')
    replicas=list(range(config['training_replicas'])) if replicas is None else replicas
    searches=list(range(config['searches_per_replica'])) if searches is None else searches
    if any(len(values)!=len(set(values)) for values in (methods,replicas,searches)):
        raise ValueError('Duplicate methods or replica/search indices would duplicate an experimental unit')
    if any(type(i) is not int or not 0<=i<config['training_replicas'] for i in replicas):raise ValueError('Invalid training replica index')
    if any(type(i) is not int or not 0<=i<config['searches_per_replica'] for i in searches):raise ValueError('Invalid nested search index')
    base_plan=studies.build_plan(studies=[config['study']],methods=['specrsi'],replicates=[0],_fingerprint=source)
    templates=[j for j in base_plan['jobs'] if j['task']==config['task']]
    if len(templates)!=1:raise ValueError('Nested target must name one available native/prompt/workflow SpecRSI task')
    template=templates[0]
    if template['runner'] not in {'specrsi','science-specrsi'}:raise ValueError('Nested learning requires an actual SpecRSI execution target')
    jobs=[];artifact_cache={}
    for method in methods:
        trained=method.removeprefix('trained_') if method.startswith('trained_') else method if method in PRIMARY else 'gvo'
        intervention=INFERENCE.get(method.removeprefix('trained_'),{}) if method not in PRIMARY else {}
        training_index=PRIMARY.index(trained) if trained in PRIMARY else 6+sorted(INFERENCE).index(trained)
        for replica in replicas:
            training_seed=config['training_seed_base']+training_index*10000+replica
            training_root=f"artifacts/training/nested-{source['sha256'][:12]}/{config['task']}"
            stage='warmup' if trained=='sft' else 'learned'
            checkpoint=f'{training_root}/{trained}/replica-{training_seed}/{stage}/checkpoint'
            for search in searches:
                c=merge(template['config'],config.get('science_overrides' if template['runner']=='science-specrsi' else 'prompt_overrides',{}))
                c.pop('output',None)
                c.update(seed=config['search_seed_base']+replica*10000+search,training_method=trained,
                    training_replica=training_seed,training_output_root=training_root)
                settings=c if template['runner']=='science-specrsi' else c['search']
                settings.update(controller='learned',role_checkpoint=checkpoint,draft_exploration=.1)
                settings.update(intervention)
                c['study_provenance']={'study':'T01_nested_training','base_task_recipe':config['study'],
                    'method':method,'training_replica_index':replica,'training_seed':training_seed,'nested_search_index':search,
                    'experimental_unit':'independent_training_replica','nested_searches_are_not_independent_replicas':True,
                    'author_exact':False,'original_unit_ids':None,'paper_budget_comparable':False,
                    'intervention':'development_retrained' if method.startswith('trained_') else 'inference_only_same_GVO_checkpoint' if method in INFERENCE else 'training_method_comparison'}
                cache_key=(checkpoint,trained)
                if cache_key not in artifact_cache:artifact_cache[cache_key]=studies.required_artifacts(c,template['runner'])
                required=copy.deepcopy(artifact_cache[cache_key])
                identity=hashlib.sha256(studies.canonical({'config':c,'source':source['sha256'],'artifacts':required}).encode()).hexdigest()
                uid=f"T01/{config['task']}/{method}/replica_{replica:02d}/search_{search:02d}"
                output=Path(config['output_root'])/uid/identity[:12];c['output']=str(output)
                job={**template,'id':uid,'study':'T01','method':method,'seed':c['seed'],'config':c,
                    'training_replica':replica,'nested_search':search,'replicate':replica,'reported_n':config['training_replicas'],
                    'pair_id':f"T01/{config['task']}/replica_{replica:02d}/search_{search:02d}",
                    'output':str(output),'config_path':str(Path(config['plan_root'])/'jobs'/uid/(identity[:12]+'.json')),
                    'fingerprint':identity,'source_fingerprint':source['sha256'],'required_artifacts':required,'dependencies':[],
                    'status':'requires_preparation' if any(x['status']!='present' for x in required) else 'runnable_prospective',
                    'notes':['Independent method-specific training seeds; summarize 12 nested searches within each replica before across-replica statistics.']}
                job['command']=studies.command_for(job);job['shell_display']=__import__('shlex').join(job['command']);jobs.append(job)
    return {'schema_version':1,'author_exact':False,'source_fingerprint':source,'jobs':jobs,'job_count':len(jobs),
        'registry':'configs/experiments/prospective_studies.json','request':{'plan_root':config['plan_root']},
        'nested_config':config,'nested_filters':{'methods':methods,'replicas':replicas,'searches':searches},
        'unavailable_requested_tracks':[],'declared_training_replicas':config['training_replicas'],
        'declared_nested_searches_per_replica':config['searches_per_replica'],
        'source_counts':'5 source-named methods × 8 independently trained replicas × 12 nested searches = 480; optional controls are additional experiments'}


def summarize_nested(plan):
    groups={}
    for job in plan['jobs']:
        key=(job['method'],job['training_replica']);row={'search_index':job['nested_search'],'complete':studies.valid_completion(job),'output':job['output']}
        if row['complete']:
            output=Path(job['output']);search=studies.read(output/'search.json');audit=studies.read(output/'audit.json')
            row.update(final_f=search['final_f'],search_cost=search['total_cost'],cost_unit=search['cost_unit'],
                audit_score=sum(x['score'] for x in audit)/len(audit))
        groups.setdefault(key,[]).append(row)
    rows=[]
    for (method,replica),searches in groups.items():
        complete=all(x['complete'] for x in searches) and len(searches)==plan['declared_nested_searches_per_replica']
        rows.append({'method':method,'training_replica':replica,'complete_full_replica':complete,'searches':searches,
            'mean_final_f':statistics.mean(x['final_f'] for x in searches) if complete else None,
            'mean_audit_score':statistics.mean(x['audit_score'] for x in searches) if complete else None})
    methods=[]
    for method in sorted({x['method'] for x in rows}):
        observed=[x for x in rows if x['method']==method];complete=all(x['complete_full_replica'] for x in observed) and len(observed)==plan['declared_training_replicas']
        values=[x['mean_final_f'] for x in observed] if complete else []
        methods.append({'method':method,'complete_eight_replica_endpoint':complete,'n_training_replicas':len(values),
            'mean_replica_return':statistics.mean(values) if values else None,
            'sample_sd_across_replicas':statistics.stdev(values) if len(values)>1 else None})
    return {'author_exact':False,'experimental_unit':'independent_training_replica','replicas':rows,'methods':methods,
        'collapse_definition':None,'collapse_counts':None,'original_author_effects_reconstructed':False}


def observation_groups(paths):
    """Only actual full pre-truncation observation records; no synthetic fillers."""
    groups=[];rejected=[];seen_states=set()
    for filename in sorted(set(paths)):
        path=Path(filename);source=studies.read(path)
        if source.get('source_changed_during_run'):
            rejected.append({'file':str(path),'reason':'source_changed_during_run'});continue
        if not source.get('search_id'):
            rejected.append({'file':str(path),'reason':'missing_actual_engine_search_id'});continue
        records=source.get('observation_records',source.get('controller_observations',[]));pool={}
        checkpoint=None;checkpoint_hashes={}
        if source.get('role_checkpoint_inputs'):
            checkpoint=source['role_checkpoint_inputs'][0]['path'];checkpoint_hashes=source['role_checkpoint_inputs'][0]['sha256']
        elif source.get('config',{}).get('search',{}).get('role_checkpoint'):
            checkpoint=source['config']['search']['role_checkpoint']
            checkpoint_hashes={str(Path(name).relative_to(checkpoint)):value for name,value in source.get('role_checkpoint_hashes',{}).items()}
        for row in records:
            if row.get('mode')!='full' or row.get('kind') not in {'outer','batch'}:continue
            try:context=json.loads(row['context_text'])
            except (ValueError,TypeError):continue
            key=(row['key'].split(':',1)[0],row['kind'],row['context_text'],tuple(row['features']))
            pool.setdefault(key,[]).append(row)
        for key,rows in pool.items():
            if len(rows)<2 or any(len(r['options'])!=1 for r in rows):continue
            state_identity=(source['search_id'],key[0],key[1])
            if state_identity in seen_states:continue
            support=[r['key'].split(':',2)[2] for r in rows]
            if len(set(support))!=len(support):continue
            decisions=[d for d in source.get('decisions',[]) if str(d.get('state_version'))==key[0]
                       and d.get('kind')==key[1] and d.get('support')==support]
            if len(decisions)!=1:
                rejected.append({'file':str(path),'state_version':key[0],'kind':key[1],
                                 'reason':'observation options lack a unique matching recorded legal decision support'})
                continue
            seen_states.add(state_identity)
            context=json.loads(rows[0]['context_text'])
            record={'source_file':str(path),'source_sha256':studies.sha(path),'search_id':source['search_id'],'state_version':key[0],
                'kind':key[1],'features':rows[0]['features'],'context_text':rows[0]['context_text'],
                'options':[r['options'][0] for r in rows],'support':support,
                'original_selected_id':decisions[0]['selected_id'],
                'original_values':decisions[0]['values'],
                'role_checkpoint':checkpoint,'role_checkpoint_sha256':checkpoint_hashes,
                'cohort':[context.get('scope'),len(context.get('tasks',[])),len(context.get('candidates',[])),len(rows)]}
            record['id']=hashlib.sha256(studies.canonical(record).encode()).hexdigest();groups.append(record)
    return groups,rejected


def prepare_pairs(config):
    from specrsi.learning.observations import views,FrozenDonor
    p=config['paired_states'];paths=[filename for pattern in p['inputs'] for filename in glob.glob(pattern,recursive=True)]
    states,rejected=observation_groups(paths);pairs=[];used=set();required=p['pairs_per_stratum']
    for kind in p['strata']:
        count=0
        for state in states:
            if state['kind']!=kind or state['id'] in used:continue
            for other in states:
                if other['id']==state['id'] or other['id'] in used or other['kind']!=kind or other['cohort']!=state['cohort']:continue
                # Different real searches; no duplicated pair or same-run sibling donor.
                if other['search_id']==state['search_id']:continue
                donor=FrozenDonor.from_view(views(other['features'],other['context_text'],other['options'],kind),donor_id=other['id'])
                try:changed=views(state['features'],state['context_text'],state['options'],kind,mode='mismatched_state',donor=donor)
                except ValueError:continue
                pairs.append({'id':hashlib.sha256((state['id']+other['id']).encode()).hexdigest(),'kind':kind,
                    'receiver':state,'donor':other,'frozen_intervened_view':asdict(changed)})
                used.update((state['id'],other['id']));count+=1;break
            if count==required:break
    counts={kind:sum(row['kind']==kind for row in pairs) for kind in p['strata']}
    result={'author_exact':False,'source_fingerprint':studies.source_fingerprint(ROOT),'pairs':pairs,'counts':counts,
        'required_pairs_per_stratum':required,'complete_registered_cohorts':all(n==required for n in counts.values()),
        'observed_eligible_states':len(states),'rejected_sources':rejected,
        'missing_pairs':{kind:max(0,required-n) for kind,n in counts.items()},
        'endpoint':'choice changes on frozen observed legal supports; no fabricated optimal-action labels, regret, both-correct counts, or paper loss'}
    output=Path(p['output']);output.mkdir(parents=True,exist_ok=True)
    (output/'paired_states.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def execute_pairs(config,prepared,*,allow_partial=False):
    if not prepared['complete_registered_cohorts'] and not allow_partial:raise ValueError('Insufficient observed pairs; first run more registered nested searches. No paired-state rows will be invented.')
    if not prepared['pairs']:raise ValueError('No compatible independent actual-state pairs')
    source=studies.source_fingerprint(ROOT)
    if source['sha256']!=prepared['source_fingerprint']['sha256']:raise ValueError('Sources changed since donor preparation')
    p=config['paired_states']
    # Validate frozen inputs before either cached return or a model load.
    for pair in prepared['pairs']:
        for state in (pair['receiver'],pair['donor']):
            if studies.sha(state['source_file'])!=state['source_sha256']:
                raise ValueError('Frozen state source changed')
        if p.get('checkpoint_policy')=='receiver_frozen_checkpoint':
            state=pair['receiver'];path=state.get('role_checkpoint');expected=state.get('role_checkpoint_sha256',{})
            if not path or not expected:raise ValueError('Receiver needs its actual frozen checkpoint and hashes')
            if any(studies.sha(Path(path)/name)!=value for name,value in expected.items()):
                raise ValueError('Receiver role checkpoint changed since observation')
    fixed_weights={}
    if p.get('checkpoint_policy')!='receiver_frozen_checkpoint':
        fixed_weights={str(x):studies.sha(x) for x in Path(p['checkpoint']).rglob('*') if x.is_file()}
    identity=hashlib.sha256(studies.canonical({'config':p,'prepared':prepared,'fixed_checkpoint':fixed_weights}).encode()).hexdigest()
    output=Path(p['output'])/'runs'/identity[:12]
    marker_path=output/'completed.json'
    if marker_path.exists():
        marker=studies.read(marker_path)
        if marker.get('identity')==identity and all(studies.sha(output/name)==digest for name,digest in marker['outputs'].items()):
            return studies.read(output/'paired_state_results.json')
    if output.exists():output.rename(output.with_name(output.name+'.incomplete-'+str(time.time_ns())))
    output.mkdir(parents=True)
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2)+'\n')
    from specrsi.learning.hf import HFRoleModel
    from specrsi.learning.bridge import NeuralSearchBridge
    from specrsi.learning.records import DecisionRecord
    from specrsi.learning.observations import views,FrozenDonor
    rows=[];loaded=None;weights={};model=None;bridge=None
    for pair in sorted(prepared['pairs'],key=lambda x:x['receiver'].get('role_checkpoint') or ''):
        state=pair['receiver'];other=pair['donor']
        for item in (state,other):
            if studies.sha(item['source_file'])!=item['source_sha256']:raise ValueError('Frozen state source changed')
        path=state.get('role_checkpoint') if p.get('checkpoint_policy')=='receiver_frozen_checkpoint' else p['checkpoint']
        if not path:raise ValueError('Receiver has no frozen actual role checkpoint; cannot manufacture replay weights')
        checkpoint=Path(path)
        expected=state.get('role_checkpoint_sha256',{}) if p.get('checkpoint_policy')=='receiver_frozen_checkpoint' else {}
        if p.get('checkpoint_policy')=='receiver_frozen_checkpoint' and not expected:raise ValueError('Missing receiver checkpoint hashes')
        if any(studies.sha(checkpoint/name)!=value for name,value in expected.items()):raise ValueError('Receiver role checkpoint changed since observation')
        if loaded!=path:
            if model is not None:
                del bridge,model
                import torch
                if torch.cuda.is_available():torch.cuda.empty_cache()
            weights[path]={str(x.relative_to(checkpoint)):studies.sha(x) for x in checkpoint.rglob('*') if x.is_file()}
            if 'model_config.json' not in weights[path]:raise ValueError('Missing actual paired-state role checkpoint')
            model=HFRoleModel.load(checkpoint,device=p.get('device','cuda:1'));model.eval();bridge=NeuralSearchBridge(model,cost_unit='wall-ms');loaded=path
        donor=FrozenDonor.from_view(views(other['features'],other['context_text'],other['options'],other['kind']),donor_id=other['id'])
        values={};receipts=[]
        for mode in ['full',*p['modes']]:
            view=views(state['features'],state['context_text'],state['options'],state['kind'],mode=mode,donor=donor)
            text=bridge.bounded_text(view.context_text,model.config.max_length//3)
            record=DecisionRecord(state['kind'],tuple(state['support']),0,0.,state_features=view.features,option_features=view.options,context_text=text)
            started=time.monotonic()
            import torch
            error=None
            try:
                with torch.no_grad():prediction=model.decision_values(record).detach().cpu().tolist()
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}';raise
            finally:
                elapsed=time.monotonic()-started
                receipt={'pair_id':pair['id'],'mode':mode,'wall_seconds':elapsed,'charged_ms':math.ceil(elapsed*1000),
                    'gpu_count':1 if str(model.device).startswith('cuda') else 0,'prompt_tokens':len(model.tokenizer.encode(text)),
                    'completion_tokens':0,'role_checkpoint':path,'status':'error' if error else 'complete','error':error,
                    'observation':asdict(view)}
                with (output/'physical_calls.jsonl').open('a') as journal:journal.write(json.dumps(receipt)+'\n')
            if any(not math.isfinite(x) for x in prediction):raise ValueError('Nonfinite actual role value')
            values[mode]=prediction;receipts.append(receipt)
        choice=lambda scores:max(range(len(scores)),key=lambda index:scores[index])
        rows.append({'pair_id':pair['id'],'kind':state['kind'],'values':values,'choices':{mode:choice(scores) for mode,scores in values.items()},
            'changed_from_full':{mode:choice(scores)!=choice(values['full']) for mode,scores in values.items() if mode!='full'},
            'physical_inferences':receipts,'role_checkpoint':path,'optimal_action':None,'regret':None})
        # Persist completed pairs immediately; incomplete runs are not counted complete.
        (output/'paired_state_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    for path,expected in weights.items():
        if expected!={str(x.relative_to(path)):studies.sha(x) for x in Path(path).rglob('*') if x.is_file()}:raise ValueError('Role checkpoint changed during paired scoring')
    result={'author_exact':False,'complete_registered_cohorts':prepared['complete_registered_cohorts'],
        'checkpoint_policy':p.get('checkpoint_policy'),'checkpoint_sha256':weights,'pairs':rows,'endpoint':prepared['endpoint']}
    if studies.source_fingerprint(ROOT)['sha256']!=source['sha256']:raise ValueError('Sources changed during paired scoring')
    (output/'paired_state_results.json').write_text(json.dumps(result,indent=2)+'\n')
    names=['resolved_config.json','physical_calls.jsonl','paired_state_results.json']
    marker={'identity':identity,'outputs':{name:studies.sha(output/name) for name in names},'complete_registered_cohorts':prepared['complete_registered_cohorts']}
    temporary=output/'completed.tmp';temporary.write_text(json.dumps(marker,indent=2)+'\n');os.replace(temporary,marker_path)
    (Path(p['output'])/'latest.json').write_text(json.dumps({'output':str(output),'identity':identity,'complete_registered_cohorts':prepared['complete_registered_cohorts']},indent=2)+'\n')
    return result


def main(argv=None):
    os.chdir(ROOT);parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiments/prospective_ablations.json')
    parser.add_argument('--methods',nargs='+');parser.add_argument('--replicas',nargs='+',type=int);parser.add_argument('--searches',nargs='+',type=int)
    parser.add_argument('--prepare-pairs',action='store_true');parser.add_argument('--execute-pairs',action='store_true')
    parser.add_argument('--allow-partial-diagnostic',action='store_true')
    group=parser.add_mutually_exclusive_group();group.add_argument('--plan',action='store_true');group.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv);config=studies.read(args.config)
    if args.plan and args.execute_pairs:parser.error('--plan cannot execute paired-state inference')
    if args.execute and (args.prepare_pairs or args.execute_pairs):parser.error('Use either nested --execute or the explicit paired-state mode')
    if args.prepare_pairs or args.execute_pairs:
        prepared=prepare_pairs(config)
        print(json.dumps({key:prepared[key] for key in ['counts','missing_pairs','complete_registered_cohorts']},indent=2))
        if args.execute_pairs:execute_pairs(config,prepared,allow_partial=args.allow_partial_diagnostic)
        return 0
    plan=build_nested(config,methods=args.methods,replicas=args.replicas,searches=args.searches)
    studies.materialize(plan);output=Path(config['plan_root']);output.mkdir(parents=True,exist_ok=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    commands={tuple(a['preparation_command']) for j in plan['jobs'] for a in j['required_artifacts']}
    print(json.dumps({'action':'execute' if args.execute else 'plan_only','nested_search_jobs':len(plan['jobs']),
        'training_recipes':len(commands),'formal_jobs_started':False,'plan':str(output/'plan.json')},indent=2))
    if args.execute:
        rebuild=lambda old:build_nested(old['nested_config'],**old['nested_filters'])
        records=studies.execute_plan(plan,rebuild=rebuild)
        latest=studies.read(output/'plan.json');(output/'summary.json').write_text(json.dumps(summarize_nested(latest),indent=2)+'\n')
        return int(any(r['status']=='failed' for r in records))
    return 0


if __name__=='__main__':raise SystemExit(main())
