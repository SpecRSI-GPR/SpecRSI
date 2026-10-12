"""Prospective paired mechanism experiments backed by real scientific kernels.

No learned-head results are inferred. Candidate pools are explicit immutable JSON
parameter programs; all admitted outcomes come from native execution or proofs.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

from specrsi.core.acquisition import AcquisitionCaps,ResultAcquirer
from specrsi.core.budget import BudgetExceeded,BudgetLedger,CostCapViolation
from specrsi.core.ledger import Ledger
from specrsi.core.models import Candidate,digest,thaw
from specrsi.science.checker import ScienceParameterChecker
from specrsi.science.executor import ScienceExecutor,make_candidate,make_scope,mutate,selected_parameters


def child_edit(parent,scope,rng,index):
    domain=parent.runtime['domain']
    task=scope.tasks[rng.randrange(len(scope.tasks))]
    parameters=mutate(domain,selected_parameters(parent,task,scope),rng)
    overrides=thaw(parent.artifacts['scenario_overrides'])
    overrides[task.uid]=parameters
    return make_candidate(domain,thaw(parent.artifacts['parameters']),f'pool-edit-{index}',parent.uid,overrides)


def frozen_pool(domain,scope,count,seed):
    if count<2:raise ValueError('A fixed pool needs an incumbent and at least one child')
    rng=random.Random(seed)
    pool=[make_candidate(domain)]
    for index in range(1,count):pool.append(child_edit(pool[-1],scope,rng,index))
    return pool


def run_variant(config,scope,pool,*,reuse,seed):
    checker=ScienceParameterChecker();ledger=Ledger(checker)
    setup_budget=BudgetLedger(config.get('setup_budget_ms',10000000))
    budget=BudgetLedger(config.get('budget_ms',10000000))
    executor=ScienceExecutor()
    execute_cap=config.get('execution_cap_ms',180000)
    check_cap=config.get('checker_cap_ms',1000)
    source_limit=config.get('max_sources',1) if reuse else 0
    caps=AcquisitionCaps(execute_cap,0,check_cap,source_limit)
    setup=ResultAcquirer(ledger,setup_budget,executor)
    incumbent=pool[0]
    started=time.monotonic()
    for task in scope.tasks:
        result=setup.acquire(incumbent,task,scope,(),caps)
        if result.record is None or not result.record.outcome.admissible:raise RuntimeError('Initial native reference must be complete and valid')
    setup_elapsed=time.monotonic()-started
    acquirer=ResultAcquirer(ledger,budget,executor)
    observed=[incumbent];events=[];online_rng=random.Random(seed)
    variant_started=time.monotonic();stopped=None
    for index in range(1,len(pool)):
        candidate=pool[index] if config['mode']=='fixed_pool' else child_edit(incumbent,scope,online_rng,index)
        source_order=[p for p in reversed(observed) if p.uid==candidate.parent_id]
        source_order += [p for p in reversed(observed) if p.uid!=candidate.parent_id]
        candidate_events=[]
        for task in scope.tasks:
            started=time.monotonic()
            try:result=acquirer.acquire(candidate,task,scope,source_order,caps)
            except BudgetExceeded:
                stopped='insufficient_remaining_budget_for_declared_checker_plus_fallback_envelope';break
            except CostCapViolation as exc:
                stopped='explicit_backend_cost_cap_violation:'+str(exc);break
            event={'candidate_id':candidate.uid,'task_id':task.uid,'case_id':task.inputs['case_id'],
                   'status':result.status,'route':result.record.route if result.record else None,'from_cache':result.from_cache,
                   'actual_cost_ms':result.actual_cost,'elapsed_seconds':time.monotonic()-started,
                   'checks':[asdict(c) for c in result.checks],'error':result.error}
            candidate_events.append(event)
            if result.record is None:
                stopped='native_evaluation_incomplete';break
        events.extend(candidate_events)
        observed.append(candidate)
        if ledger.complete(candidate,scope):
            if all(o.admissible for o in ledger.outcomes(candidate,scope)):
                if ledger.score(candidate,scope)>ledger.score(incumbent,scope):incumbent=candidate
        if stopped:break
    elapsed=time.monotonic()-variant_started
    provenance=ledger.audit_provenance()
    complete=[c for c in observed if ledger.complete(c,scope)]
    valid=[c for c in complete if all(o.admissible for o in ledger.outcomes(c,scope))]
    scored={c.uid:{'score':ledger.score(c,scope),'outcomes':[o.manifest() for o in ledger.outcomes(c,scope)]} for c in complete}
    # This is a trace receipt, never an outcome supplied to another run.
    return {'variant':'scoped_reuse' if reuse else 'all_direct','mode':config['mode'],
            'seed':seed,'selected_id':incumbent.uid,'selected_score':ledger.score(incumbent,scope),
            'candidate_manifests':[c.manifest() for c in observed],'complete_candidate_results':scored,
            'events':events,'physical_checker_calls':checker.physical_checks,
            'accounting':{'charged_search_ms':budget.spent,'charged_setup_ms':setup_budget.spent,
                'end_to_end_search_seconds':elapsed,'end_to_end_setup_seconds':setup_elapsed,
                'native_cpu_seconds':sum(r.receipt.get('cpu_seconds',0.) for r in ledger.records.values() if r.route=='U'),
                'checker_wall_seconds':sum(x['wall_seconds'] for x in checker.physical_checks if x['operation']=='check'),
                'provenance_recheck_wall_seconds':sum(x['wall_seconds'] for x in checker.physical_checks if x['operation']=='provenance_verify'),
                'unit':'actual native/check wall milliseconds; extra host work included in end-to-end time',
                'cooperative_deadline':True,'budget_breached':budget.breached},
            'statistics':{'P':sum(x['route']=='P' for x in events),'U':sum(x['route']=='U' for x in events),
                'checks':sum(len(x['checks']) for x in events),'lookup_cache_hits':sum(x['from_cache'] for x in events),
                'attempted_candidates':len(observed),'complete_candidates':len(complete),'valid_complete_candidates':len(valid),
                'provenance_records':provenance,'stopped_reason':stopped}}


def run_pair(config,seed):
    if config['mode'] not in {'fixed_pool','online'}:raise ValueError('Unknown mechanism mode')
    domain=config.get('domain','reactor')
    scope=make_scope(domain,'selection',tuple(config.get('case_ids',[0,1])))
    pool=frozen_pool(domain,scope,int(config.get('candidate_count',6)),seed)
    # The immutable manifest is fixed before either variant evaluates any child.
    pool_manifest=[c.manifest() for c in pool]
    order=[False,True] if seed%2==0 else [True,False]
    runs={}
    for reuse in order:
        result=run_variant(config,scope,pool,reuse=reuse,seed=seed)
        runs[result['variant']]=result
    direct,reuse=runs['all_direct'],runs['scoped_reuse']
    common=set(direct['complete_candidate_results'])&set(reuse['complete_candidate_results'])
    for uid in common:
        if direct['complete_candidate_results'][uid]!=reuse['complete_candidate_results'][uid]:
            raise RuntimeError('Real direct and P/U evaluations disagree for a common frozen candidate')
    complete_both=all(r['statistics']['complete_candidates']==len(pool) for r in (direct,reuse))
    same_selection=direct['selected_id']==reuse['selected_id']
    if config['mode']=='fixed_pool' and complete_both and not same_selection:
        raise RuntimeError('Complete identical fixed pool selected different results')
    return {'implementation':'prospective_native_science_mechanism_control','author_exact':False,
            'paper_budget_comparable':False,'learned_heads_used':False,'proposal_method':'seeded_single_case_parameter_edits',
            'seed':seed,'scope':scope.manifest(),
            'pool_before_evaluation':pool_manifest if config['mode']=='fixed_pool' else None,
            'pool_sha256':digest(pool_manifest) if config['mode']=='fixed_pool' else None,
            'online_proposal_rule':None if config['mode']=='fixed_pool' else {'seed':seed,'parent':'current fully valid incumbent','limit':len(pool)},
            'variant_execution_order':['scoped_reuse' if r else 'all_direct' for r in order],
            'variants':runs,'comparison':{'common_complete_candidates':len(common),'common_complete_outcomes_equal':True,
                'both_finished_entire_pool':complete_both,'same_selected_candidate':same_selection,
                'charged_search_ms_saved':direct['accounting']['charged_search_ms']-reuse['accounting']['charged_search_ms'],
                'end_to_end_search_seconds_saved':direct['accounting']['end_to_end_search_seconds']-reuse['accounting']['end_to_end_search_seconds']}}


def run_experiment(config):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2)+'\n')
    summaries=[]
    for seed in config.get('seeds',[0]):
        result=run_pair(config,int(seed))
        path=output/f'pair_{seed}.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        summaries.append({'seed':seed,**result['comparison'],'path':str(path),
                          'reuse_statistics':result['variants']['scoped_reuse']['statistics']})
        print(json.dumps(summaries[-1]),flush=True)
    summary={'mode':config['mode'],'domain':config.get('domain','reactor'),'pairs':len(summaries),
             'author_exact':False,'paper_budget_comparable':False,'learned_heads_used':False,'results':summaries}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
