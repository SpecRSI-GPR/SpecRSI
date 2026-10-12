#!/usr/bin/env python3
"""Prepare development-only SFT records from real completed searches/checks."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from specrsi.learning.records import CheckerAudit,EditExample,save_jsonl,trajectory_from_dict


def verified_edits(raw_search, trajectory):
    """Use exact sampled continuations only when their child completed evaluation."""
    complete=set(raw_search['acceptance']['complete_candidates'])
    edits=[]
    for transition in raw_search['transitions']:
        frozen=[e.get('candidate_id') for e in transition['events'] if e.get('event')=='freeze']
        if len(frozen)!=1 or frozen[0] not in complete:continue
        decisions=[d for d in trajectory.decisions if d.kind=='edit_token' and d.transition_index==transition['index']]
        if not decisions:continue
        prefix=list(decisions[0].input_ids)
        labels=[-100]*len(prefix)
        for decision in decisions:
            if tuple(prefix)!=decision.input_ids:
                raise ValueError('Collected edit token prefixes are discontinuous')
            token=int(decision.support[decision.selected_index]);prefix.append(token);labels.append(token)
        edits.append(EditExample(trajectory.lineage_id,tuple(prefix),tuple(labels),True))
    return edits


def prepare(source, output, science_variants=8):
    metadata=json.loads((source/'collection_metadata_round_0.json').read_text())
    if metadata.get('split')!='development' or metadata.get('reference_tasks_only'):
        raise ValueError('A real development-only prompt collection is required')
    trajectories=[trajectory_from_dict(json.loads(line)) for line in (source/'trajectories_round_0.jsonl').read_text().splitlines() if line]
    by_id={t.search_id:t for t in trajectories}
    records=list(trajectories);proofs=[]
    for raw in metadata['raw_search_results']:
        t=by_id[raw['search_id']]
        for edit in verified_edits(raw,t):
            records.append(edit)
            proofs.append({'kind':'native_completed_edit','search_id':t.search_id,'lineage_id':t.lineage_id,
                           'complete_candidates':raw['acceptance']['complete_candidates']})
    # Separate native DEVELOPMENT cases: labels come from actual bounded checker
    # calls, with paid source records, never from equality sampled in a simulator.
    from specrsi.science.executor import make_scope,make_candidate,ScienceExecutor,mutate
    from specrsi.science.checker import ScienceParameterChecker
    from specrsi.core.models import thaw
    from specrsi.core.ledger import Ledger
    scope=make_scope('reactor','development',(0,1));parent=make_candidate('reactor')
    checker=ScienceParameterChecker();ledger=Ledger(checker=checker);executor=ScienceExecutor()
    initial=[]
    for task in scope.tasks:
        result=executor.run(parent,task,scope,120000)
        if result.status!='complete':raise RuntimeError(result.error)
        ledger.admit_direct(parent,task,scope,result.outcome,result.receipt)
        initial.append({'task':task.uid,'actual_cost_ms':result.actual_cost,'receipt':thaw(result.receipt)})
    for i in range(science_variants):
        rng=random.Random(20260916+i)
        parameters=mutate('reactor',thaw(parent.artifacts['parameters']),rng)
        changed=scope.tasks[i%2]
        child=make_candidate('reactor',name=f'warmup-{i}',parent_id=parent.uid,scenario_overrides={'case:'+str(changed.inputs['case_id']):parameters})
        for task in scope.tasks:
            cap=1000
            context=json.dumps({'checker':checker.version,'source_policy':'one_paid_parent','check_cap_ms':cap,
                'source':parent.manifest(),'child':child.manifest(),'task':{'id':task.uid,'inputs':thaw(task.inputs)},
                'scope':scope.uid},sort_keys=True,separators=(',',':'))
            result=checker.check(parent,child,task,scope,cap)
            completed=result.status in {'P','U'}
            records.append(CheckerAudit('development-reactor-'+str(i),checker.version+':one-paid-parent:1000ms',
                completed,int(result.status=='P') if completed else None,(0.,)*8,context))
            proofs.append({'kind':'actual_bounded_checker','child':child.uid,'task':task.uid,'status':result.status,
                'actual_cost_ms':result.actual_cost,'certificate':asdict(result.certificate) if result.certificate else None})
    output.mkdir(parents=True,exist_ok=True)
    save_jsonl(output/'warmup.jsonl',records)
    report={'author_records':False,'data_role':'development_only','source':str(source),
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'collection_metadata_round_0.json',source/'trajectories_round_0.jsonl']},
        'checker_version':checker.version,'science_source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/specrsi/science').glob('*.py'))},
        'warmup_sha256':hashlib.sha256((output/'warmup.jsonl').read_bytes()).hexdigest(),
        'records':len(records),'verified_edit_count':sum(isinstance(r,EditExample) for r in records),
        'surface_labels':{s:sum(p.get('status')==s for p in proofs) for s in ['P','U','interrupted']},
        'source_acquisition':initial,'proofs':proofs,
        'scope_note':'Prompt edits and values plus separately identified Reactor checker curriculum; no audit data or claimed population calibration.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['proofs','source_acquisition']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/training/qwen_prompt_gvo_final'))
    parser.add_argument('--output',type=Path,default=Path('data/training/prospective_v2'))
    parser.add_argument('--science-variants',type=int,default=8)
    args=parser.parse_args()
    if args.science_variants<1:parser.error('science-variants must be positive')
    prepare(args.source,args.output,args.science_variants)
if __name__=='__main__':main()
