#!/usr/bin/env python3
"""Supplied-PDF VATS on native scientific cases, followed by a frozen audit."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import yaml

from specrsi.core import BudgetLedger, SearchConfig
from specrsi.core.vats import VATSEngine
from specrsi.core.models import thaw
from specrsi.paper_profile import (algorithm_name, paper_provenance, require_role_checkpoint, vats_config,
                                   vats_cost_accounting, objective_coefficients, realized_search_return)
from specrsi.science.executor import (ParameterDrafter, ScienceExecutor, make_candidate,
                                     make_science_ledger, make_scope)


def run(config):
    search = config['search']
    if algorithm_name(search) != 'vats_pdf':
        raise ValueError('This runner requires the supplied-PDF VATS profile')
    paper = paper_provenance(config)
    checkpoint = require_role_checkpoint(search)
    objective_coefficients(search)
    cap = search.get('execution_cap_ms', 120000)
    base = SearchConfig(execution_cap=cap, draft_cap=search.get('draft_cap_ms', 60000),
        planning_cost=0, max_actions=search.get('max_actions', 12),
        max_candidates=search.get('max_candidates', 8), max_children=search.get('max_children', 2),
        seed=config['seed'], cost_unit='contracted-native-and-role-ms', all_u=True, reserve_completion=False)
    options = vats_config(base, search, learned_query_cap=search.get('head_cap_ms', 5000) if checkpoint else 0)
    budget = BudgetLedger(search['budget_ms'])
    output = Path(config['output'])
    # A fresh directory prevents accidental replacement of completed evidence.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'resolved_config.json').write_text(json.dumps(config, indent=2) + '\n')
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / 'src/specrsi').rglob('*.py'))}
    bridge = None
    checkpoint_hashes = {}
    if checkpoint is not None:
        from specrsi.learning.hf import HFRoleModel
        from specrsi.learning.paper_bridge import VATSSearchBridge
        from specrsi.learning.science_bridge import HFParameterDrafter
        checkpoint_hashes = {str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(checkpoint.rglob('*')) if p.is_file()}
        snapshot = hashlib.sha256(json.dumps(checkpoint_hashes, sort_keys=True).encode()).hexdigest()
        model = HFRoleModel.load(checkpoint, device=search.get('role_device', 'cuda:1'))
        model.eval()
        bridge = VATSSearchBridge(model, cost_unit='wall-ms', snapshot_id=snapshot)
        drafter = HFParameterDrafter(bridge, max_new_tokens=search.get('draft_max_tokens', 256),
            snapshot=snapshot, proposal_mode=search.get('proposal_mode', 'numeric_catalog'),
            catalog_size=search.get('catalog_size', 12))
    else:
        if search['controller'] != 'heuristic_bootstrap':
            raise ValueError('Choose learned or explicitly heuristic_bootstrap')
        drafter = ParameterDrafter()
    domain = config['domain']
    scope = make_scope(domain, 'selection', tuple(config['selection_cases']), native_utility=True)
    root = make_candidate(domain)
    native_calls = []

    class RecordingExecutor:
        def __init__(self):
            self.inner = ScienceExecutor()

        def run(self, candidate, task, scope, cap, resume_token=None):
            started = time.monotonic()
            result = self.inner.run(candidate, task, scope, cap, resume_token=resume_token)
            native_calls.append({'kind': 'native_execution', 'candidate': candidate.uid, 'task': task.uid,
                'scope': scope.uid, 'split': scope.split, 'cost_ms': result.actual_cost, 'cap_ms': cap,
                'elapsed_seconds': time.monotonic() - started, 'gpu_count': 0, 'status': result.status,
                'receipt': thaw(result.receipt), 'error': result.error})
            if result.actual_cost > cap:
                raise RuntimeError('Native executor exceeded its registered cap')
            return result

    ledger, executor = make_science_ledger(), RecordingExecutor()
    initial_cost, initial_results = 0, []
    for task in scope.tasks:
        result = executor.run(root, task, scope, cap)
        initial_cost += result.actual_cost
        initial_results.append({'task': task.uid, 'status': result.status,
                                'receipt': thaw(result.receipt), 'error': result.error})
        if result.status != 'complete' or not result.outcome.admissible:
            raise RuntimeError('Initial root is incomplete or inadmissible: ' + str(result.error))
        ledger.admit_direct(root, task, scope, result.outcome, result.receipt)
    if bridge is not None:
        options = replace(options, scorer_snapshot=bridge.snapshot_id, policy_snapshot=bridge.snapshot_id)
    hooks = {'scorer': bridge.scorer, 'policy_menu': bridge.policy_menu} if bridge else {}
    engine = VATSEngine(scope, root, ledger, budget, executor, drafter, options, **hooks)
    started = time.monotonic()
    result = engine.run()
    report = result.to_dict()
    report.update({'config': config, 'paper_reference': paper, 'scope': scope.manifest(),
        'role_checkpoint_hashes': checkpoint_hashes, 'source_hashes': sources,
        'native_utility': True, 'initial_acquisition_ms': initial_cost, 'initial_acquisition': initial_results,
        'author_checkpoint_used': False, 'paper_budget_comparable': False,
        'wall_seconds': time.monotonic() - started,
        'search_return': realized_search_return(result, search),
        'objective_coefficients': dict(zip(('cost_lambda', 'time_mu'), objective_coefficients(search))),
        'cost_accounting': vats_cost_accounting('wall-ms', options.policy_call_cost,
                                               model.config.architecture_version if bridge is not None else 2),
        'source_changed_during_run': sources != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'src/specrsi').rglob('*.py'))}})
    # This immutable choice is saved before either candidate is run on audit inputs.
    (output / 'search.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'selected.json').write_text(json.dumps(result.selected.manifest(), indent=2) + '\n')
    audit_scope = make_scope(domain, 'audit', tuple(config['audit_cases']), native_utility=True)
    audit = []
    for task in audit_scope.tasks:
        row = {'task': task.uid}
        for label, candidate in (('initial', root), ('selected', result.selected)):
            acquired = executor.run(candidate, task, audit_scope, cap)
            row[label] = {'status': acquired.status, 'cost_ms': acquired.actual_cost,
                'native_score': acquired.outcome.fields['native_score'] if acquired.outcome else None,
                'admissible': acquired.outcome.admissible if acquired.outcome else None,
                'receipt': thaw(acquired.receipt), 'error': acquired.error}
        audit.append(row)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    calls = native_calls + (bridge.physical_inferences if bridge else [])
    (output / 'physical_calls.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in calls))
    return {'output': str(output), 'selection_native_gain': result.final_f - result.initial_f,
            'selection_native_utility': result.acceptance.selected_score,
            'audit_complete': all(r[k]['status'] == 'complete' for r in audit for k in ('initial', 'selected')),
            'controller': search['controller'], 'algorithm': 'vats_pdf'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(args.config.read_text())), indent=2))


if __name__ == '__main__':
    main()
