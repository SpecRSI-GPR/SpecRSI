"""Compare native-science searches on frozen, common heldout scenarios.

The base controls share the trained HF drafter; explicitly named parameter
adaptations use the pinned Qwen3.5 BF16 service and their actual search rules.
Every method uses the same complete native utility, development root/cases,
expense/attempt limits and (when registered) per-method elapsed-time limit.
Completed stages resume unchanged. Interrupted stages retain their physical
charges and receive distinct attempt records. Candidate selection completes
for all methods before heldout evaluation starts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time
import uuid

from specrsi.core import BudgetLedger, SearchConfig
from specrsi.core.acceptance import accept
from specrsi.core.models import Candidate, digest, thaw
from specrsi.core.vats import VATSEngine, VATSPlan
from specrsi.planning.completion import CompletionProblem
from specrsi.paper_profile import (algorithm_name, objective_coefficients,
    paper_provenance, require_role_checkpoint, realized_search_return, vats_config,
    vats_cost_accounting)
from specrsi.science.executor import ScienceExecutor, make_candidate, make_scope, make_science_ledger
from specrsi.science.registry import ROOT, case_count
from specrsi.science.named_comparison import (NAMED_METHODS, UNAVAILABLE_METHODS,
    SearchDeadline, run_named_method, validate_named_config)

METHODS = ('initial', 'best_of_n', 'random_search', 'specrsi')
SUPPORTED_METHODS = METHODS + NAMED_METHODS


class ComparisonVATSEngine(VATSEngine):
    """Deployment-only external wall admission, preserving measured bills.

    A comparison's wall cap is separate from the paper's additive expense B.
    If preparation or selected work cannot fit its entire declared slice, do
    no new physical work and stop at the best already complete candidate. This
    guard never fabricates an evaluator result or forces finalist completion.
    """
    deadline = None
    wall_stopped = False

    def _prepare(self):
        if self.deadline is not None and (self.wall_stopped or not self.deadline.fits(self.config.prep_cap)):
            self.wall_stopped = True
            events = [{'event': 'comparison_wall_stop', 'phase': 'before_preparation', 'cost': 0}]
            self.state.history.extend(events)
            return VATSPlan('accept'), CompletionProblem((), {}, 0, 0), events
        return super()._prepare()

    def _execute_plan(self, plan, problem):
        if plan.kind != 'accept' and self.deadline is not None and not self.deadline.fits(plan.work_cap):
            self.wall_stopped = True
            return [{'event': 'comparison_wall_stop', 'phase': 'before_selected_work',
                     'selected_plan': plan.uid, 'executed': False, 'cost': 0}]
        return super()._execute_plan(plan, problem)


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _hash_file(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def _tree_hashes(root, paths):
    return {str(p.relative_to(root)): _hash_file(p) for p in sorted(set(paths)) if p.is_file()}


def _source_hashes():
    return _tree_hashes(ROOT, list((ROOT / 'src/specrsi').rglob('*.py'))
        + list((ROOT / 'src/specrsi').rglob('*.cc'))
        + list((ROOT / 'configs/science').rglob('*.json'))
        + [ROOT / 'scripts/run_paper_science_comparison.py'])


class Journal:
    """Sync every event, including starts, so process death is never free work."""
    def __init__(self, path, stage):
        self.path = Path(path)
        self.stage = stage
        self.lock = threading.Lock()

    def append(self, row):
        with self.lock, self.path.open('a') as handle:
            handle.write(json.dumps({'stage': self.stage, **row}, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())


class RecordingExecutor:
    parallel_safe = True

    def __init__(self, journal):
        self.inner = ScienceExecutor()
        self.journal = journal

    def run(self, candidate, task, scope, cap, resume_token=None):
        call_id = uuid.uuid4().hex
        self.journal.append({'event': 'start', 'kind': 'native_execution', 'call_id': call_id,
            'candidate': candidate.uid, 'task': task.uid, 'scope': scope.uid,
            'split': scope.split, 'cap_ms': cap})
        started = time.monotonic()
        try:
            result = self.inner.run(candidate, task, scope, cap, resume_token=resume_token)
        except BaseException as error:
            self.journal.append({'event': 'error', 'kind': 'native_execution', 'call_id': call_id,
                'elapsed_seconds': time.monotonic() - started, 'error': repr(error),
                'actual_cost_unknown': True})
            raise
        self.journal.append({'event': 'finish', 'kind': 'native_execution', 'call_id': call_id,
            'cost_ms': result.actual_cost, 'elapsed_seconds': time.monotonic() - started,
            'gpu_count': 0, 'status': result.status, 'receipt': thaw(result.receipt), 'error': result.error,
            'outcome': {'score': result.outcome.score, 'fields': thaw(result.outcome.fields),
                        'admissible': result.outcome.admissible} if result.outcome else None})
        if result.actual_cost > cap:
            raise RuntimeError('Native executor exceeded its registered cap; actual bill retained')
        return result


class PhysicalRecords(list):
    def __init__(self, journal):
        super().__init__()
        self.journal = journal

    def append(self, row):
        self.journal.append({'event': 'physical_inference', **row})
        super().append(row)

    def extend(self, rows):
        for row in rows:
            self.append(row)


def _load_roles(checkpoint, search):
    from specrsi.learning.hf import HFRoleModel
    model = HFRoleModel.load(checkpoint, device=search.get('role_device', 'cuda:1'))
    model.eval()
    return model


def _roles(model, search, snapshot, journal):
    from specrsi.learning.paper_bridge import VATSSearchBridge
    from specrsi.learning.science_bridge import HFParameterDrafter
    bridge = VATSSearchBridge(model, cost_unit='wall-ms', snapshot_id=snapshot)
    bridge.physical_inferences = PhysicalRecords(journal)
    drafter = HFParameterDrafter(bridge, max_new_tokens=search.get('draft_max_tokens', 256),
        top_k=search.get('draft_top_k', 32), temperature=search.get('temperature', 1.),
        exploration=search.get('exploration', .1), snapshot=snapshot,
        proposal_mode=search.get('proposal_mode', 'numeric_catalog'), catalog_size=search.get('catalog_size', 12))
    return bridge, drafter


def _options(config, snapshot):
    search = config['search']
    base = SearchConfig(execution_cap=search.get('execution_cap_ms', 120000),
        draft_cap=search.get('draft_cap_ms', 60000), planning_cost=0,
        max_actions=search.get('max_actions', 12), max_candidates=search.get('max_candidates', 8),
        max_children=search.get('max_children', 2), seed=config['seed'],
        mode=search.get('mode', 'deploy'), temperature=search.get('temperature', 1.),
        exploration=search.get('exploration', .1),
        cost_unit='contracted-native-and-role-ms', all_u=True, reserve_completion=False)
    options = vats_config(base, search, learned_query_cap=search.get('head_cap_ms', 5000))
    return replace(options, scorer_snapshot=snapshot, policy_snapshot=snapshot)


def validate_comparison_resources(config, methods):
    """Reject registered caps that make the initial nonempty work impossible.

    This uses declared upper bounds and case counts only; no evaluator outcome
    or heldout observation influences resource registration. For the complete
    initial root, VATS's first productive menu option must be a DRAFT.
    """
    wall = config['search'].get('wall_limit_seconds')
    SearchDeadline(wall)
    if wall is None or set(methods) <= {'initial'}:
        return {}
    options = _options(config, 'resource-preflight-no-model-forward')
    if (options.draft_cap <= 0 or options.max_attempts < 1 or options.max_actions < 1
            or options.max_candidates < 2 or options.max_children < 1):
        raise ValueError('Comparison registration leaves no initial nonempty search work')
    required = len(config['search_cases'])
    native_wall = math.ceil(required / options.workers) * options.execution_cap
    receipt = {}
    for method in methods:
        expense, elapsed, attempts = 0, 0, 0
        if method == 'specrsi':
            expense = elapsed = options.prep_cap + options.draft_cap
            attempts = 1
        elif method in ('best_of_n', 'random_search'):
            expense = options.draft_cap + required * options.execution_cap
            elapsed = options.draft_cap + native_wall
            attempts = 1 + required
        elif method in NAMED_METHODS:
            expense = elapsed = options.draft_cap
            attempts = 1
        if (expense > config['search']['budget_ms'] or elapsed > wall * 1000
                or attempts > options.max_attempts):
            raise ValueError(f'Comparison caps give {method} no initial nonempty work opportunity: '
                             f'need expense_ms={expense}, wall_ms={elapsed}, attempts={attempts}')
        receipt[method] = {'first_nonempty_expense_cap_ms': expense,
                           'first_nonempty_wall_cap_ms': elapsed, 'primitive_attempts': attempts}
    return receipt


def _control(method, engine, bridge, options, journal, deadline=None):
    """Full native evaluations; no learned value/policy calls in these controls."""
    state, rng = engine.state, random.Random(options.seed)
    root, ledger, budget = state.incumbent, state.ledger, state.budget
    initial = ledger.score(root, state.scope)
    attempts, proposals = 0, []
    started = time.monotonic()
    for iteration in range(options.max_actions):
        if len(state.candidates) >= options.max_candidates:
            break
        required = len(state.scope.tasks)
        # Eq. (16): N_{t+1} = N_t - n_t. A DRAFT consumes one
        # primitive attempt even when its edit is invalid/duplicate; the full
        # evaluation control must reserve that attempt AND every child check.
        # Otherwise a nominal N-attempt control can execute N checks plus an
        # uncounted number of drafts, violating the matched resource contract.
        # Section G.1.3 explicitly includes failed edits in n_t.
        if attempts + 1 + required > options.max_attempts:
            break
        # Reserve a complete child before sampling it, like a full-evaluation control.
        cap = options.draft_cap + required * options.execution_cap
        wall_cap = options.draft_cap + math.ceil(required / options.workers) * options.execution_cap
        if cap > budget.available or (deadline is not None and not deadline.fits(wall_cap)):
            break
        parents = [c for c in state.candidates.values() if ledger.admissible(c, state.scope)]
        parent = root if method == 'best_of_n' else rng.choice(parents)
        identity = f'control-{iteration}'
        budget.reserve(identity, cap, 'full_child')
        state.current_transition_index = iteration
        draft = engine.drafter.draft(parent, state, options.draft_cap, rng)
        attempts += 1
        spent = draft.actual_cost
        proposal = {'parent_id': parent.uid, 'draft_status': draft.status, 'draft_cost_ms': spent,
                    'draft_error': draft.error, 'candidate': draft.candidate.manifest() if draft.candidate else None}
        proposals.append(proposal)
        if draft.actual_cost > options.draft_cap:
            budget.settle(identity, spent)
            raise RuntimeError('HF draft exceeded its registered slice; physical bill retained')
        if draft.candidate is not None and draft.candidate.uid not in state.candidates:
            candidate = draft.candidate
            state.candidates[candidate.uid] = candidate
            tasks = state.scope.tasks
            if options.workers > 1:
                with ThreadPoolExecutor(max_workers=min(options.workers, len(tasks))) as pool:
                    futures = [pool.submit(engine.executor.run, candidate, task, state.scope, options.execution_cap)
                               for task in tasks]
                    results = [future.result() for future in futures]
            else:
                results = [engine.executor.run(candidate, task, state.scope, options.execution_cap) for task in tasks]
            # Only this coordinator mutates the observed ledger and budget.
            for task, result in zip(tasks, results):
                spent += result.actual_cost
                attempts += 1
                if result.status == 'complete':
                    ledger.admit_direct(candidate, task, state.scope, result.outcome, result.receipt)
                state.history.append({'kind': 'execution', 'candidate_id': candidate.uid,
                    'task_id': task.uid, 'status': result.status, 'error': result.error})
        budget.settle(identity, spent)
        state.version += 1
        state.remaining_attempts = options.max_attempts - attempts
    eligible = [c for c in state.candidates.values() if ledger.admissible(c, state.scope)]
    # The same fixed acceptance/tie rule is applied to every method.
    selected = accept(root, eligible, ledger, state.scope).selected
    final = ledger.score(selected, state.scope)
    return selected, {'method': method, 'initial_native_utility': initial,
        'selected_native_utility': final, 'selection_native_gain': final - initial,
        'search_budget_ms': budget.total, 'search_cost_ms': budget.spent,
        'budget_remaining_ms': budget.available,
        'native_attempts': attempts - len(proposals),
        'draft_attempts': len(proposals), 'primitive_attempts': attempts,
        'remaining_attempts': options.max_attempts - attempts,
        'proposals': proposals, 'charges': [asdict(x) for x in budget.charges],
        'search_elapsed_seconds': time.monotonic() - started}


def _search_stage(method, config, scope, options, model, snapshot, journal):
    executor, root, ledger = RecordingExecutor(journal), make_candidate(config['domain']), make_science_ledger()
    initial_cost = 0
    # Each method pays for its own initial acquisition; this is outside B for all.
    for task in scope.tasks:
        result = executor.run(root, task, scope, options.execution_cap)
        initial_cost += result.actual_cost
        if result.status != 'complete' or not result.outcome.admissible:
            raise RuntimeError(f'Initial native acquisition failed: {result.error}')
        ledger.admit_direct(root, task, scope, result.outcome, result.receipt)
    initial = ledger.score(root, scope)
    deadline = SearchDeadline(config['search'].get('wall_limit_seconds'))
    if method == 'initial':
        selected, report = root, {'method': method, 'initial_native_utility': initial,
            'selected_native_utility': initial, 'selection_native_gain': 0.,
            'search_budget_ms': config['search']['budget_ms'], 'search_cost_ms': 0,
            'search_elapsed_seconds': 0.}
    elif method in NAMED_METHODS:
        selected, report = run_named_method(method, config, scope, root, ledger,
                                             executor, options, journal, deadline)
    else:
        bridge, drafter = _roles(model, config['search'], snapshot, journal)
        engine = ComparisonVATSEngine(scope, root, ledger, BudgetLedger(config['search']['budget_ms']),
            executor, drafter, options, scorer=bridge.scorer, policy_menu=bridge.policy_menu)
        engine.deadline = deadline
        bridge.initial_budget, bridge.workers, bridge.contract_id = engine.state.budget.total, options.workers, scope.uid
        if method == 'specrsi':
            started = time.monotonic()
            result = engine.run()
            selected, report = result.selected, result.to_dict()
            report.update(method=method, initial_native_utility=result.acceptance.incumbent_score,
                selected_native_utility=result.acceptance.selected_score,
                selection_native_gain=result.final_f - result.initial_f,
                search_budget_ms=config['search']['budget_ms'], search_cost_ms=result.total_cost,
                search_elapsed_seconds=time.monotonic() - started,
                search_return=realized_search_return(result, config['search']),
                cost_accounting=vats_cost_accounting('wall-ms', options.policy_call_cost,
                    model.config.architecture_version))
        else:
            selected, report = _control(method, engine, bridge, options, journal, deadline)
    report.update(deadline.report())
    report.update(initial_acquisition_ms=initial_cost, selected=selected.manifest(),
        selected_id=selected.uid, checkpoint_snapshot=snapshot,
        algorithm=('vats_pdf' if method == 'specrsi' else
                   'prospective_' + method if method in NAMED_METHODS else 'native_hf_reference_control'))
    return report


def _validate(config):
    search = config['search']
    if algorithm_name(search) != 'vats_pdf' or search.get('controller') != 'learned':
        raise ValueError('Comparison requires learned vats_pdf and a real shared HF checkpoint')
    checkpoint = require_role_checkpoint(search)
    objective_coefficients(search)
    BudgetLedger(search['budget_ms'])
    if config.get('search_split', 'development') != 'development' or config.get('heldout_split', 'selection') != 'selection':
        raise ValueError('This development runner requires development search and selection holdout; audit is sealed')
    methods = tuple(config.get('methods', METHODS))
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(SUPPORTED_METHODS):
        raise ValueError('Unknown or duplicate comparison methods')
    validate_named_config(config)
    SearchDeadline(search.get('wall_limit_seconds'))
    if search.get('wall_limit_seconds') is not None and search.get('mode', 'deploy') != 'deploy':
        raise ValueError('Comparison wall admission is a deployment guard, not an actor-training collection profile')
    for key in ('search_cases', 'heldout_cases'):
        cases = config[key]
        if not cases or len(set(cases)) != len(cases) or any(type(c) is not int or not 0 <= c < case_count(config['domain']) for c in cases):
            raise ValueError(f'Invalid registered {key}')
    if set(config.get('training_cases', ())) & set(config['search_cases']):
        raise ValueError('Training and development comparison case IDs must be disjoint')
    if type(config.get('max_stage_attempts', 2)) is not int or config.get('max_stage_attempts', 2) < 1:
        raise ValueError('max_stage_attempts must be a positive integer')
    validate_comparison_resources(config, methods)
    return checkpoint, methods


def _bill(output):
    events = []
    truncated = False
    for path in output.rglob('physical_calls.jsonl'):
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                truncated = True
    started = {x['call_id'] for x in events if x.get('event') == 'start'}
    finished = {x['call_id'] for x in events if x.get('event') == 'finish'}
    native = [x for x in events if x.get('event') == 'finish' and x.get('kind') == 'native_execution']
    inferences = [x for x in events if x.get('event') == 'physical_inference']
    abandoned = list(output.glob('methods/*/attempt_*/abandoned.json'))
    unfinished = [p for p in output.glob('methods/*/attempt_*/started.json') if not (p.parent / 'result.json').exists()]
    return {'native_cost_ms': sum(x['cost_ms'] for x in native), 'native_calls': len(native),
        'native_elapsed_seconds': math.fsum(x['elapsed_seconds'] for x in native),
        'native_cpu_seconds': math.fsum(x.get('receipt', {}).get('cpu_seconds', 0.) for x in native),
        'model_load_seconds': math.fsum(x['elapsed_seconds'] for x in events if x.get('event') == 'model_load'),
        'role_elapsed_seconds': math.fsum(x.get('elapsed_seconds', 0.) for x in inferences),
        'role_gpu_seconds': math.fsum(x.get('elapsed_seconds', 0.) * x.get('gpu_count', 0) for x in inferences),
        'prompt_tokens': sum(x.get('prompt_tokens', 0) for x in inferences),
        'completion_tokens': sum(x.get('completion_tokens', 0) for x in inferences),
        'unfinished_native_calls': len(started - finished), 'truncated_journal': truncated,
        'abandoned_search_attempts': len(abandoned), 'unfinished_search_attempts': len(unfinished),
        'physical_bill_complete': not (started - finished or truncated or abandoned or unfinished),
        'note': 'Includes failed/restarted stages; unknown interrupted work is never reported as zero. Search quota and physical time differ.'}


def _recorded_heldout_call(path, candidate, task, scope):
    """Recover a terminal journal receipt or seal unknown work; never repeat it."""
    if not path.exists():
        return None
    try:
        events = [json.loads(line) for line in path.read_text().splitlines()]
    except json.JSONDecodeError:
        return {'status': 'error', 'cost_ms': None, 'outcome': None, 'receipt': {},
                'error': 'Truncated heldout journal; unreceipted tasks are sealed without rerun',
                'actual_cost_unknown': True}
    starts = [event for event in events if event.get('event') == 'start'
              and (event['candidate'], event['task'], event['scope']) == (candidate.uid, task.uid, scope.uid)]
    if not starts:
        return None
    if len(starts) == 1:
        finished = next((event for event in events if event.get('event') == 'finish'
                         and event['call_id'] == starts[0]['call_id']), None)
        if finished is not None:
            return finished
    return {'status': 'error', 'cost_ms': None, 'outcome': None,
            'receipt': {'unresolved_calls': [event['call_id'] for event in starts]},
            'error': 'Interrupted or ambiguous heldout execution sealed without rerun',
            'actual_cost_unknown': True}


def _heldout_stage(method, candidate, scope, options, path, freeze_digest):
    path.mkdir(parents=True, exist_ok=True)
    journal_path = path / 'physical_calls.jsonl'
    executor = RecordingExecutor(Journal(journal_path, 'heldout:' + method))
    rows = []
    for task in scope.tasks:
        receipt_path = path / (task.uid + '.json')
        if receipt_path.exists():
            row = _read(receipt_path)
            if row['candidate_id'] != candidate.uid or row['freeze_digest'] != freeze_digest:
                raise ValueError('Heldout receipt identity changed')
        else:
            event = _recorded_heldout_call(journal_path, candidate, task, scope)
            if event is None:
                try:
                    executor.run(candidate, task, scope, options.execution_cap)
                except Exception:
                    # The journal records either measured completion or unknown
                    # expense. Neither authorizes another run of this heldout task.
                    pass
                event = _recorded_heldout_call(journal_path, candidate, task, scope)
                if event is None:
                    raise RuntimeError('Heldout executor did not emit a start receipt')
            outcome = event.get('outcome')
            row = {'task_id': task.uid, 'candidate_id': candidate.uid, 'freeze_digest': freeze_digest,
                'status': event['status'], 'cost_ms': event.get('cost_ms'), 'error': event.get('error'),
                'admissible': outcome['admissible'] if outcome else False,
                'native_score': outcome['fields']['native_score'] if outcome else None,
                'receipt': event.get('receipt', {}), 'actual_cost_unknown': event.get('actual_cost_unknown', False)}
            _write(receipt_path, row)
        rows.append(row)
    complete = all(row['status'] == 'complete' for row in rows)
    admissible = complete and all(row['admissible'] for row in rows)
    utility = math.fsum(weight * row['native_score'] for weight, row in zip(scope.weights, rows)) if admissible else None
    return {'method': method, 'candidate_id': candidate.uid, 'complete': complete,
        'admissible': admissible, 'native_utility': utility, 'items': rows}


def _check_barrier(output, expected_digest, barrier, barrier_sha256):
    frozen_path = output / 'frozen_candidates.json'
    if not frozen_path.exists():
        raise ValueError('Heldout requires already frozen candidates')
    actual = digest(_read(frozen_path))
    if expected_digest is not None and expected_digest != actual:
        raise ValueError('Frozen-candidate digest does not match heldout registration')
    if barrier is None:
        return
    barrier = Path(barrier)
    if _hash_file(barrier) != barrier_sha256:
        raise ValueError('Global freeze barrier SHA-256 changed')
    entries = _read(barrier).get('comparisons')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Global freeze barrier has no comparisons')
    seen, matched = set(), False
    for entry in entries:
        path = Path(entry['output']).resolve()
        if path in seen:
            raise ValueError('Duplicate comparison in global freeze barrier')
        seen.add(path)
        frozen = path / 'frozen_candidates.json'
        if not frozen.is_file() or digest(_read(frozen)) != entry['frozen_candidates_digest']:
            raise ValueError('Global freeze barrier contains missing or changed candidates')
        if path == output.resolve():
            matched = True
            if entry['frozen_candidates_digest'] != actual:
                raise ValueError('Current comparison does not match global freeze barrier')
    if not matched:
        raise ValueError('Current comparison is absent from global freeze barrier')
    pin = {'freeze_barrier': str(barrier.resolve()), 'freeze_barrier_sha256': barrier_sha256,
           'frozen_candidates_digest': actual, 'comparison_count': len(entries)}
    destination = output / 'heldout_authorization.json'
    if destination.exists() and _read(destination) != pin:
        raise ValueError('Heldout global freeze authorization changed')
    if not destination.exists():
        _write(destination, pin)


def run(config, *, stage='all', expected_freeze_digest=None, freeze_barrier=None, freeze_barrier_sha256=None):
    """Resume completed stages only when every registered input still matches."""
    if stage not in {'all', 'search', 'heldout'}:
        raise ValueError('Unknown comparison stage')
    if stage == 'heldout' and not expected_freeze_digest:
        raise ValueError('Heldout requires an explicit expected frozen-candidate digest')
    if bool(freeze_barrier) != bool(freeze_barrier_sha256):
        raise ValueError('Global freeze barrier requires both a path and SHA-256')
    checkpoint, methods = _validate(config)
    config = json.loads(json.dumps(config))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A comparison process already owns this output') from error
        return _run_locked(config, checkpoint, methods, output, stage, expected_freeze_digest,
                           freeze_barrier, freeze_barrier_sha256)


def _run_locked(config, checkpoint, methods, output, stage, expected_freeze_digest, freeze_barrier, freeze_barrier_sha256):
    checkpoint_hashes = _tree_hashes(checkpoint, checkpoint.rglob('*'))
    snapshot = digest(checkpoint_hashes)
    options = _options(config, snapshot)
    scopes = {'search': make_scope(config['domain'], 'development', tuple(config['search_cases']), native_utility=True),
              'heldout': make_scope(config['domain'], 'selection', tuple(config['heldout_cases']), native_utility=True)}
    source_hashes = _source_hashes()
    named_identity = validate_named_config(config)
    manifest = {'version': 'native-development-comparison-v1', 'config': config,
        'paper': paper_provenance(config), 'source_hashes': source_hashes,
        'checkpoint_hashes': checkpoint_hashes, 'checkpoint_snapshot': snapshot,
        'scopes': {key: scope.manifest() for key, scope in scopes.items()},
        'methods': list(methods), 'native_utility': True, 'author_exact': False,
        'official_audit_consumed': False, 'paper_budget_comparable': False,
        'paper_named_baselines_missing': ['Alg.-Only', 'Direct-LLM', 'Embed-GP', 'Janus'],
        'named_science_adaptations': [method for method in methods if method in NAMED_METHODS],
        'named_model_identity': named_identity,
        'unavailable_scientific_methods': UNAVAILABLE_METHODS,
        'control_definitions': {'initial': 'fixed initial policy, zero search budget spent',
            'best_of_n': 'independent root HF proposals; full native evaluation; best admissible child or root',
            'random_search': 'uniform admissible parent, shared HF edit law; full native evaluation; best admissible node',
            'specrsi': 'learned supplied-PDF vats_pdf with the same HF drafter',
            'janus_science': 'Actual Janus co-evolution, arithmetic proxy DSL and registered numeric catalog; prospective parameter-space adaptation',
            'hgm_science': 'Actual HGM clade Thompson allocation with registered numeric catalog; prospective parameter-space adaptation',
            'aflow_science': 'Actual AFlow mixed parent sampling, repeated validation and experience; numeric parameter space replaces workflow DAG editing'},
        'cost_contract': {'search_budget_ms_per_method': config['search']['budget_ms'],
            'wall_limit_seconds_per_method': config['search'].get('wall_limit_seconds'),
            'primitive_attempts_per_method': options.max_attempts,
            'native_execution_cap_ms': options.execution_cap,
            'generation_cap_ms': options.draft_cap,
            'initial_nonempty_work_admission': validate_comparison_resources(config, methods),
            'initial_acquisition': 'paid independently for every method, outside search budget',
            'heldout': 'paid independently for every frozen method, outside search budget',
            'retries': 'separately retained; invalidate equal-budget comparison',
            'physical_bill': 'every native call and model inference, including failed attempts'},
        'holdout_rule': 'Freeze every method before any common selection-split evaluation; no adaptive use of holdout'}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if _read(manifest_path) != manifest:
            raise ValueError('Resume refused: registered config, source, split, or checkpoint changed')
    else:
        if any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError('Nonempty comparison output lacks an immutable manifest')
        _write(manifest_path, manifest)
    summary_path = output / 'summary.json'
    if summary_path.exists() and stage != 'search':
        _check_barrier(output, expected_freeze_digest, freeze_barrier, freeze_barrier_sha256)
        return _read(summary_path)
    reports, model = {}, None
    for method in methods:
        folder = output / 'methods' / method
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / 'search.json'
        if result_path.exists():
            reports[method] = _read(result_path)
            continue
        if stage == 'heldout':
            raise ValueError('Heldout cannot run missing search stages')
        if (output / 'frozen_candidates.json').exists():
            raise ValueError('Frozen comparison is missing a search stage; refusing post-holdout search')
        # A fully written attempt can be promoted without repeating any work.
        attempts = sorted(folder.glob('attempt_*'))
        completed = next((p for p in attempts if (p / 'result.json').exists()), None)
        if completed:
            reports[method] = _read(completed / 'result.json')
            _write(result_path, reports[method])
            continue
        for attempt in attempts:
            if not (attempt / 'abandoned.json').exists():
                _write(attempt / 'abandoned.json', {'reason': 'process interrupted or stage failed',
                    'bill_complete': False, 'prior_work_retained': True})
        if len(attempts) >= config.get('max_stage_attempts', 2):
            raise RuntimeError(f'{method} exhausted bounded stage retries; prior bills retained')
        attempt = folder / f'attempt_{len(attempts) + 1:03d}'
        attempt.mkdir()
        journal = Journal(attempt / 'physical_calls.jsonl', 'search:' + method)
        _write(attempt / 'started.json', {'method': method, 'registered_budget_ms': config['search']['budget_ms']})
        try:
            if method in ('best_of_n', 'random_search', 'specrsi') and model is None:
                started = time.monotonic()
                model = _load_roles(checkpoint, config['search'])
                journal.append({'event': 'model_load', 'elapsed_seconds': time.monotonic() - started})
            reports[method] = _search_stage(method, config, scopes['search'], options, model, snapshot, journal)
            reports[method]['attempt_count'] = len(attempts) + 1
            _write(attempt / 'result.json', reports[method])
            _write(result_path, reports[method])
        except BaseException as error:
            _write(attempt / 'error.json', {'error': repr(error), 'prior_work_retained': True})
            _write(output / 'physical_bill.json', _bill(output))
            raise
    frozen = {'manifest_digest': digest(manifest), 'candidates': {
        method: {'id': row['selected_id'], 'manifest': row['selected']} for method, row in reports.items()}}
    frozen_path = output / 'frozen_candidates.json'
    if frozen_path.exists():
        if _read(frozen_path) != frozen:
            raise ValueError('Frozen candidates changed')
    else:
        _write(frozen_path, frozen)
    # Hash source again at the barrier. A mutable checkout is not valid evidence.
    if _source_hashes() != source_hashes or _tree_hashes(checkpoint, checkpoint.rglob('*')) != checkpoint_hashes:
        raise RuntimeError('Source or checkpoint changed during search; heldout remains sealed')
    if stage == 'search':
        bill = _bill(output)
        result = {'output': str(output), 'domain': config['domain'], 'seed': config['seed'],
            'status': 'search_complete', 'frozen_candidates_digest': digest(frozen),
            'methods': list(methods), 'physical_bill': bill, 'sota_established': False,
            'equal_budget_comparison': all(row.get('attempt_count', 1) == 1 and row.get('wall_limit_respected', True)
                for row in reports.values()) and bill['physical_bill_complete'],
            'named_science_adaptations': manifest['named_science_adaptations'],
            'unavailable_scientific_methods': manifest['unavailable_scientific_methods'],
            'paper_named_baselines_missing': manifest['paper_named_baselines_missing']}
        _write(output / 'search_summary.json', result)
        _write(output / 'physical_bill.json', bill)
        return result
    _check_barrier(output, expected_freeze_digest, freeze_barrier, freeze_barrier_sha256)
    heldout = {}
    for method in methods:
        candidate = Candidate(**frozen['candidates'][method]['manifest'])
        if candidate.uid != frozen['candidates'][method]['id']:
            raise ValueError('Candidate artifact hash does not match frozen identity')
        heldout[method] = _heldout_stage(method, candidate, scopes['heldout'], options,
            output / 'heldout' / method, digest(frozen))
    stable = _source_hashes() == source_hashes and _tree_hashes(checkpoint, checkpoint.rglob('*')) == checkpoint_hashes
    bill = _bill(output)
    summary = {'output': str(output), 'domain': config['domain'], 'seed': config['seed'],
        'status': 'complete' if stable and all(x['complete'] for x in heldout.values()) else 'incomplete',
        'frozen_candidates_digest': digest(frozen), 'source_and_checkpoint_stable': stable,
        'heldout': heldout, 'search': reports, 'physical_bill': bill,
        'equal_budget_comparison': all(row.get('attempt_count', 1) == 1 and row.get('wall_limit_respected', True)
            for row in reports.values()) and bill['physical_bill_complete'],
        'named_science_adaptations': manifest['named_science_adaptations'],
        'unavailable_scientific_methods': manifest['unavailable_scientific_methods'],
        'sota_established': False, 'paper_named_baselines_missing': manifest['paper_named_baselines_missing'],
        'limitation': 'Registered scientific parameter adaptations and frozen selection-split comparison; unavailable methods, author-exact replications and official benchmark audit remain outside this claim.'}
    _write(output / 'physical_bill.json', bill)
    _write(summary_path, summary)
    return _read(summary_path)
