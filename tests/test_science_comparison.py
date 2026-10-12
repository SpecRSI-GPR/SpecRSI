"""CPU checks of frozen holdout ordering, actual bills, and crash recovery."""
from types import SimpleNamespace
from pathlib import Path
import json

import pytest

from specrsi.core.execution import ExecutionResult
from specrsi.core.models import Outcome, thaw
from specrsi.core.search import DraftResult
from specrsi.core.vats import heuristic_vats_score, heuristic_vats_preference
from specrsi.science.executor import make_candidate, selected_parameters
from specrsi import science_comparison as comparison


@pytest.fixture
def named_service(monkeypatch):
    from specrsi.models.client import LocalModelClient, ModelReply
    from specrsi.paper_profile import REFERENCE_BACKBONE_REVISION
    requests = []

    def health(client):
        return {'model': 'Qwen3.5-35B-A3B', 'revision': REFERENCE_BACKBONE_REVISION,
                'protocol_version': 2}

    def generate(client, messages, **kwargs):
        requests.append((client.base_url, client.model, messages, kwargs))
        prompt = messages[-1]['content']
        text = json.dumps({'name': 'observed-proxy', 'expressions': ['target_days']}) if '"expressions"' in prompt else '{"edit":0}'
        return ModelReply(text, 30, 5, .001, 'stop', REFERENCE_BACKBONE_REVISION, str(len(requests)))

    monkeypatch.setattr(LocalModelClient, 'health', health)
    monkeypatch.setattr(LocalModelClient, 'generate', generate)
    return requests


def add_named_config(config, methods):
    config['methods'] = methods
    config['search']['wall_limit_seconds'] = 600
    config['search']['vats']['max_attempts'] = 12
    config['search']['max_candidates'] = 6
    config['named_baselines'] = {'model': {
        'base_url': 'http://127.0.0.1:18085', 'name': 'Qwen3.5-35B-A3B',
        'lock': str(comparison.ROOT / 'configs/models/qwen35_35b_a3b_bf16.lock.json')},
        'janus_science': {'proposals_per_promotion': 1, 'minimum_archive': 2},
        'aflow_science': {'validation_repeats': 1}}
    return config


def config_for(tmp_path, domain='irp'):
    checkpoint = tmp_path / 'roles'
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / 'adapters').mkdir(exist_ok=True)
    (checkpoint / 'heads.pt').write_bytes(b'cpu-test-fixture')
    (checkpoint / 'model_config.json').write_text(json.dumps({
        'backend': 'hf', 'architecture_version': 2, 'option_dim': 12, 'feature_dim': 8}))
    return {'domain': domain, 'seed': 7, 'output': str(tmp_path / 'comparison'),
        'search_cases': [4], 'heldout_cases': [4], 'training_cases': [0, 1],
        'search': {'algorithm': 'vats_pdf', 'controller': 'learned',
            'role_checkpoint': str(checkpoint), 'role_device': 'cpu',
            'budget_ms': 500, 'execution_cap_ms': 10, 'draft_cap_ms': 5,
            'max_candidates': 3, 'max_actions': 4, 'head_cap_ms': 1,
            'vats': {'max_queries': 1, 'max_attempts': 4, 'max_draft_depth': 1,
                'max_draft_parents': 1, 'max_draft_plans': 2, 'max_extensions': 0,
                'solver_backend': 'exhaustive'}}}


@pytest.fixture
def instrumented(monkeypatch, tmp_path):
    config = config_for(tmp_path)
    calls, role_loads, role_methods = [], [], []
    monkeypatch.setattr(comparison, '_source_hashes', lambda: {'unit-test-source': 'frozen'})

    class Executor:
        parallel_safe = True

        def run(self, candidate, task, scope, cap, resume_token=None):
            if task.inputs['scenario_set'] == 'selection':
                folder = tmp_path / 'comparison'
                assert (folder / 'frozen_candidates.json').exists()
                assert all((folder / 'methods' / method / 'search.json').exists()
                    for method in config.get('methods', comparison.METHODS))
            params = selected_parameters(candidate, task, scope)
            native = 1000. + params['target_days']
            # Opposite monotonicity deliberately exposes accidental bounded-score selection.
            result = ExecutionResult('complete', 3, Outcome(1. / native,
                {'native_score': native, 'valid': True,
                 'evaluator_identity': task.metadata['evaluator_identity']}),
                {'test_cpu_receipt': True})
            calls.append((candidate.uid, task.uid, scope.split))
            return result

    def load(checkpoint, search):
        role_loads.append(str(checkpoint))
        return SimpleNamespace(config=SimpleNamespace(architecture_version=2))

    def roles(model, search, snapshot, journal):
        role_methods.append((journal.stage, snapshot))
        bridge = SimpleNamespace(scorer=heuristic_vats_score,
            policy_menu=lambda state, plans: [heuristic_vats_preference(state, plan) for plan in plans])

        class Drafter:
            def draft(self, parent, state, cap, rng):
                params = thaw(parent.artifacts['parameters'])
                params['target_days'] += .1 + rng.random() * .01
                candidate = make_candidate('irp', params, f'edit-{state.version}', parent.uid)
                journal.append({'event': 'physical_inference', 'kind': 'learned_control',
                    'elapsed_seconds': .002, 'gpu_count': 0, 'prompt_tokens': 3, 'completion_tokens': 2})
                return DraftResult(candidate, 2)

        return bridge, Drafter()

    monkeypatch.setattr(comparison, 'ScienceExecutor', Executor)
    monkeypatch.setattr(comparison, '_load_roles', load)
    monkeypatch.setattr(comparison, '_roles', roles)
    return config, calls, role_loads, role_methods


def test_all_methods_freeze_before_common_holdout_and_select_native_utility(instrumented):
    config, calls, role_loads, role_methods = instrumented
    result = comparison.run(config)
    assert result['status'] == 'complete'
    assert result['equal_budget_comparison']
    assert not result['sota_established']
    assert result['paper_named_baselines_missing'] == ['Alg.-Only', 'Direct-LLM', 'Embed-GP', 'Janus']
    assert len(role_loads) == 1
    assert {name for name, _ in role_methods} == {'search:best_of_n', 'search:random_search', 'search:specrsi'}
    assert len({snapshot for _, snapshot in role_methods}) == 1
    assert result['physical_bill']['native_cost_ms'] == len(calls) * 3
    assert result['physical_bill']['native_calls'] == len(calls)
    assert result['physical_bill']['completion_tokens'] > 0
    for method, report in result['search'].items():
        assert report['initial_acquisition_ms'] == 3
        assert report['search_cost_ms'] <= config['search']['budget_ms']
        assert report['initial_native_utility'] > 1000  # Never VATS's initial gain (=0).
        assert result['heldout'][method]['candidate_id'] == report['selected_id']
    for method in ('best_of_n', 'random_search'):
        assert result['search'][method]['selection_native_gain'] > 0
        assert result['heldout'][method]['native_utility'] > result['heldout']['initial']['native_utility']


def test_completed_run_resumes_without_loading_roles_or_spending(instrumented):
    config, calls, role_loads, _ = instrumented
    original = comparison.run(config)
    lengths = len(calls), len(role_loads)
    assert comparison.run(config) == original
    assert (len(calls), len(role_loads)) == lengths
    changed = {**config, 'seed': config['seed'] + 1}
    with pytest.raises(ValueError, match='Resume refused'):
        comparison.run(changed)
    assert (len(calls), len(role_loads)) == lengths


@pytest.mark.parametrize('method', ['best_of_n', 'random_search'])
def test_full_evaluation_control_reserves_draft_and_verification_attempts(instrumented, method):
    config, calls, _, _ = instrumented
    config['methods'] = [method]
    config['search']['vats']['max_attempts'] = 3
    result = comparison.run(config, stage='search')
    assert result['status'] == 'search_complete'
    report = comparison._read(Path(config['output']) / 'methods' / method / 'search.json')
    # One draft plus one check leaves only one attempt. Another full child
    # requires two, so the control must stop instead of overspending N.
    assert report['draft_attempts'] == 1
    assert report['native_attempts'] == 1
    assert report['primitive_attempts'] == 2
    assert report['remaining_attempts'] == 1
    assert len(calls) == 2  # One initial-root acquisition and one child check.


@pytest.mark.parametrize('method', ['best_of_n', 'random_search'])
def test_failed_control_drafts_consume_attempts_and_keep_their_cost(instrumented, monkeypatch, method):
    config, calls, _, _ = instrumented
    config['methods'] = [method]
    config['search']['vats']['max_attempts'] = 3
    failed_drafts = []
    original = comparison._roles

    def roles(*args, **kwargs):
        bridge, drafter = original(*args, **kwargs)

        def fail(parent, state, cap, rng):
            failed_drafts.append(state.remaining_attempts)
            return DraftResult(None, 2, status='error', error='Invalid sampled edit')

        drafter.draft = fail
        return bridge, drafter

    monkeypatch.setattr(comparison, '_roles', roles)
    result = comparison.run(config, stage='search')
    assert result['status'] == 'search_complete'
    report = comparison._read(Path(config['output']) / 'methods' / method / 'search.json')
    assert len(failed_drafts) == 2
    assert report['primitive_attempts'] == report['draft_attempts'] == 2
    assert report['native_attempts'] == 0
    assert report['search_cost_ms'] == 4
    assert report['selection_native_gain'] == 0
    assert len(calls) == 1  # Failed edits acquire no child evidence.


def test_interrupted_search_keeps_bill_and_never_claims_equal_budget(instrumented, monkeypatch):
    config, calls, _, _ = instrumented
    original = comparison._search_stage
    raised = False

    def interrupted(method, *args, **kwargs):
        nonlocal raised
        result = original(method, *args, **kwargs)
        if method == 'best_of_n' and not raised:
            raised = True
            raise RuntimeError('simulated worker loss after paid work')
        return result

    monkeypatch.setattr(comparison, '_search_stage', interrupted)
    with pytest.raises(RuntimeError, match='worker loss'):
        comparison.run(config)
    assert not any(split == 'selection' for _, _, split in calls)
    before = len(calls)
    result = comparison.run(config)
    assert len(calls) > before
    assert result['search']['initial']['attempt_count'] == 1
    assert result['search']['best_of_n']['attempt_count'] == 2
    assert not result['equal_budget_comparison']
    assert result['physical_bill']['abandoned_search_attempts'] == 1
    assert not result['physical_bill']['physical_bill_complete']
    assert result['physical_bill']['native_cost_ms'] == len(calls) * 3


def test_heldout_resume_uses_frozen_search_and_completed_receipts(instrumented, monkeypatch):
    config, calls, role_loads, _ = instrumented
    original = comparison._heldout_stage
    raised = False

    def interrupted(method, *args, **kwargs):
        nonlocal raised
        if method == 'best_of_n' and not raised:
            raised = True
            raise RuntimeError('between heldout methods')
        return original(method, *args, **kwargs)

    monkeypatch.setattr(comparison, '_heldout_stage', interrupted)
    with pytest.raises(RuntimeError, match='between heldout'):
        comparison.run(config)
    search_calls = sum(split == 'dev' for _, _, split in calls)
    initial_holdout = calls[-1]
    result = comparison.run(config)
    assert result['status'] == 'complete'
    assert sum(split == 'dev' for _, _, split in calls) == search_calls
    assert calls.count(initial_holdout) == 1
    assert len(role_loads) == 1


def test_checkpoint_change_refuses_resume_before_work(instrumented):
    config, calls, _, _ = instrumented
    comparison.run(config)
    before = len(calls)
    from pathlib import Path
    (Path(config['search']['role_checkpoint']) / 'heads.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Resume refused'):
        comparison.run(config)
    assert len(calls) == before


@pytest.mark.parametrize('mutation', [
    {'heldout_split': 'audit'}, {'search_cases': [4, 4]},
    {'training_cases': [4]}, {'methods': ['Janus']}, {'max_stage_attempts': 0},
])
def test_rejects_unregistered_or_leaky_contracts_before_work(instrumented, mutation):
    config, calls, _, _ = instrumented
    with pytest.raises(ValueError):
        comparison.run({**config, **mutation})
    assert not calls


@pytest.mark.parametrize('domain', ['reactor', 'irp'])
def test_real_cpu_initial_complete_native_receipts(tmp_path, domain, monkeypatch):
    config = config_for(tmp_path, domain)
    config['methods'] = ['initial']
    config['search']['execution_cap_ms'] = 30000
    monkeypatch.setattr(comparison, '_source_hashes', lambda: {'cpu-native-registry': 'fixed'})
    result = comparison.run(config)
    assert result['status'] == 'complete'
    assert result['heldout']['initial']['admissible']
    assert result['physical_bill']['native_calls'] == 2
    assert result['physical_bill']['native_cost_ms'] > 0
    assert result['heldout']['initial']['items'][0]['receipt']['native_result']['domain'] == domain


def test_global_freeze_barrier_blocks_all_holdout_until_every_comparison_is_frozen(instrumented, tmp_path):
    config, calls, role_loads, _ = instrumented
    search = comparison.run(config, stage='search')
    assert search['status'] == 'search_complete'
    assert not any(split == 'selection' for _, _, split in calls)
    output = tmp_path / 'comparison'
    assert not (output / 'summary.json').exists()
    assert not (output / 'heldout').exists()
    barrier = tmp_path / 'barrier.json'
    body = {'comparisons': [
        {'output': str(output), 'frozen_candidates_digest': search['frozen_candidates_digest']},
        {'output': str(tmp_path / 'unfrozen'), 'frozen_candidates_digest': 'missing'},
    ]}
    comparison._write(barrier, body)
    before = len(calls)
    with pytest.raises(ValueError, match='missing or changed'):
        comparison.run(config, stage='heldout', expected_freeze_digest=search['frozen_candidates_digest'],
            freeze_barrier=barrier, freeze_barrier_sha256=comparison._hash_file(barrier))
    assert len(calls) == before
    body['comparisons'].pop()
    comparison._write(barrier, body)
    result = comparison.run(config, stage='heldout', expected_freeze_digest=search['frozen_candidates_digest'],
        freeze_barrier=barrier, freeze_barrier_sha256=comparison._hash_file(barrier))
    assert result['status'] == 'complete'
    assert len(role_loads) == 1
    assert (output / 'heldout_authorization.json').exists()


def test_heldout_never_trains_or_searches_to_fill_a_missing_stage(instrumented):
    config, calls, role_loads, _ = instrumented
    with pytest.raises(ValueError, match='missing search stages'):
        comparison.run(config, stage='heldout', expected_freeze_digest='not-yet-frozen')
    assert not calls and not role_loads


def test_native_crash_receipt_retains_unknown_work(tmp_path, monkeypatch):
    class Broken:
        def run(self, *args, **kwargs):
            raise RuntimeError('simulator died without a meter')
    monkeypatch.setattr(comparison, 'ScienceExecutor', Broken)
    journal = comparison.Journal(tmp_path / 'physical_calls.jsonl', 'unit-test')
    scope = comparison.make_scope('irp', 'development', (4,), native_utility=True)
    with pytest.raises(RuntimeError, match='without a meter'):
        comparison.RecordingExecutor(journal).run(comparison.make_candidate('irp'), scope.tasks[0], scope, 10)
    bill = comparison._bill(tmp_path)
    assert bill['unfinished_native_calls'] == 1
    assert not bill['physical_bill_complete']


def test_learned_vats_observes_complete_invalid_native_results_without_scoring_them():
    from specrsi.core.budget import BudgetLedger
    from specrsi.core.vats import VATSState
    from specrsi.learning.paper_bridge import VATSSearchBridge
    scope = comparison.make_scope('irp', 'development', (4,), native_utility=True)
    root = comparison.make_candidate('irp')
    invalid = comparison.make_candidate('irp', name='violates-native-constraint', parent_id=root.uid)
    ledger = comparison.make_science_ledger()
    task = scope.tasks[0]
    identity = task.metadata['evaluator_identity']
    ledger.admit_direct(root, task, scope, Outcome(.5,
        {'native_score': 123., 'valid': True, 'evaluator_identity': identity}), {})
    ledger.admit_direct(invalid, task, scope, Outcome(0.,
        {'native_score': None, 'valid': False, 'evaluator_identity': identity}, admissible=False), {})
    state = VATSState(scope, root, {root.uid: root, invalid.uid: invalid}, ledger, BudgetLedger(500))
    bridge = object.__new__(VATSSearchBridge)
    bridge.model = SimpleNamespace(backend='hf', config=SimpleNamespace(architecture_version=2))
    context = json.loads(bridge.context_text(state, bounded=False))
    bad = next(row for row in context['candidates'] if row['id'] == invalid.uid[:12])
    assert bad['missing'] == [] and bad['complete_score'] is None
    assert any(row['admissible'] is False for row in context['observed_outcomes'])
    nodes, _, _ = bridge._graph_state(state, ())
    bad_index = sorted(state.candidates).index(invalid.uid)
    assert nodes[bad_index][4] == 1.  # Complete, with a separate rejected mask.
    assert nodes[bad_index][5:7] == (0., 0.)


def test_eval_policy_honors_registered_sampling_law(instrumented):
    config, _, _, _ = instrumented
    config['search'].update(exploration=0., temperature=.7, mode='deploy')
    options = comparison._options(config, 'fixed-snapshot')
    assert options.exploration == 0. and options.temperature == .7 and options.mode == 'deploy'


def test_interrupted_heldout_call_is_sealed_without_reexecution(instrumented, monkeypatch, tmp_path):
    config, calls, _, _ = instrumented
    comparison.run(config, stage='search')
    original = comparison.RecordingExecutor.run
    failed_calls = []

    def interrupted(self, candidate, task, scope, cap, resume_token=None):
        if scope.split == 'selection' and not failed_calls:
            failed_calls.append((candidate.uid, task.uid))
            self.journal.append({'event': 'start', 'kind': 'native_execution', 'call_id': 'killed',
                'candidate': candidate.uid, 'task': task.uid, 'scope': scope.uid, 'cap_ms': cap})
            raise KeyboardInterrupt('simulated abrupt process death')
        return original(self, candidate, task, scope, cap, resume_token)

    monkeypatch.setattr(comparison.RecordingExecutor, 'run', interrupted)
    with pytest.raises(KeyboardInterrupt):
        comparison.run(config)
    result = comparison.run(config)
    assert result['status'] == 'incomplete'
    assert result['heldout']['initial']['items'][0]['actual_cost_unknown']
    assert 'without rerun' in result['heldout']['initial']['items'][0]['error']
    assert failed_calls[0] not in [(candidate, task) for candidate, task, _ in calls]
    assert result['physical_bill']['unfinished_native_calls'] == 1


def test_completed_heldout_journal_is_recovered_without_reexecution(instrumented, monkeypatch):
    config, calls, _, _ = instrumented
    comparison.run(config, stage='search')
    original = comparison._write
    failed = []

    def die_before_task_receipt(path, value):
        if '/heldout/' in str(path) and str(path).endswith('.json') and not failed:
            failed.append(str(path))
            raise KeyboardInterrupt('after native receipt before item save')
        return original(path, value)

    monkeypatch.setattr(comparison, '_write', die_before_task_receipt)
    with pytest.raises(KeyboardInterrupt):
        comparison.run(config)
    previous_heldout = [row for row in calls if row[2] == 'selection']
    assert len(previous_heldout) == 1
    result = comparison.run(config)
    assert result['status'] == 'complete'
    assert calls.count(previous_heldout[0]) == 1


@pytest.mark.parametrize('method,event', [
    ('janus_science', 'janus_real_promotion'), ('hgm_science', 'clade_expansion'),
    ('aflow_science', 'aflow_expand_evaluate_update')])
def test_actual_named_algorithms_share_scope_meter_and_native_final_selection(instrumented, named_service, method, event):
    config, calls, role_loads, role_methods = instrumented
    add_named_config(config, [method])
    result = comparison.run(config)
    assert result['status'] == 'complete' and result['equal_budget_comparison']
    report = result['search'][method]
    assert any(row['event'] == event for row in report['algorithm_report']['events'])
    assert report['implementation'] == 'prospective_scientific_parameter_adaptation'
    assert report['initial_acquisition_ms'] == 3
    assert report['primitive_attempts'] == report['native_attempts'] + report['draft_diagnosis_proxy_attempts']
    assert report['primitive_attempts'] <= config['search']['vats']['max_attempts']
    assert report['search_cost_ms'] == sum(charge['actual'] for charge in report['charges'])
    assert report['search_cost_ms'] <= config['search']['budget_ms']
    assert report['selected_native_utility'] >= report['initial_native_utility'] > 1000
    assert result['heldout'][method]['candidate_id'] == report['selected_id']
    assert not role_loads and not role_methods  # Named adapter uses its registered model service.
    assert named_service and all(row[:2] == ('http://127.0.0.1:18085', 'Qwen3.5-35B-A3B') for row in named_service)
    assert result['physical_bill']['native_calls'] == len(calls)
    assert result['physical_bill']['completion_tokens'] == len(named_service) * 5
    assert result['named_science_adaptations'] == [method]
    assert {'emas', 'flash_evolve', 'metarsi'} <= set(result['unavailable_scientific_methods'])
    # The fixture deliberately gives higher native utility LOWER bounded score.
    # Final adoption still takes the maximum actual native receipt across all
    # complete candidates, independent of the algorithm's allocation statistic.
    journal = Path(config['output']) / 'methods' / method / 'attempt_001' / 'physical_calls.jsonl'
    observations = [json.loads(line) for line in journal.read_text().splitlines()]
    native = [row['outcome']['fields']['native_score'] for row in observations
              if row.get('kind') == 'native_execution' and row['event'] == 'finish']
    assert report['selected_native_utility'] == max(native)


def test_named_initial_read_does_not_rebill_but_aflow_repeats_are_paid(instrumented, named_service):
    config, calls, _, _ = instrumented
    add_named_config(config, ['aflow_science'])
    config['named_baselines']['aflow_science'] = {'rounds': 1, 'validation_repeats': 3}
    result = comparison.run(config, stage='search')
    assert result['status'] == 'search_complete'
    report = comparison._read(Path(config['output']) / 'methods' / 'aflow_science' / 'search.json')
    assert report['native_attempts'] == 2  # First read uses the paid root, two subsequent repeats execute.
    assert report['primitive_attempts'] == 2 and report['search_cost_ms'] == 6
    assert report['initial_acquisition_ms'] == 3 and len(calls) == 3
    assert not named_service


def test_named_proxy_and_draft_generation_share_attempt_allowance(instrumented, named_service):
    config, _, _, _ = instrumented
    add_named_config(config, ['janus_science'])
    config['search']['vats']['max_attempts'] = 1
    result = comparison.run(config, stage='search')
    report = comparison._read(Path(config['output']) / 'methods' / 'janus_science' / 'search.json')
    assert result['equal_budget_comparison']
    assert report['primitive_attempts'] == 1 and report['native_attempts'] == 0
    assert len(named_service) == 1 and report['selection_native_gain'] == 0


def test_shared_wall_admission_stops_every_search_before_unaffordable_work(instrumented, named_service):
    config, calls, _, _ = instrumented
    add_named_config(config, list(comparison.SUPPORTED_METHODS))
    # Model/native slices are 5/10 ms. An effectively zero wall allowance must
    # stop all physical search work while preserving initial/heldout accounting.
    config['search']['wall_limit_seconds'] = .0000001
    with pytest.raises(ValueError, match='no initial nonempty work opportunity'):
        comparison.run(config, stage='search')
    assert not calls  # Impossible registrations fail before initial acquisition.
    assert not named_service


@pytest.mark.parametrize('change', ['fp8', 'other_port', 'missing_wall'])
def test_named_registration_rejects_unmatched_model_or_wall_contract_before_work(instrumented, named_service, change):
    config, calls, role_loads, _ = instrumented
    add_named_config(config, ['janus_science'])
    if change == 'fp8':
        config['named_baselines']['model']['name'] += '-FP8'
    elif change == 'other_port':
        config['named_baselines']['model']['base_url'] = 'http://127.0.0.1:18080'
    else:
        del config['search']['wall_limit_seconds']
    with pytest.raises(ValueError, match='Named'):
        comparison.run(config, stage='search')
    assert not calls and not role_loads and not named_service


def test_unimplemented_graph_baseline_cannot_be_silently_renamed(instrumented, named_service):
    config, calls, role_loads, _ = instrumented
    config['methods'] = ['emas']
    with pytest.raises(ValueError, match='Unknown'):
        comparison.run(config, stage='search')
    assert not calls and not role_loads and not named_service


def test_canonical_paper33_caps_admit_real_work_within_registered_common_wall():
    import yaml
    template = yaml.safe_load((comparison.ROOT / 'configs/training/qwen35_paper33_science.yaml').read_text())
    config = {'seed': 0, 'search_cases': [4, 5], 'search': {
        **template['collection'], 'wall_limit_seconds': 1200}}
    receipt = comparison.validate_comparison_resources(config, comparison.SUPPORTED_METHODS)
    assert receipt['specrsi']['first_nonempty_wall_cap_ms'] < 1200000
    assert all(row['first_nonempty_expense_cap_ms'] <= config['search']['budget_ms'] for row in receipt.values())
    # Regression for the rejected preflight: 32*60s scorer queries + 60s policy
    # makes preparation alone exceed the entire 1200s comparison allowance.
    config['search']['head_cap_ms'] = 60000
    config['search']['vats'] = {**config['search']['vats'], 'max_queries': 32, 'policy_call_cost': 60000}
    with pytest.raises(ValueError, match='specrsi.*no initial nonempty work'):
        comparison.validate_comparison_resources(config, comparison.SUPPORTED_METHODS)


def test_named_native_blocks_use_the_same_registered_worker_count(instrumented, named_service, monkeypatch):
    import threading
    import time
    config, _, _, _ = instrumented
    add_named_config(config, ['aflow_science'])
    config['search_cases'] = [4, 5]
    config['search']['vats']['workers'] = 2
    config['named_baselines']['aflow_science'] = {'rounds': 1, 'validation_repeats': 2}
    original = comparison.ScienceExecutor.run
    lock = threading.Lock()
    active = peak = 0

    def observed(self, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(.01)
            return original(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(comparison.ScienceExecutor, 'run', observed)
    comparison.run(config, stage='search')
    report = comparison._read(Path(config['output']) / 'methods' / 'aflow_science' / 'search.json')
    assert report['native_workers'] == peak == 2
    assert report['native_attempts'] == 2 and report['search_cost_ms'] == 6


def test_actual_wall_overrun_is_preserved_and_invalidates_equal_budget_claim(instrumented, monkeypatch):
    config, _, _, _ = instrumented
    config['methods'] = ['best_of_n']
    config['search']['wall_limit_seconds'] = 600
    clock = [0.]
    monkeypatch.setattr(comparison.time, 'monotonic', lambda: clock[0])
    original = comparison.ScienceExecutor.run

    def advance(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += 601.
        return result

    monkeypatch.setattr(comparison.ScienceExecutor, 'run', advance)
    result = comparison.run(config, stage='search')
    report = comparison._read(Path(config['output']) / 'methods' / 'best_of_n' / 'search.json')
    assert report['metered_search_wall_seconds'] == 601.
    assert not report['wall_limit_respected']
    assert not result['equal_budget_comparison']


def test_named_parameter_catalog_uses_registered_common_size(instrumented, named_service):
    config, _, _, _ = instrumented
    add_named_config(config, ['hgm_science'])
    config['search']['catalog_size'] = 8
    comparison.run(config, stage='search')
    proposal_prompts = [row[2][-1]['content'] for row in named_service if '"choices"' in row[2][-1]['content']]
    assert proposal_prompts
    assert all(len(json.loads(prompt.split('\n', 1)[1])['choices']) == 8 for prompt in proposal_prompts)
