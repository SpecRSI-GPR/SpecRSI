"""End-to-end prospective protocol checks; all model outputs here are test fixtures."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from specrsi.benchmarks import BenchmarkExample
from specrsi.core import Candidate
from specrsi.core.search import DraftResult
from specrsi.models import ModelReply

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('registered_runner', ROOT / 'scripts/run_registered_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [0] * 4

    def encode(self, *args, **kwargs):
        return [1] * 8


class Roles:
    backend = 'hf'
    device = 'cpu'
    config = SimpleNamespace(max_length=64, option_dim=12, feature_dim=8)
    tokenizer = Tokenizer()

    def decision_values(self, record):
        return torch.zeros(len(record.support))

    def decision_logits(self, record):
        return torch.tensor([2 * row[2] + row[1] for row in record.option_features])


class Draft:
    def __init__(self, bridge):
        self.bridge = bridge

    def draft(self, parent, state, cap, rng):
        self.bridge.physical_inferences.append({'role': 'draft', 'prompt_tokens': 8, 'completion_tokens': 2,
                                               'elapsed_seconds': .01, 'gpu_count': 0})
        return DraftResult(Candidate('improved', {'system_prompt': 'improved'}, parent.runtime,
                                     'opaque', False, parent_id=parent.uid), 10)


def fixture(tmp_path, monkeypatch, *, replicas=2, search_calls=30):
    lock = tmp_path / 'model.lock.json'
    lock.write_text(json.dumps({'revision': 'pinned', 'local_path': '/fake/base'}))
    checkpoints = []
    for seed in range(replicas):
        checkpoint = tmp_path / f'training-{seed}' / 'checkpoint'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'model_config.json').write_text(json.dumps({'backend': 'hf', 'architecture_version': 2,
            'option_dim': 12, 'feature_dim': 8, 'model_path': '/fake/base', 'revision': 'pinned'}))
        (checkpoint / 'heads.pt').write_bytes(f'fixture-weights-{seed}'.encode())
        (checkpoint / 'adapters').mkdir()
        (checkpoint / 'training_config.json').write_text(json.dumps({'seed': seed}))
        (checkpoint.parent / 'resolved_config.json').write_text(json.dumps({'training': {'seed': seed}}))
        (checkpoint.parent / 'metrics.json').write_text(json.dumps({'algorithm': 'vats_pdf', 'synthetic_demo': False,
            'source_changed_during_run': False, 'history': [{'fixture_only': True}]}))
        checkpoints.append({'seed': seed, 'checkpoint': str(checkpoint)})
    config = {'dataset': 'hotpotqa', 'data_root': str(tmp_path), 'selection_limit': 1, 'audit_limit': 1,
        'initial_prompt': 'initial', 'resource_profile': 'fake_cpu_test_only', 'model': {'lock': str(lock),
        'base_url': 'http://fake', 'max_tokens': 4}, 'role_device': 'cpu', 'training_replicas': checkpoints,
        'search_seeds': [0], 'methods': list(runner.METHODS), 'bootstrap_resamples': 1000,
        'budget': {'search_tokens': 300, 'search_calls': search_calls, 'execution_cap': 16, 'draft_cap': 16},
        'search': {'max_actions': 3, 'max_candidates': 2, 'max_children': 2, 'vats': {'max_queries': 2,
        'max_extensions': 0, 'max_draft_depth': 1, 'max_draft_parents': 1, 'max_completion_targets': 1,
        'max_pair_targets': 0, 'max_singletons': 1, 'max_menu_options': 4}}}
    output = tmp_path / 'campaign'
    physical = []

    class Client:
        def generate(self, messages, **kwargs):
            physical.append(messages)
            if messages[-1]['content'] == 'audit question':
                frozen = output / 'all_successors_frozen.json'
                assert frozen.is_file(), 'All methods must finish optimization before any audit call'
                assert len(json.loads(frozen.read_text())['selected']) == len(checkpoints) * len(config['methods'])
            text = 'Paris' if messages[0]['content'] == 'improved' else 'wrong'
            return ModelReply(text, 4, 2, .01, 'stop', 'pinned', f'test-{len(physical)}')

    def examples(dataset, split, **kwargs):
        return [BenchmarkExample(split, dataset, split, split + ' question', 'Paris')]

    monkeypatch.setattr(runner, 'load_examples', examples)
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'fixture': 'immutable'})
    monkeypatch.setattr(runner, 'load_backbone', lambda config: (Client(), Tokenizer(),
                        json.loads(lock.read_text()), {'revision': 'pinned', 'fixture_only': True}))
    monkeypatch.setattr(runner, 'load_roles', lambda *args: Roles())
    monkeypatch.setattr(runner, 'make_drafter', lambda bridge, settings, validator: Draft(bridge))
    return config, output, physical


def test_staged_search_freezes_all_arms_and_audit_resumes_without_roles_or_search(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    runner.register(config, output)
    result = runner.execute(output, phase='search-only')
    assert result['heldout_model_calls'] == 0 and result['frozen_candidates'] == 8
    assert all(messages[-1]['content'] != 'audit question' for messages in physical)
    assert not list(output.glob('runs/*/audit.json'))
    assert len(list(output.glob('runs/*/selection_frozen.json'))) == 8
    count = len(physical)
    monkeypatch.setattr(runner, 'load_roles', lambda *args: pytest.fail('Audit cannot load/train roles'))
    monkeypatch.setattr(runner, 'optimize_arm', lambda *args: pytest.fail('Audit cannot repeat selection'))
    for checksum in (None, '0' * 64):
        with pytest.raises(ValueError, match='caller-held'):
            runner.execute(output, phase='audit-only', expected_frozen_sha256=checksum)
    assert len(physical) == count
    runner.execute(output, phase='audit-only', expected_frozen_sha256=result['selection_frozen_sha256'])
    assert len(physical) == count + 8
    assert all(m[-1]['content'] == 'audit question' for m in physical[count:])
    assert len(list(output.glob('runs/*/run.json'))) == 8
    with pytest.raises(FileExistsError):
        runner.execute(output, phase='audit-only', expected_frozen_sha256=result['selection_frozen_sha256'])
    assert len(physical) == count + 8


def test_registered_output_contract_is_identical_for_all_four_methods_and_all_phases(tmp_path, monkeypatch):
    from specrsi.output_contracts import HOTPOT_SHORT_ANSWER_V1
    config, output, physical = fixture(tmp_path, monkeypatch)
    config['model']['output_contract'] = 'hotpot_short_answer_v1'
    runner.register(config, output)
    frozen = runner.execute(output, phase='search-only')
    search_calls = len(physical)
    assert search_calls > 8 and all('audit question' != m[-1]['content'] for m in physical)
    runner.execute(output, phase='audit-only', expected_frozen_sha256=frozen['selection_frozen_sha256'])
    assert len(physical) == search_calls + 8
    assert all(messages[0]['content'].endswith(HOTPOT_SHORT_ANSWER_V1) for messages in physical)
    selected = [json.loads(path.read_text()) for path in output.glob('runs/*/selected.json')]
    assert len(selected) == 8
    assert all(row['runtime']['generation']['output_contract'] == 'hotpot_short_answer_v1' for row in selected)


@pytest.mark.parametrize('filename', ['selected.json', 'selection_frozen.json', 'initial.json',
                                      'search.json', 'physical_calls.jsonl', 'physical_calls_operations.jsonl'])
def test_staged_audit_refuses_changed_frozen_artifacts_before_inference(tmp_path, monkeypatch, filename):
    config, output, physical = fixture(tmp_path, monkeypatch)
    runner.register(config, output)
    frozen = runner.execute(output, phase='search-only')['selection_frozen_sha256']
    count = len(physical)
    path = next(output.glob('runs/*/' + filename))
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        runner.execute(output, phase='audit-only', expected_frozen_sha256=frozen)
    assert len(physical) == count


def cross_barrier(output, config, tmp_path):
    external = tmp_path / 'native-run'
    external.mkdir()
    (external / 'selected.json').write_text('{"system_prompt":"initial"}')
    (external / 'selection_frozen.json').write_text('{"test_fixture":true}')
    entries = {}
    for run_id in config['comparison']['planned_run_ids']:
        directory = external if run_id == 'native-test' else output / 'runs' / run_id
        entries[run_id] = {key: {'path': str((directory / (key + '.json')).resolve()),
                                'sha256': runner.file_hash(directory / (key + '.json'))}
                           for key in ('selection_frozen', 'selected')}
    barrier = tmp_path / 'cross_method_barrier.json'
    barrier.write_text(json.dumps({'schema': 'cross_method_selection_barrier_v1',
        'planned_run_ids': config['comparison']['planned_run_ids'], 'runs': entries}))
    return barrier


def test_staged_cross_method_audit_requires_and_rechecks_complete_barrier(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    config['comparison'] = {'planned_run_ids': [
        f"seed-{r['seed']}-search-0-{m}" for r in config['training_replicas'] for m in config['methods']
    ] + ['native-test']}
    runner.register(config, output)
    with pytest.raises(ValueError, match='cross-method|Cross-method'):
        runner.execute(output)
    assert not physical
    frozen = runner.execute(output, phase='search-only')['selection_frozen_sha256']
    count = len(physical)
    with pytest.raises(ValueError, match='barrier'):
        runner.execute(output, phase='audit-only', expected_frozen_sha256=frozen)
    assert len(physical) == count
    barrier = cross_barrier(output, config, tmp_path)
    value = json.loads(barrier.read_text())
    missing = copy.deepcopy(value)
    missing['runs'].pop('native-test')
    barrier.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match='exact registered run set'):
        runner.verify_external_barrier(config, output, path=barrier, expected_sha256=runner.file_hash(barrier))
    duplicate = copy.deepcopy(value)
    duplicate['runs']['native-test'] = copy.deepcopy(next(iter(value['runs'].values())))
    barrier.write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match='distinct frozen artifact'):
        runner.verify_external_barrier(config, output, path=barrier, expected_sha256=runner.file_hash(barrier))
    barrier.write_text(json.dumps(value))
    runner.execute(output, phase='audit-only', expected_frozen_sha256=frozen,
                   barrier_path=barrier, barrier_sha256=runner.file_hash(barrier))
    assert len(physical) == count + 8


@pytest.mark.parametrize('corrupt', ['before', 'settlement', 'receipt', 'pending'])
def test_search_meter_resume_checks_journal_against_paid_receipts(tmp_path, corrupt):
    caps = {phase: {'tokens': 100, 'calls': 20} for phase in ('initial', 'search', 'audit')}
    meter = runner.PhysicalMeter(caps, tmp_path / 'physical_calls.jsonl')
    meter.begin(10, 1, 'task')
    meter.charge([{'role': 'task', 'prompt_tokens': 4, 'completion_tokens': 2}], 10)
    meter.settle()
    frozen = {'phase_caps': caps, 'phase_receipts': meter.spent, 'operation_count': 1,
        'search_receipts_sha256': runner.file_hash(meter.path)}
    operations = [json.loads(line) for line in meter.journal_path.read_text().splitlines()]
    if corrupt == 'before':
        operations[0]['before_tokens'] = 1
    elif corrupt == 'settlement':
        operations[1]['role'] = 'draft'
    elif corrupt == 'receipt':
        operations[1]['measured_tokens'] = 5
    else:
        operations = operations[:1]
    meter.journal_path.write_text('\n'.join(json.dumps(r) for r in operations) + '\n')
    frozen['search_operations_sha256'] = runner.file_hash(meter.journal_path)
    with pytest.raises(ValueError):
        runner.PhysicalMeter.restore_search(caps, meter.path, frozen)


def test_cached_role_keeps_head_call_and_zero_encoder_token_bill(tmp_path):
    meter = runner.PhysicalMeter({'initial': {'tokens': 8, 'calls': 2}}, tmp_path / 'calls.jsonl')
    receipt = {'role': 'policy', 'prompt_tokens': 0, 'completion_tokens': 0,
        'encoder_forward_count': 0, 'head_forward_count': 1, 'encoder_cache_hit': True,
        'encoder_snapshot': 'immutable-test-base', 'context_input_token_ids': [1, 2, 3]}
    meter.charge([receipt], 8)
    assert meter.spent['initial'] == {'tokens': 0, 'calls': 1}
    invalid = {**receipt, 'prompt_tokens': 3}
    with pytest.raises(RuntimeError, match='skipped encoder'):
        meter.charge([invalid], 8)
    assert not meter.valid
    assert meter.spent['initial'] == {'tokens': 3, 'calls': 2}  # Preserve claimed costs on invalid output.


def test_resume_reads_physical_jsonl_lines_with_unicode_answer_separators(tmp_path):
    caps = {phase: {'tokens': 100, 'calls': 10} for phase in ('initial', 'search', 'audit')}
    meter = runner.PhysicalMeter(caps, tmp_path / 'calls.jsonl')
    meter.begin(10, 1, 'task')
    meter.charge([{'role': 'task', 'prompt_tokens': 4, 'completion_tokens': 2,
                   'text': 'observed\u2028answer\u2029tail\u0085retained'}], 10)
    meter.settle()
    frozen = {'phase_caps': caps, 'phase_receipts': meter.spent, 'operation_count': 1,
        'search_receipts_sha256': runner.file_hash(meter.path),
        'search_operations_sha256': runner.file_hash(meter.journal_path)}
    restored = runner.PhysicalMeter.restore_search(caps, meter.path, frozen)
    assert restored.records == meter.records and restored.spent == meter.spent


@pytest.mark.parametrize('filename', ['initial.json', 'search.json', 'selection_frozen.json',
                                      'physical_calls.jsonl', 'physical_calls_operations.jsonl'])
def test_audit_cannot_publish_if_frozen_search_evidence_changes_during_scoring(tmp_path, monkeypatch, filename):
    config, output, physical = fixture(tmp_path, monkeypatch)
    runner.register(config, output)
    original = runner.load_backbone
    changed = False

    def backbone(config):
        client, *rest = original(config)
        generate = client.generate

        def wrapped(messages, **kwargs):
            nonlocal changed
            result = generate(messages, **kwargs)
            if messages[-1]['content'] == 'audit question' and not changed:
                path = next(output.glob('runs/*/' + filename))
                path.write_text(' ' + path.read_text())
                changed = True
            return result
        client.generate = wrapped
        return client, *rest

    monkeypatch.setattr(runner, 'load_backbone', backbone)
    with pytest.raises(ValueError, match='changed during audit'):
        runner.execute(output)
    assert changed and physical and not (output / 'comparison.json').exists()
    assert (output / 'failure.json').is_file()


def test_registered_runner_produces_verified_paired_records_and_meters_every_arm(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    registered = runner.register(config, output)
    assert registered['paid_model_calls'] == 0 and not physical
    result = runner.execute(output)
    assert result['completed_runs'] == 8
    assert not result['all_registered_comparisons_supported']  # Two seeds are not proof of superiority.
    records = [json.loads(line) for line in (output / 'runs.jsonl').open()]
    assert {row['method'] for row in records} == set(runner.METHODS)
    assert len({row['contract']['execution_contract_sha256'] for row in records}) == 1
    search_costs = {}
    for record in records:
        directory = output / 'runs' / record['run_id']
        raw = json.loads((directory / 'run.json').read_text())
        calls = [json.loads(line) for line in (directory / 'physical_calls.jsonl').open()]
        assert raw['physical_tokens'] == sum(row['prompt_tokens'] + row['completion_tokens'] for row in calls)
        assert raw['actual_model_calls'] == len(calls)
        assert raw['phase_receipts']['initial'] == {'tokens': 6, 'calls': 1}
        assert raw['phase_receipts']['audit'] == {'tokens': 6, 'calls': 1}
        assert all(row['cost_kind'] == 'actual' for row in calls)
        assert raw['physical_calls_sha256'] == runner.file_hash(directory / 'physical_calls.jsonl')
        search_costs[record['method']] = raw['phase_receipts']['search']['tokens']
    assert search_costs['initial'] == 0
    assert search_costs['specrsi'] > search_costs['best_of_n'] == search_costs['random_search'] == 16
    assert len(json.loads((output / 'comparison.json').read_text())['comparisons']) == 3
    with pytest.raises(FileExistsError):
        runner.execute(output)


def test_hard_physical_call_cap_stops_before_inference_and_preserves_audit_reserve(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch, search_calls=1)
    runner.register(config, output)
    runner.execute(output)
    for directory in (output / 'runs').iterdir():
        raw = json.loads((directory / 'run.json').read_text())
        assert raw['phase_receipts']['search']['calls'] <= 1
        assert raw['phase_receipts']['audit'] == {'tokens': 6, 'calls': 1}
        assert raw['audit_mean'] == 0  # No optimizer has both proposed and fully checked a successor.
        if not directory.name.endswith('-initial'):
            assert json.loads((directory / 'search.json').read_text())['stop_reason'] == 'physical_budget_exhausted_before_call'


def test_single_training_seed_requires_disclosed_engineering_mode(tmp_path, monkeypatch):
    config, output, _ = fixture(tmp_path, monkeypatch, replicas=1)
    with pytest.raises(ValueError, match='independently trained'):
        runner.register(config, output)
    config['engineering_smoke_only'] = True
    runner.register(config, output)
    result = runner.execute(output)
    assert not result['all_registered_comparisons_supported']
    assert json.loads((output / 'comparison.json').read_text())['engineering_smoke_only']


@pytest.mark.parametrize('field,value,match', [
    ('alpha', 0, 'inference settings'),
    ('bootstrap_resamples', 999, 'inference settings'),
    ('analysis_seed', -1, 'inference settings'),
    ('minimum_effect', -1, 'minimum_effect'),
    ('engineering_smoke_only', 'true', 'must be a boolean'),
])
def test_invalid_analysis_protocol_refused_before_registration_or_paid_calls(tmp_path, monkeypatch, field, value, match):
    config, output, physical = fixture(tmp_path, monkeypatch)
    config[field] = value
    with pytest.raises(ValueError, match=match):
        runner.register(config, output)
    assert not output.exists() and not physical


def test_invalid_vats_menu_refused_before_freezing_campaign_or_acquiring_root(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    config['search']['vats']['max_menu_options'] = 3
    with pytest.raises(ValueError, match='promised completion seeds'):
        runner.register(config, output)
    assert not output.exists() and not physical


@pytest.mark.parametrize('mutation,match', [
    ('protocol', 'protocol'), ('registration', 'protocol'), ('checkpoint', 'checkpoint'), ('split', 'examples'), ('source', 'source'),
])
def test_changed_registration_inputs_refuse_all_paid_calls(tmp_path, monkeypatch, mutation, match):
    config, output, physical = fixture(tmp_path, monkeypatch)
    runner.register(config, output)
    if mutation == 'protocol':
        path = output / 'protocol.json'
        value = json.loads(path.read_text())
        value['config']['budget']['search_tokens'] += 1
        path.write_text(json.dumps(value))
    elif mutation == 'registration':
        path = output / 'registration.json'
        value = json.loads(path.read_text())
        value['profiles'][0]['contract']['budget']['cap'] += 1
        path.write_text(json.dumps(value))
    elif mutation == 'checkpoint':
        (Path(config['training_replicas'][0]['checkpoint']) / 'heads.pt').write_bytes(b'changed')
    elif mutation == 'split':
        (output / 'audit_examples.json').write_text('[]')
    else:
        monkeypatch.setattr(runner, 'source_hashes', lambda: {'fixture': 'changed'})
    with pytest.raises(ValueError, match=match):
        runner.execute(output)
    assert not physical


def test_duplicate_weight_files_cannot_masquerade_as_training_replicas(tmp_path, monkeypatch):
    config, output, _ = fixture(tmp_path, monkeypatch)
    first, second = [Path(row['checkpoint']) / 'heads.pt' for row in config['training_replicas']]
    second.write_bytes(first.read_bytes())
    # Different optimizer/RNG snapshots do not make identical inference weights
    # into independent trained replicas.
    (first.parent / 'trainer.pt').write_bytes(b'optimizer-state-1')
    (second.parent / 'trainer.pt').write_bytes(b'optimizer-state-2')
    with pytest.raises(ValueError, match='Identical weights'):
        runner.register(config, output)


def test_physical_meter_retains_cost_of_failed_or_wrong_identity_reply(tmp_path):
    class BadClient:
        def generate(self, *args, **kwargs):
            return ModelReply('x', 4, 2, .1, 'timeout', 'changed-revision', 'bad-call')
    meter = runner.PhysicalMeter({'initial': {'tokens': 16, 'calls': 1}}, tmp_path / 'calls.jsonl')
    client = runner.MeteredClient(BadClient(), Tokenizer(), 'pinned', meter)
    with pytest.raises(RuntimeError, match='identity'):
        client.generate([{'role': 'user', 'content': 'x'}], max_tokens=4, total_token_cap=16)
    assert meter.spent['initial']['tokens'] == 6 and not meter.valid
    assert json.loads(meter.path.read_text())['finish_reason'] == 'timeout'


def test_offsets_select_new_frozen_examples_without_mutating_datasets(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    config.update(selection_offset=1, selection_limit=2, audit_offset=2, audit_limit=2)
    examples = {split: [BenchmarkExample(f'{split}-{i}', 'hotpotqa', split, f'{split} question {i}', 'Paris')
                       for i in range(5)] for split in ('selection', 'audit')}
    before = copy.deepcopy(examples)
    monkeypatch.setattr(runner, 'load_examples', lambda dataset, split, **kwargs: examples[split])
    runner.register(config, output)
    assert [row['id'] for row in json.loads((output / 'selection_examples.json').read_text())] == ['selection-1', 'selection-2']
    assert [row['id'] for row in json.loads((output / 'audit_examples.json').read_text())] == ['audit-2', 'audit-3']
    assert examples == before and not physical


def test_selected_artifact_mutation_during_audit_invalidates_campaign(tmp_path, monkeypatch):
    config, output, _ = fixture(tmp_path, monkeypatch)
    runner.register(config, output)
    original = runner.MeteredClient.generate
    changed = False

    def mutate_after_audit_call(self, messages, **kwargs):
        nonlocal changed
        result = original(self, messages, **kwargs)
        if self.meter.phase == 'audit' and not changed:
            selected = next((output / 'runs').glob('*/selected.json'))
            value = json.loads(selected.read_text())
            value['artifacts']['system_prompt'] = 'a different unevaluated artifact'
            selected.write_text(json.dumps(value))
            changed = True
        return result

    monkeypatch.setattr(runner.MeteredClient, 'generate', mutate_after_audit_call)
    with pytest.raises(ValueError, match='selected artifacts changed'):
        runner.execute(output)
    assert (output / 'failure.json').is_file() and not (output / 'comparison.json').exists()


def test_epoch_checkpoints_cannot_be_relabeled_as_independent_training_seeds(tmp_path, monkeypatch):
    config, output, _ = fixture(tmp_path, monkeypatch)
    checkpoint = Path(config['training_replicas'][1]['checkpoint'])
    (checkpoint / 'training_config.json').write_text(json.dumps({'seed': 0}))
    with pytest.raises(ValueError, match='actual training seed|training seed disagrees'):
        runner.register(config, output)


def two_stage_fixture(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    for replica in config['training_replicas']:
        checkpoint = Path(replica['checkpoint'])
        root, seed = checkpoint.parent, replica['seed']
        (root / 'gvo').mkdir()
        checkpoint.rename(root / 'gvo' / 'checkpoint')
        shutil.copytree(root / 'gvo' / 'checkpoint', root / 'warmup' / 'checkpoint')
        (root / 'warmup' / 'checkpoint' / 'heads.pt').write_bytes(f'warmup-{seed}'.encode())
        checkpoint.symlink_to('gvo/checkpoint', target_is_directory=True)
        stages, evidence = [], []
        for stage in ('warmup', 'gvo'):
            directory = root / stage
            resolved = {'stage': stage, 'output': str(directory), 'training': {'seed': seed}}
            if stage == 'gvo':
                resolved['resume'] = str(root / 'warmup' / 'checkpoint')
            stages.append(resolved)
            (directory / 'resolved_config.json').write_text(json.dumps(resolved))
            metrics = {'algorithm': 'vats_pdf', 'synthetic_demo': False, 'source_changed_during_run': False,
                       'source_hashes': {'test_fixture': 'same_sources'},
                       'checkpoint': str(directory / 'checkpoint'),
                       'history': [{'stage': stage, 'steps': 4 if stage == 'warmup' else 7}]}
            path = directory / 'metrics.json'
            path.write_text(json.dumps(metrics))
            evidence.append({'stage': stage, 'metrics': str(path), 'metrics_sha256': runner.file_hash(path),
                             'checkpoint': metrics['checkpoint']})
        pipeline = 'pdf_sft_then_refreshed_gvo_v1'
        (root / 'registration.json').write_text(json.dumps({'pipeline': pipeline, 'stages': stages}))
        (root / 'metrics.json').write_text(json.dumps({'pipeline': pipeline, 'complete': True,
                                                      'stages': evidence, 'checkpoint': str(checkpoint)}))
    return config, output, physical


def test_two_stage_training_chain_is_verified_and_frozen_before_comparison(tmp_path, monkeypatch):
    config, output, physical = two_stage_fixture(tmp_path, monkeypatch)
    runner.register(config, output)
    assert not physical
    protocol = json.loads((output / 'protocol.json').read_text())
    for replica in protocol['replicas']:
        provenance = replica['training_provenance']
        assert any(path.endswith('/warmup/checkpoint/heads.pt') for path in provenance)
        assert any(path.endswith('/registration.json') for path in provenance)
        assert len([path for path in provenance if path.endswith('/metrics.json')]) == 3
    warmup_heads = Path(config['training_replicas'][0]['checkpoint']).parent / 'warmup' / 'checkpoint' / 'heads.pt'
    warmup_heads.write_bytes(b'changed after registration')
    with pytest.raises(ValueError, match='checkpoint changed'):
        runner.execute(output)
    assert not physical


@pytest.mark.parametrize('mutation,match', [
    ('incomplete', 'completed pipeline'),
    ('wrong_link', 'final GVO checkpoint'),
    ('changed_source', 'sources'),
    ('wrong_resume', 'resume'),
    ('wrong_warmup_seed', 'registered replica'),
    ('no_gvo_updates', 'actual optimizer updates'),
    ('wrong_update_delta', 'optimizer updates disagree'),
])
def test_two_stage_chain_rejects_incomplete_or_relabelled_training(tmp_path, monkeypatch, mutation, match):
    config, output, physical = two_stage_fixture(tmp_path, monkeypatch)
    checkpoint = Path(config['training_replicas'][0]['checkpoint'])
    root = checkpoint.parent
    summary = json.loads((root / 'metrics.json').read_text())
    if mutation == 'incomplete':
        summary['complete'] = False
    elif mutation == 'wrong_link':
        checkpoint.unlink()
        checkpoint.symlink_to('warmup/checkpoint', target_is_directory=True)
        # Explicitly select GVO so resolving the user path cannot hide this mismatch.
        config['training_replicas'][0]['checkpoint'] = str(root / 'gvo' / 'checkpoint')
    elif mutation in {'changed_source', 'no_gvo_updates', 'wrong_update_delta'}:
        path = root / 'gvo' / 'metrics.json'
        metrics = json.loads(path.read_text())
        if mutation == 'changed_source':
            metrics['source_hashes']['test_fixture'] = 'changed_between_stages'
        elif mutation == 'no_gvo_updates':
            metrics['history'][-1]['steps'] = 4
        else:
            metrics.update(initial_optimizer_steps=4, optimizer_updates=7)
        path.write_text(json.dumps(metrics))
        summary['stages'][1]['metrics_sha256'] = runner.file_hash(path)
    else:
        stage = 'warmup' if mutation == 'wrong_warmup_seed' else 'gvo'
        path = root / stage / 'resolved_config.json'
        resolved = json.loads(path.read_text())
        if mutation == 'wrong_warmup_seed':
            resolved['training']['seed'] = 999
        else:
            resolved['resume'] = str(root / 'gvo' / 'checkpoint')
        path.write_text(json.dumps(resolved))
    (root / 'metrics.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=match):
        runner.register(config, output)
    assert not output.exists() and not physical


def prior_performance(tmp_path, *, dataset='hotpotqa', prompt='protected input', example_id='old-id'):
    path = tmp_path / 'prior' / 'registration.json'
    path.parent.mkdir(exist_ok=True)
    identity = runner.example_identities([BenchmarkExample(example_id, dataset, 'audit', prompt, 'unused')])[0]
    path.write_text(json.dumps({'config': {'schema': 'inference_performance_v1', 'dataset': dataset,
        'counts': {'audit': 1}}, 'examples': {'audit': [{'id': example_id, 'input_hash': identity['input_hash']}]}}))
    return path


def exclusion_examples():
    return {split: [BenchmarkExample(split, 'hotpotqa', split, split + ' question', 'unused')]
            for split in ('development', 'selection', 'audit')}


def test_prior_global_input_hash_filters_before_offset_without_changing_other_splits(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    source = prior_performance(tmp_path, dataset='another-dataset', prompt=' Protected   INPUT ')
    examples = exclusion_examples()
    examples['audit'] = [BenchmarkExample('new-id', 'hotpotqa', 'audit', 'protected input', 'do not consult'),
        BenchmarkExample('fresh-0', 'hotpotqa', 'audit', 'fresh input zero', 'unused'),
        BenchmarkExample('fresh-1', 'hotpotqa', 'audit', 'fresh input one', 'unused')]
    config.update(exclude_prior_campaigns=[str(source.parent)], audit_offset=1)
    monkeypatch.setattr(runner, 'load_examples', lambda dataset, split, **kw: examples[split])
    runner.register(config, output)
    protocol = json.loads((output / 'protocol.json').read_text())
    exclusions = protocol['audit_exclusions']
    assert exclusions['audit_pool_before_count'] == 3 and exclusions['audit_pool_after_count'] == 2
    assert exclusions['excluded_examples'][0]['matched_by'] == ['input_hash']
    assert protocol['split_identities']['audit'][0]['id'] == 'fresh-1'
    assert protocol['split_identities']['selection'] == runner.example_identities(examples['selection'])
    assert 'do not consult' not in json.dumps(exclusions)
    assert not physical


def test_prior_id_is_dataset_qualified_but_same_dataset_id_is_protected(tmp_path):
    source = prior_performance(tmp_path)
    examples = exclusion_examples()
    examples['audit'] = [BenchmarkExample('old-id', 'foreign', 'audit', 'different text', ''),
                         BenchmarkExample('old-id', 'hotpotqa', 'audit', 'changed text', '')]
    filtered, provenance = runner.exclude_prior_audit({'exclude_prior_campaigns': [str(source)]},
                                                     tmp_path / 'new', examples)
    assert len(filtered['audit']) == 1 and filtered['audit'][0].dataset == 'foreign'
    assert provenance['excluded_examples'][0]['matched_by'] == ['dataset_and_id']


@pytest.mark.parametrize('split', ['development', 'selection'])
def test_protected_audit_cannot_enter_development_or_selection(tmp_path, split):
    source = prior_performance(tmp_path)
    examples = exclusion_examples()
    examples[split] = [BenchmarkExample('renamed', 'hotpotqa', split, 'protected input', '')]
    with pytest.raises(ValueError, match=f'overlaps {split}'):
        runner.exclude_prior_audit({'exclude_prior_campaigns': [str(source)]}, tmp_path / 'new', examples)


@pytest.mark.parametrize('mutation', ['whitespace', 'deleted'])
def test_changed_prior_source_rejected_before_any_paid_execution(tmp_path, monkeypatch, mutation):
    config, output, physical = fixture(tmp_path, monkeypatch)
    source = prior_performance(tmp_path)
    config['exclude_prior_campaigns'] = [str(source)]
    runner.register(config, output)
    if mutation == 'whitespace':
        source.write_text(source.read_text() + '\n')
    else:
        source.unlink()
    with pytest.raises(ValueError, match='exclusion source'):
        runner.execute(output)
    assert not physical and not (output / '.started').exists()


@pytest.mark.parametrize('mutation', ['missing_hash', 'wrong_count', 'duplicate_id', 'wrong_dataset', 'bad_schema'])
def test_incomplete_prior_identity_metadata_fails_closed(tmp_path, mutation):
    source = prior_performance(tmp_path)
    value = json.loads(source.read_text())
    if mutation == 'missing_hash':
        del value['examples']['audit'][0]['input_hash']
    elif mutation == 'wrong_count':
        value['config']['counts']['audit'] = 2
    elif mutation == 'duplicate_id':
        value['examples']['audit'] *= 2
        value['config']['counts']['audit'] = 2
    elif mutation == 'wrong_dataset':
        value['examples']['audit'][0]['dataset'] = 'different'
    else:
        value['config']['schema'] = 'unknown'
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.prior_audit_identities(source)


def test_legacy_registered_comparison_exclusions_bind_protocol_and_input_snapshot(tmp_path):
    directory = tmp_path / 'legacy'
    directory.mkdir()
    examples = [BenchmarkExample('old', 'hotpotqa', 'audit', 'Old INPUT', {'private_reference': 'never exposed'}).to_dict()]
    protocol = {'config': {'dataset': 'hotpotqa', 'audit_limit': 1},
                'split_hashes': {'audit': runner.digest(examples)}}
    registration = {'schema_version': 1, 'candidate': 'specrsi'}
    protocol['comparison_registration_sha256'] = runner.digest(registration)
    registration['protocol_sha256'] = runner.digest(protocol)
    for name, value in (('registration.json', registration), ('protocol.json', protocol), ('audit_examples.json', examples)):
        (directory / name).write_text(json.dumps(value))
    identities, files = runner.prior_audit_identities(directory / 'registration.json')
    assert len(files) == 3 and identities[0]['id'] == 'old'
    assert 'never exposed' not in json.dumps(identities)
    provenance = {'sources': [{'files_sha256': files}]}
    (directory / 'audit_examples.json').write_text(json.dumps(examples) + ' ')
    with pytest.raises(ValueError, match='source changed'):
        runner.verify_exclusion_sources(provenance)
    examples[0]['prompt'] = 'different input'
    (directory / 'audit_examples.json').write_text(json.dumps(examples))
    with pytest.raises(ValueError, match='frozen protocol'):
        runner.prior_audit_identities(directory / 'registration.json')


def test_new_registered_metadata_can_be_used_without_opening_old_answers(tmp_path, monkeypatch):
    config, output, _ = fixture(tmp_path, monkeypatch)
    runner.register(config, output)
    (output / 'audit_examples.json').unlink()  # New protocol contains the complete frozen input identity.
    identities, files = runner.prior_audit_identities(output / 'registration.json')
    assert identities[0]['id'] == 'audit'
    assert len(files) == 2


@pytest.mark.parametrize('requested', ['string', [None], [''], ['missing/registration.json']])
def test_bad_prior_source_list_refused(tmp_path, requested):
    with pytest.raises(ValueError):
        runner.exclude_prior_audit({'exclude_prior_campaigns': requested}, tmp_path / 'new', exclusion_examples())


def role_fixture(tmp_path, version=3, *, tokens=100, calls=10):
    bridge = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(architecture_version=version, max_length=64)),
        physical_inferences=[], context_text=lambda state: 'context', inference_cost=lambda text: 8)
    state = SimpleNamespace(candidates={'a': None, 'b': None}, scope=SimpleNamespace(tasks=('x', 'y')))
    meter = runner.PhysicalMeter({'initial': {'tokens': tokens, 'calls': calls}}, tmp_path / 'calls.jsonl')
    return bridge, state, meter


def receipt(role, tokens, **extra):
    return {'role': role, 'prompt_tokens': tokens, 'completion_tokens': 0, 'status': 'complete', **extra}


@pytest.mark.parametrize('role', ['scorer', 'policy'])
@pytest.mark.parametrize('cold', [True, False])
def test_v3_multi_forward_admission_and_actual_cache_miss_settlement(tmp_path, role, cold):
    bridge, state, meter = role_fixture(tmp_path)
    cap, calls = runner.role_admission(bridge, state)
    assert (cap, calls) == (64, 5)
    def operation():
        pending = json.loads(meter.journal_path.read_text().splitlines()[0])
        assert pending['status'] == 'pending' and pending['reserved_calls'] == 5
        if cold:
            bridge.physical_inferences.extend([receipt('frozen_node_text', 11), receipt('frozen_node_text', 13)])
        bridge.physical_inferences.append(receipt(role, 8))
        return 'value'
    assert runner.measured_role(bridge, meter, operation, cap, role, calls=calls) == 'value'
    assert meter.spent['initial'] == {'tokens': 32 if cold else 8, 'calls': 3 if cold else 1}
    assert meter.pending is None and not meter.unknown_reservations


@pytest.mark.parametrize('tokens,calls', [(63, 5), (64, 4)])
def test_v3_admits_both_token_and_forward_caps_before_any_inference(tmp_path, tokens, calls):
    bridge, state, meter = role_fixture(tmp_path, tokens=tokens, calls=calls)
    cap, n = runner.role_admission(bridge, state)
    with pytest.raises(runner.PhysicalBudgetStop):
        runner.measured_role(bridge, meter, lambda: pytest.fail('must not start'), cap, 'scorer', calls=n)
    assert not meter.path.read_text() and not meter.journal_path.read_text()


def test_v3_partial_failure_keeps_real_node_receipts_and_unknown_remainder(tmp_path):
    bridge, state, meter = role_fixture(tmp_path)
    def failed():
        bridge.physical_inferences.extend([receipt('frozen_node_text', 11),
                                          receipt('frozen_node_text', 13, status='error')])
        raise RuntimeError('second encoder failed')
    with pytest.raises(RuntimeError, match='encoder failed'):
        runner.measured_role(bridge, meter, failed, 64, 'scorer', calls=5)
    assert meter.spent['initial'] == {'tokens': 24, 'calls': 2}
    assert len(meter.path.read_text().splitlines()) == 2
    unresolved = meter.unknown_reservations[-1]
    assert unresolved['unresolved_reserved_tokens'] == 40 and unresolved['unresolved_reserved_calls'] == 3
    assert not meter.valid and meter.pending is None
    with pytest.raises(RuntimeError, match='unmetered failure'):
        meter.begin(1, 1, 'scorer')


def test_v3_invalid_receipt_sequence_still_retains_actual_cost(tmp_path):
    bridge, state, meter = role_fixture(tmp_path)
    def malformed():
        bridge.physical_inferences.extend([receipt('scorer', 8), receipt('frozen_node_text', 10)])
    with pytest.raises(RuntimeError, match='incompatible'):
        runner.measured_role(bridge, meter, malformed, 64, 'scorer', calls=5)
    assert meter.spent['initial'] == {'tokens': 18, 'calls': 2} and not meter.valid


def test_malformed_receipt_does_not_discard_other_measured_node_costs(tmp_path):
    bridge, state, meter = role_fixture(tmp_path)
    def malformed():
        bridge.physical_inferences.extend([receipt('frozen_node_text', 11),
                                          {'role': 'frozen_node_text', 'prompt_tokens': None},
                                          receipt('scorer', 8)])
    with pytest.raises(RuntimeError, match='Invalid physical token receipt'):
        runner.measured_role(bridge, meter, malformed, 64, 'scorer', calls=5)
    assert meter.spent['initial'] == {'tokens': 19, 'calls': 2}
    assert meter.unknown_reservations[-1]['unresolved_reserved_tokens'] == 45


def test_crash_journal_retains_pending_reservation_before_work(tmp_path):
    bridge, state, meter = role_fixture(tmp_path)
    class ProcessCrash(BaseException):
        pass
    def crash():
        raise ProcessCrash()
    with pytest.raises(ProcessCrash):
        runner.measured_role(bridge, meter, crash, 64, 'policy', calls=5)
    records = [json.loads(line) for line in meter.journal_path.read_text().splitlines()]
    assert records == [{**records[0], 'status': 'pending'}] and len(records) == 1
    assert records[0]['reserved_tokens'] == 64 and records[0]['reserved_calls'] == 5
    with pytest.raises(RuntimeError, match='unresolved physical operation'):
        meter.begin(64, 5, 'policy')


def test_v2_keeps_single_context_forward_contract(tmp_path):
    bridge, state, meter = role_fixture(tmp_path, version=2)
    cap, calls = runner.role_admission(bridge, state)
    assert (cap, calls) == (8, 1)
    def operation():
        bridge.physical_inferences.append(receipt('scorer', 8))
    runner.measured_role(bridge, meter, operation, cap, 'scorer', calls=calls)
    assert meter.spent['initial'] == {'tokens': 8, 'calls': 1} and meter.valid


def test_redistributed_semantics_reserves_total_across_multiple_contexts(tmp_path):
    bridge, state, meter = role_fixture(tmp_path, tokens=8191)
    bridge.model.config = SimpleNamespace(
        architecture_version=3, max_length=2048, semantic_budget_mode='redistribute',
        semantic_global_tokens=512, semantic_node_min_tokens=64,
        semantic_node_max_tokens=2048, semantic_total_tokens=8192)
    cap, calls = runner.role_admission(bridge, state)
    assert (cap, calls) == (8192, 5)
    with pytest.raises(runner.PhysicalBudgetStop):
        runner.measured_role(bridge, meter, lambda: pytest.fail('must reserve all node inputs'),
                             cap, 'scorer', calls=calls)
    assert not meter.path.read_text() and not meter.journal_path.read_text()


def test_registration_uses_checkpoint_semantic_total_and_rejects_invalid_contract(tmp_path, monkeypatch):
    config, output, physical = fixture(tmp_path, monkeypatch)
    config['engineering_smoke_only'] = True
    config['budget']['search_tokens'] = 1000000
    for replica in config['training_replicas']:
        path = Path(replica['checkpoint']) / 'model_config.json'
        model = json.loads(path.read_text())
        model.update(architecture_version=3, max_length=2048, semantic_budget_mode='redistribute',
                     semantic_global_tokens=512, semantic_node_min_tokens=64,
                     semantic_node_max_tokens=2048, semantic_total_tokens=8192,
                     frozen_text_snapshot='frozen_node_text_v1:sha256:' + '0' * 64)
        path.write_text(json.dumps(model))
        torch.save({'value_head.node_text_projection.weight': torch.ones(2, 2) * (replica['seed'] + 1),
                    'policy_head.node_text_projection.weight': torch.ones(2, 2)}, path.parent / 'heads.pt')
        (path.parent / 'training_evidence.json').write_text(json.dumps({'fixture_only': True}))
    observed = []
    original = runner.registered_search_config
    def capture(*args, **kwargs):
        observed.append(kwargs['learned_query_cap'])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'registered_search_config', capture)
    runner.register(config, output)
    assert observed and set(observed) == {8192} and not physical
    path = Path(config['training_replicas'][0]['checkpoint']) / 'model_config.json'
    model = json.loads(path.read_text())
    model['semantic_global_tokens'] = 4096
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match='fit max_length'):
        runner.register(config, tmp_path / 'invalid-campaign')
    assert not physical and not (tmp_path / 'invalid-campaign').exists()


def test_registered_v3_execution_accounts_node_receipts_in_complete_campaign(tmp_path, monkeypatch):
    # Real registered/VATS control flow; neural outputs remain explicit fixtures.
    from specrsi.learning import frozen_text
    from specrsi.learning.paper_bridge import VATSSearchBridge
    config, output, _ = fixture(tmp_path, monkeypatch)
    class V3Roles(Roles):
        config = SimpleNamespace(max_length=64, option_dim=12, feature_dim=8, architecture_version=3)
    monkeypatch.setattr(runner, 'load_roles', lambda *a: V3Roles())
    monkeypatch.setattr(frozen_text, 'node_text_cache', lambda model: SimpleNamespace(validate_identity=lambda: 'fixture'))
    def semantics(self, state, context):
        if not getattr(self, '_fixture_nodes_cached', False):
            self.physical_inferences.extend([receipt('frozen_node_text', 8), receipt('frozen_node_text', 8)])
            self._fixture_nodes_cached = True
        n = len(state.candidates) * (1 + len(state.scope.tasks))
        return {'graph_node_texts': ('fixture input',) * n, 'graph_node_text_features': ((0., 1.),) * n,
                'graph_text_snapshot': 'explicit-neural-fixture-only'}
    monkeypatch.setattr(VATSSearchBridge, '_node_text_record', semantics)
    runner.register(config, output)
    result = runner.execute(output)
    assert result['completed_runs'] == 8
    for directory in (output / 'runs').glob('*-specrsi'):
        raw = json.loads((directory / 'run.json').read_text())
        calls = [json.loads(line) for line in (directory / 'physical_calls.jsonl').read_text().splitlines()]
        assert sum(row['role'] == 'frozen_node_text' for row in calls) == 2
        assert raw['physical_tokens'] == sum(r['prompt_tokens'] + r['completion_tokens'] for r in calls)
        assert raw['actual_model_calls'] == len(calls)
        assert raw['unresolved_physical_reservations'] == []


def test_unknown_task_failure_retains_pre_call_admission_and_blocks_retry(tmp_path):
    class Client:
        def generate(self, *args, **kwargs):
            raise TimeoutError('network lost')
    meter = runner.PhysicalMeter({'initial': {'tokens': 20, 'calls': 1}}, tmp_path / 'calls.jsonl')
    client = runner.MeteredClient(Client(), Tokenizer(), 'pinned', meter)
    with pytest.raises(TimeoutError):
        client.generate([{'role': 'user', 'content': 'q'}], max_tokens=4, total_token_cap=16)
    assert meter.spent['initial'] == {'tokens': 0, 'calls': 0}
    assert meter.unknown_reservations[0]['unresolved_reserved_tokens'] == 8
    assert meter.unknown_reservations[0]['unresolved_reserved_calls'] == 1
    assert json.loads(meter.journal_path.read_text().splitlines()[0])['status'] == 'pending'


def test_exclusion_source_is_rechecked_before_each_physical_call(tmp_path):
    source = prior_performance(tmp_path)
    _, provenance = runner.exclude_prior_audit({'exclude_prior_campaigns': [str(source)]},
                                              tmp_path / 'new', exclusion_examples())
    meter = runner.PhysicalMeter({'initial': {'tokens': 20, 'calls': 2}}, tmp_path / 'calls.jsonl',
                                before_call=lambda: runner.verify_exclusion_sources(provenance))
    meter.begin(8, 1, 'task')
    meter.charge([receipt('task', 8)], 8)
    meter.settle()
    source.write_text(source.read_text() + ' ')
    with pytest.raises(ValueError, match='source changed'):
        meter.begin(8, 1, 'task')
    assert meter.spent['initial']['calls'] == 1


def v3_two_stage_fixture(tmp_path, monkeypatch, *, gvo_updates=0):
    from specrsi.learning.trainer import merge_learning_signals
    config, output, physical = two_stage_fixture(tmp_path, monkeypatch)
    def signal(steps, active_role):
        roles = {r: {'gradient_steps': 0, 'nonzero_gradient_steps': 0, 'parameter_change_steps': 0,
            'signal_parameter_change_steps': 0, 'zero_gradient_change_steps': 0, 'changed_elements': 0,
            'gradient_l2_max': 0.} for r in ('draft', 'policy', 'value', 'verify', 'surface', 'other')}
        if steps:
            roles[active_role].update(gradient_steps=steps, nonzero_gradient_steps=steps,
                parameter_change_steps=steps, signal_parameter_change_steps=steps, changed_elements=steps,
                gradient_l2_max=1.)
        return merge_learning_signals([{'schema': 'role_update_evidence_v1', 'optimizer_steps': steps, 'roles': roles}])
    for replica in config['training_replicas']:
        root = Path(replica['checkpoint']).parent
        summary = json.loads((root / 'metrics.json').read_text())
        warmup_signal = signal(4, 'policy')
        for stage in ('warmup', 'gvo'):
            d = root / stage; checkpoint = d / 'checkpoint'
            model = json.loads((checkpoint / 'model_config.json').read_text())
            model.update(architecture_version=3, frozen_text_snapshot='frozen_node_text_v1:sha256:' + 'a' * 64)
            (checkpoint / 'model_config.json').write_text(json.dumps(model))
            torch.save({'value_head.node_text_projection.weight': torch.ones(2, 2) * (replica['seed'] + 1),
                        'policy_head.node_text_projection.weight': torch.ones(2, 2)}, checkpoint / 'heads.pt')
            steps = 4 if stage == 'warmup' else gvo_updates
            observed = warmup_signal if stage == 'warmup' else signal(steps, 'value')
            evidence = {'schema': 'role_training_evidence_v1', 'architecture_version': 3,
                'learning_signal': warmup_signal if stage == 'warmup' else merge_learning_signals([warmup_signal, observed]),
                'gvo_learning_signal': observed if stage == 'gvo' else merge_learning_signals([])}
            (checkpoint / 'training_evidence.json').write_text(json.dumps(evidence))
            path = d / 'metrics.json'; m = json.loads(path.read_text())
            m.update(initial_optimizer_steps=0 if stage == 'warmup' else 4, optimizer_updates=steps)
            m['history'][0].update(steps=4 if stage == 'warmup' else 4 + gvo_updates, learning_signal=observed)
            path.write_text(json.dumps(m))
            next(r for r in summary['stages'] if r['stage'] == stage)['metrics_sha256'] = runner.file_hash(path)
        (root / 'metrics.json').write_text(json.dumps(summary))
    return config, output, physical


@pytest.mark.parametrize('updates,status', [(0, 'no_parameter_change'), (2, 'critic_only_updated')])
def test_v3_provenance_accepts_honest_zero_actor_signal_without_fabricating_learning(tmp_path, monkeypatch, updates, status):
    config, output, physical = v3_two_stage_fixture(tmp_path, monkeypatch, gvo_updates=updates)
    runner.register(config, output)
    protocol = json.loads((output / 'protocol.json').read_text())
    assert all(r['training_learning_signal']['status'] == status for r in protocol['replicas'])
    assert not any(r['training_learning_signal']['all_actor_roles_updated'] for r in protocol['replicas'])
    assert not physical


@pytest.mark.parametrize('mutation', ['missing', 'incomplete', 'changed_manifest'])
def test_v3_zero_update_requires_consistent_explicit_evidence(tmp_path, monkeypatch, mutation):
    config, output, physical = v3_two_stage_fixture(tmp_path, monkeypatch)
    checkpoint = Path(config['training_replicas'][0]['checkpoint'])
    evidence = checkpoint / 'training_evidence.json'
    if mutation == 'missing':
        evidence.unlink()
    else:
        value = json.loads(evidence.read_text())
        if mutation == 'incomplete':
            value['gvo_learning_signal']['diagnostics_complete'] = False
        else:
            value['gvo_learning_signal']['optimizer_steps'] = 9
        evidence.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='training provenance|role-update evidence'):
        runner.register(config, output)
    assert not physical and not output.exists()
