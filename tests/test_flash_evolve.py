import asyncio
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import yaml

from specrsi.baselines import flash_evolve as flash
from specrsi.benchmarks.base import BenchmarkExample
from specrsi.models import ModelReply


@pytest.fixture(autouse=True)
def optional_upstream_runtime(monkeypatch):
    """Keep pure adapter tests active before optional source download.

    Only tests which actually request the upstream runtime are skipped. Once
    its directory exists, the real pin/hash validation runs without alteration.
    """
    original = flash.pinned_flash

    def require_runtime():
        lock = json.loads(flash.SOURCE_LOCK.read_text())
        if not (flash.ROOT / lock['local_path']).is_dir():
            pytest.skip('Download pinned FlashEvolve with scripts/prepare_upstream.py')
        return original()

    monkeypatch.setattr(flash, 'pinned_flash', require_runtime)


def configuration():
    config = yaml.safe_load((flash.ROOT / 'configs/baselines/flash_evolve_hotpotqa_dev_smoke.yaml').read_text())
    for i, role in enumerate(('development', 'selection', 'audit')):
        config['data'][role] = {'offset': i, 'count': 1}
    config['generation'] = {'task': {'max_tokens': 8, 'temperature': 0., 'top_p': 1.},
                            'proposal': {'max_tokens': 8, 'temperature': .7, 'top_p': .8}}
    return config


class FakeClient:
    model = 'Qwen3.5-35B-A3B'

    def __init__(self, output=None, failure=False, delay=0):
        self.requests, self.output, self.failure, self.delay = [], output, failure, delay

    def generate(self, messages, **settings):
        self.requests.append({'messages': messages, 'settings': settings})
        time.sleep(self.delay)
        if self.failure:
            raise OSError('fixture transport failure after dispatch')
        prompt = json.dumps(messages)
        if len(messages) == 1:
            assert 'AUDIT-SECRET' not in prompt and 'selection question' not in prompt
            answer = '```\nImproved short answer instruction\n```'
        else:
            answer = '42' if 'Improved' in messages[0]['content'] else 'wrong'
            if 'AUDIT-SECRET' in prompt and self.output:
                assert (self.output / 'selected.json').is_file()
                assert (self.output / 'selection_frozen.json').is_file()
        return ModelReply(answer, 3, 2, .01, 'stop', 'fixture-revision', f'request-{len(self.requests)}')


def meter(tmp_path, config=None, client=None):
    config = config or configuration()
    return flash.FlashMeter(client or FakeClient(), SimpleNamespace(apply_chat_template=lambda *a, **kw: [0]*3),
        revision='fixture-revision', budgets=config['budget'], generation=config['generation'],
        seed=config['seed'], context_length=4096, journal=tmp_path / 'calls.jsonl',
        request_token_caps=config.get('request_token_caps'))


def examples():
    return [BenchmarkExample('dev', 'hotpotqa', 'development', 'development question', '42'),
            BenchmarkExample('sel', 'hotpotqa', 'selection', 'selection question', '42'),
            BenchmarkExample('aud', 'hotpotqa', 'audit', 'AUDIT-SECRET', '42')]


@pytest.mark.parametrize('runtime', ['async_gepa', 'sync_gepa'])
def test_actual_official_runtime_gates_candidate_and_accounts_every_call(tmp_path, runtime):
    config = configuration()
    config['algorithm']['runtime'] = runtime
    m = meter(tmp_path, config)
    dev, sel, _ = examples()
    chosen, search = asyncio.run(flash.optimize(config, [dev], [sel], m))
    assert chosen == {'system_prompt': 'Improved short answer instruction', 'selection_score': 1.0}
    assert search['runtime_class'].startswith('flashevolve.runtime.')
    assert search['admitted_candidates'] == 2
    assert not search['budget_truncated']
    assert len(m.records) == 5  # root + rollout + proposal + candidate rollout + full selection
    assert m.receipt()['initial']['actual_calls'] == 1
    assert m.receipt()['search']['actual_calls'] == 4
    assert sum(r['actual_tokens'] for r in m.receipt().values()) == 25
    assert not m.physical_tasks and not any(r['pending_tokens'] for r in m.receipt().values())


def test_native_minibatch_gate_rejects_equal_score_without_full_validation(tmp_path):
    config = configuration()
    config['seed_prompt'] = 'Improved root instruction'
    m = meter(tmp_path, config)
    dev, sel, _ = examples()
    chosen, search = asyncio.run(flash.optimize(config, [dev], [sel], m))
    assert chosen['system_prompt'] == config['seed_prompt']
    assert search['native_stats']['gated_out'] == 1
    assert search['admitted_candidates'] == 1 and len(m.records) == 4


@pytest.mark.parametrize('mode', ['none', 'hard', 'reflective', 'meta'])
def test_official_stale_modes_remain_distinct_and_repair_calls_are_paid(tmp_path, mode):
    config = configuration()
    config['algorithm'].update(staleness_mode=mode, proposal_budget=4)
    config['budget']['search'] = {'tokens': 10000, 'calls': 100}

    class SlowProposal(FakeClient):
        def generate(self, messages, **settings):
            # Let earlier validation advance while later proposals are in flight.
            time.sleep(.04 if len(messages) == 1 else .001)
            return super().generate(messages, **settings)

    m = meter(tmp_path, config, SlowProposal())
    dev, sel, _ = examples()
    _, search = asyncio.run(flash.optimize(config, [dev], [sel], m))
    assert not search['budget_truncated']
    assert search['native_stats']['stale_seen'] > 0
    repairs = [r for r in m.records if r['role'] == 'repair']
    if mode in {'reflective', 'meta'}:
        assert repairs and all(r['phase'] == 'search' and r['cost_kind'] == 'actual' for r in repairs)
        assert search['native_stats']['stale_repaired'] + search['native_stats']['meta_syntheses'] > 0
    else:
        assert not repairs
        assert (search['native_stats']['stale_discarded'] > 0) == (mode == 'hard')
    assert sum(r['actual_calls'] for r in m.receipt().values()) == len(m.records)
    assert sum(r['actual_tokens'] for r in m.receipt().values()) == 5 * len(m.records)


def test_budget_stop_cleans_up_native_async_workers_and_keeps_root(tmp_path):
    config = configuration()
    config['algorithm'].update(proposal_budget=8, rollout_workers=2, propose_workers=2)
    config['budget']['search'] = {'tokens': 100, 'calls': 2}
    m = meter(tmp_path, config, FakeClient(delay=.01))
    dev, sel, _ = examples()
    chosen, search = asyncio.run(flash.optimize(config, [dev], [sel], m))
    assert search['budget_truncated'] and m.denied
    assert chosen['system_prompt'] == config['seed_prompt']
    assert search['admitted_candidates'] == 1
    assert m.receipt()['search']['actual_calls'] == 2
    assert len(m.records) == 3
    assert all(r['cost_valid'] and not r['pending_tokens'] for r in m.receipt().values())
    assert any(e['status'] == 'interrupted' for e in search['stages'])


def test_cancelled_paid_call_is_drained_and_reserved_tokens_prevent_overspend(tmp_path):
    flash.pinned_flash()
    from flashevolve.llm.base import ChatRequest
    config = configuration()
    config['budget']['search'] = {'tokens': 11, 'calls': 2}
    m = meter(tmp_path, config, FakeClient(delay=.03))

    async def run():
        request = ChatRequest(messages=[{'role': 'system', 'content': 'old'}, {'role': 'user', 'content': 'q'}])
        first = asyncio.create_task(m.chat(request, role='task'))
        await asyncio.sleep(.005)
        with pytest.raises(flash.PhysicalBudgetStop):
            await m.chat(request, role='task')
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert m.physical_tasks
        await m.drain()
    asyncio.run(run())
    assert len(m.client.requests) == 1 and len(m.records) == 1
    assert m.receipt()['search']['actual_tokens'] == 5
    assert not m.receipt()['search']['pending_tokens']


def test_unknown_transport_failure_is_invalid_cost_not_free(tmp_path):
    m = meter(tmp_path, client=FakeClient(failure=True))
    dev, sel, _ = examples()
    with pytest.raises(OSError, match='transport failure'):
        asyncio.run(flash.optimize(configuration(), [dev], [sel], m))
    receipt = m.receipt()['initial']
    assert receipt['actual_calls'] == 1 and not receipt['cost_valid']
    assert m.records[0]['status'] == 'unmetered_failure'


def test_register_run_freezes_source_and_selected_before_audit(tmp_path, monkeypatch):
    config = configuration()
    dev_rows = [BenchmarkExample(e.id, e.dataset, 'development', e.prompt, e.reference) for e in examples()]
    monkeypatch.setattr(flash, 'load_examples', lambda *args, **kwargs: dev_rows)
    output = tmp_path / 'campaign'
    registered = flash.register(config, output)
    assert registered['data_source_splits'] == dict.fromkeys(('development', 'selection', 'audit'), 'development')
    assert registered['performance_claim_eligible'] is False
    physical = FakeClient(output)

    def factory(reg, out):
        return flash.FlashMeter(physical, SimpleNamespace(apply_chat_template=lambda *a, **kw: [0]*3),
            revision='fixture-revision', budgets=config['budget'], generation=config['generation'],
            seed=0, context_length=4096, journal=out / 'physical_calls.jsonl')
    result = flash.execute(output, meter_factory=factory)
    assert result['status'] == 'complete' and result['audit_score'] == 1.
    assert result['physical_calls'] == 6 and result['physical_tokens'] == 30
    assert result['sota_claim'] is False and result['optional_spec_implemented'] is False
    assert result['phase_receipts']['audit']['actual_calls'] == 1
    with pytest.raises(FileExistsError):
        flash.execute(output, meter_factory=factory)
    assert len(physical.requests) == 6


def test_optional_spec_label_and_leaky_split_are_rejected(tmp_path, monkeypatch):
    config = configuration()
    config['algorithm']['runtime'] = 'optional-spec'
    with pytest.raises(ValueError, match='optional-spec'):
        flash.validate_config(config)
    config = configuration()
    config['data']['selection']['offset'] = 0
    monkeypatch.setattr(flash, 'load_examples', lambda *args, **kwargs:
        [BenchmarkExample(e.id, e.dataset, 'development', e.prompt, e.reference) for e in examples()])
    with pytest.raises(ValueError, match='Duplicate example'):
        flash.register(config, tmp_path / 'leaky')
    assert not (tmp_path / 'leaky').exists()


@pytest.mark.parametrize('target', ['registration', 'selected', 'search', 'selection_frozen'])
def test_mid_audit_frozen_identity_changes_cannot_publish_success(tmp_path, monkeypatch, target):
    config = configuration()
    monkeypatch.setattr(flash, 'load_examples', lambda *args, **kwargs:
        [BenchmarkExample(e.id, e.dataset, 'development', e.prompt, e.reference) for e in examples()])
    output = tmp_path / 'tamper'
    flash.register(config, output)

    class TamperClient(FakeClient):
        def generate(self, messages, **settings):
            reply = super().generate(messages, **settings)
            if 'AUDIT-SECRET' in json.dumps(messages):
                path = output / f'{target}.json'
                value = json.loads(path.read_text())
                value['changed_during_audit'] = True
                path.write_text(json.dumps(value))
                if target == 'registration':
                    (output / 'registration.sha256').write_text(flash.file_sha(path))
            return reply

    def factory(registration, directory):
        return flash.FlashMeter(TamperClient(output), SimpleNamespace(apply_chat_template=lambda *a, **kw: [0]*3),
            revision='fixture-revision', budgets=config['budget'], generation=config['generation'],
            seed=0, context_length=4096, journal=directory / 'physical_calls.jsonl')
    with pytest.raises(ValueError, match='changed during'):
        flash.execute(output, meter_factory=factory)
    assert not (output / 'summary.json').exists()
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['phase_receipts']['audit']['actual_calls'] == 1
    assert failure['phase_receipts']['audit']['actual_tokens'] == 5


def fixture_campaign(tmp_path, monkeypatch, config=None, name='campaign'):
    config = deepcopy(config or configuration())
    model_dir = tmp_path / 'fixture-model'
    model_dir.mkdir(exist_ok=True)
    (model_dir / 'tokenizer.json').write_text('{}')
    lock = tmp_path / 'fixture-model.lock.json'
    lock.write_text(json.dumps({'revision': 'fixture-revision', 'local_path': str(model_dir)}))
    config['model']['lock'] = str(lock)
    monkeypatch.setattr(flash, 'load_examples', lambda *args, **kwargs:
        [BenchmarkExample(e.id, e.dataset, 'development', e.prompt, e.reference) for e in examples()])
    output = tmp_path / name
    registration = flash.register(config, output)
    physical = FakeClient(output)

    def factory(registration, directory):
        return flash.FlashMeter(physical, SimpleNamespace(apply_chat_template=lambda *a, **kw: [0]*3),
            revision='fixture-revision', budgets=config['budget'], generation=config['generation'],
            seed=config['seed'], context_length=4096, journal=directory / 'physical_calls.jsonl')
    return output, registration, physical, factory


@pytest.mark.parametrize('source', ['development', 'train', 'training'])
def test_native_storage_never_promotes_development_origin(tmp_path, monkeypatch, source):
    config = configuration()
    config['data']['mode'] = 'native_splits'
    for role in config['data']:
        if role != 'mode':
            config['data'][role]['offset'] = 0
    by_role = {e.split: [BenchmarkExample(e.id, e.dataset, e.split, e.prompt, e.reference,
        {'source_split': source, 'evaluation_scope': 'development_diagnostic_only_not_paper_audit'})]
        for e in examples()}
    monkeypatch.setattr(flash, 'load_examples', lambda dataset, split, **kwargs: by_role[split])
    reg = flash.register(config, tmp_path / 'native_files')
    assert reg['data_storage_splits']['audit'] == 'audit'
    assert reg['data_source_splits']['audit'] == 'development'
    assert reg['data_identity']['audit'][0]['source_split'] == 'development'
    assert not reg['performance_claim_eligible']
    assert not reg['evaluation_eligibility']['eligible']
    flash.verify_registration(tmp_path / 'native_files')


def test_explicit_diagnostic_scope_blocks_native_audit_eligibility():
    sample = BenchmarkExample('x', 'hotpotqa', 'audit', 'q', 'a',
        {'flash_source_split': 'audit', 'evaluation_scope': 'development_diagnostic_only_not_paper_audit'})
    assert not flash.evaluation_eligibility({'audit': [sample]})['eligible']


def test_identity_seed_ignores_reference_candidate_order_and_storage_role():
    contract = {'mode': 'example_identity_v1', 'seed': 991}
    a = BenchmarkExample('id', 'hotpotqa', 'audit', 'A question', 'hidden answer')
    b = BenchmarkExample('id', 'hotpotqa', 'selection', 'A question', 'different hidden answer')
    c = BenchmarkExample('another-id', 'hotpotqa', 'audit', 'A question', 'hidden answer')
    assert flash.example_generation_seed(a, contract) == flash.example_generation_seed(b, contract)
    assert flash.example_generation_seed(a, contract) != flash.example_generation_seed(c, contract)
    assert flash.example_generation_seed(a, {'mode': 'fixed', 'seed': 123}) == 123
    assert flash.example_generation_seed(a, {'mode': 'legacy_call_index'}) is None


@pytest.mark.parametrize('runtime', ['async_gepa', 'sync_gepa'])
def test_fixed_task_seeds_pair_root_and_candidates_without_changing_proposal_seeds(tmp_path, runtime):
    config = configuration()
    config['algorithm']['runtime'] = runtime
    config['task_sampling'] = {'mode': 'fixed', 'seed': 123456}
    m = meter(tmp_path, config)
    dev, sel, _ = examples()
    asyncio.run(flash.optimize(config, [dev], [sel], m))
    assert {r['sampling']['seed'] for r in m.records if r['role'] == 'task'} == {123456}
    proposal = next(r for r in m.records if r['role'] == 'proposal')
    assert proposal['sampling']['seed'] == config['seed'] + int(proposal['call_id'].split('-')[1])


def test_search_only_then_audit_restores_complete_bill_and_does_no_early_heldout_call(tmp_path, monkeypatch):
    output, reg, physical, factory = fixture_campaign(tmp_path, monkeypatch)
    frozen = flash.search_only(output, meter_factory=factory)
    assert frozen['status'] == 'search_complete' and frozen['audit_model_calls'] == 0
    assert all('AUDIT-SECRET' not in json.dumps(r['messages']) for r in physical.requests)
    assert not (output / 'summary.json').exists() and not (output / 'audit.json').exists()
    before = (output / 'physical_calls.jsonl').read_bytes()
    result = flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=factory)
    assert result['physical_calls'] == 6 and result['physical_tokens'] == 30
    assert (output / 'physical_calls.jsonl').read_bytes().startswith(before)
    assert flash.file_sha(output / 'selection_frozen.json') == frozen['selection_frozen_sha256']
    expected = flash.example_generation_seed(examples()[2], reg['sampling_contracts']['audit_sampling'])
    assert physical.requests[-1]['settings']['seed'] == expected
    with pytest.raises(FileExistsError):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=factory)
    assert len(physical.requests) == 6


@pytest.mark.parametrize('runtime', ['async_gepa', 'sync_gepa'])
def test_output_contract_reaches_initial_search_and_frozen_audit_in_official_runtime(tmp_path, monkeypatch, runtime):
    from specrsi.output_contracts import HOTPOT_SHORT_ANSWER_V1, draft_interface_guidance
    config = configuration()
    config['algorithm']['runtime'] = runtime
    config['generation']['task']['output_contract'] = 'hotpot_short_answer_v1'
    output, reg, physical, factory = fixture_campaign(tmp_path, monkeypatch, config)
    assert reg['config']['generation']['task']['output_contract'] == 'hotpot_short_answer_v1'
    assert 'src/specrsi/output_contracts.py' in flash.own_sources()
    frozen = flash.search_only(output, meter_factory=factory)
    result = flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=factory)
    assert result['status'] == 'complete'
    task_calls = [call for call in physical.requests if len(call['messages']) == 2]
    assert len(task_calls) == 5  # Initial, rollouts, selection, and audit all use it.
    assert all(call['messages'][0]['content'].endswith(HOTPOT_SHORT_ANSWER_V1) for call in task_calls)
    assert all('output_contract' not in call['settings'] for call in physical.requests)
    records = [json.loads(line) for line in (output / 'physical_calls.jsonl').read_text().splitlines()]
    assert {r['phase'] for r in records if r['role'] == 'task'} == {'initial', 'search', 'audit'}
    assert all(r['sampling']['output_contract'] == 'hotpot_short_answer_v1' for r in records if r['role'] == 'task')
    proposal = next(r for r in records if r['role'] == 'proposal')
    assert proposal['messages'][-1]['content'].endswith(draft_interface_guidance(config['generation']['task']))


@pytest.mark.parametrize('role', ['proposal', 'repair'])
def test_public_output_contract_reaches_proposers_and_is_charged_without_mutating_upstream_request(tmp_path, role):
    from specrsi.output_contracts import draft_interface_guidance
    flash.pinned_flash()
    from flashevolve.llm.base import ChatRequest
    config = configuration()
    config['generation']['task']['output_contract'] = 'hotpot_short_answer_v1'
    m = meter(tmp_path, config)
    m.token_count = lambda messages: len(json.dumps(messages).encode())
    calls = []

    def generate(messages, **kwargs):
        calls.append((messages, kwargs))
        return ModelReply('Edited instruction', m.token_count(messages), 2, .01,
                          'stop', 'fixture-revision', 'unique')

    m.client.generate = generate
    request = ChatRequest(messages=[{'role': 'user', 'content': 'ORIGINAL_TEMPLATE_AND_FEEDBACK'}])
    wrapped = flash.ModelRole(m, role, task_generation=config['generation']['task'])
    asyncio.run(wrapped.chat(request))
    assert request.messages == [{'role': 'user', 'content': 'ORIGINAL_TEMPLATE_AND_FEEDBACK'}]
    assert calls[0][0][-1]['content'].endswith(draft_interface_guidance(config['generation']['task']))
    assert m.records[0]['prompt_tokens'] == m.token_count(calls[0][0])
    assert m.receipt()['search']['actual_tokens'] == m.token_count(calls[0][0]) + 2
    assert calls[0][1]['total_token_cap'] == m.token_count(calls[0][0]) + config['generation']['proposal']['max_tokens']


def test_output_contract_cannot_be_silently_ignored_by_proposal_or_unknown_version():
    config = configuration()
    config['generation']['proposal']['output_contract'] = 'hotpot_short_answer_v1'
    with pytest.raises(ValueError, match='Invalid fixed generation'):
        flash.validate_config(config)
    config = configuration()
    config['generation']['task']['output_contract'] = 'unknown'
    with pytest.raises(ValueError, match='Unknown output_contract'):
        flash.validate_config(config)


def test_audit_seed_is_independent_of_search_request_count(tmp_path, monkeypatch):
    audit_seeds, search_counts = [], []
    for number in (1, 2):
        config = configuration()
        config['algorithm']['proposal_budget'] = number
        config['budget']['search']['calls'] = 100
        config['audit_sampling'] = {'mode': 'example_identity_v1', 'seed': 987}
        output, _, physical, factory = fixture_campaign(tmp_path, monkeypatch, config, f'run-{number}')
        frozen = flash.search_only(output, meter_factory=factory)
        search_counts.append(len(physical.requests))
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=factory)
        audit_seeds.append(physical.requests[-1]['settings']['seed'])
    assert search_counts[0] != search_counts[1]
    assert audit_seeds[0] == audit_seeds[1]


@pytest.mark.parametrize('mode', ['fixed', 'legacy_call_index'])
def test_audit_sampling_compatibility_is_explicitly_registered(tmp_path, monkeypatch, mode):
    config = configuration()
    config['audit_sampling'] = {'mode': mode, **({'seed': 777} if mode == 'fixed' else {})}
    output, registered, physical, factory = fixture_campaign(tmp_path, monkeypatch, config)
    result = flash.execute(output, meter_factory=factory)
    assert result['sampling_contracts']['audit_sampling'] == config['audit_sampling']
    expected = 777 if mode == 'fixed' else config['seed'] + len(physical.requests) - 1
    assert physical.requests[-1]['settings']['seed'] == expected


@pytest.mark.parametrize('target', ['selected.json', 'search.json', 'physical_calls.jsonl', 'audit_examples.json'])
def test_between_phase_artifact_tampering_rejected_before_model_creation(tmp_path, monkeypatch, target):
    output, _, physical, factory = fixture_campaign(tmp_path, monkeypatch)
    frozen = flash.search_only(output, meter_factory=factory)
    before = len(physical.requests)
    path = output / target
    path.write_bytes(path.read_bytes() + b' ')

    def forbidden_factory(*args):
        pytest.fail('Tampered evidence reached model creation')
    # Whitespace alone changes the raw selected/search/journal binding. For
    # canonical split identity change the actual payload, not just formatting.
    if target == 'audit_examples.json':
        rows = json.loads(path.read_text()); rows[0]['reference'] = 'changed'
        path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='changed'):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=forbidden_factory)
    assert len(physical.requests) == before and not (output / 'audit_started.json').exists()


def make_barrier(path, campaigns, planned=('a', 'b')):
    value = {'schema': 'cross_method_selection_barrier_v1', 'planned_run_ids': list(planned),
        'runs': {name: {field: {'path': str((directory / f'{field}.json').resolve()),
                              'sha256': flash.file_sha(directory / f'{field}.json')}
                        for field in ('selection_frozen', 'selected')}
                 for name, directory in campaigns.items()}}
    path.write_text(json.dumps(value))
    return flash.file_sha(path)


def test_cross_method_barrier_is_required_complete_and_bound_before_any_audit(tmp_path, monkeypatch):
    campaigns = {}
    for name in ('a', 'b'):
        config = configuration()
        config['comparison'] = {'run_id': name, 'planned_run_ids': ['a', 'b']}
        output, reg, physical, factory = fixture_campaign(tmp_path, monkeypatch, config, name)
        if name == 'a':
            with pytest.raises(ValueError, match='search-only'):
                flash.execute(output, meter_factory=factory)
            assert not physical.requests
        frozen = flash.search_only(output, meter_factory=factory)
        campaigns[name] = (output, physical, factory, frozen)
    output, physical, factory, frozen = campaigns['a']
    with pytest.raises(ValueError, match='requires.*barrier'):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=factory)
    barrier = tmp_path / 'barrier.json'
    checksum = make_barrier(barrier, {'a': output})
    with pytest.raises(ValueError, match='exact registered run set'):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'],
            barrier_path=barrier, barrier_sha256=checksum, meter_factory=factory)
    assert len(physical.requests) == 5 and not (output / 'audit_started.json').exists()
    checksum = make_barrier(barrier, {name: row[0] for name, row in campaigns.items()})
    result = flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'],
        barrier_path=barrier, barrier_sha256=checksum, meter_factory=factory)
    assert result['external_barrier']['sha256'] == checksum and result['physical_calls'] == 6
    assert len(campaigns['b'][1].requests) == 5  # The other arm is still frozen, not evaluated.


def test_other_method_candidate_tamper_blocks_this_methods_heldout(tmp_path, monkeypatch):
    config = configuration()
    config['comparison'] = {'run_id': 'a', 'planned_run_ids': ['a', 'b']}
    output, _, physical, factory = fixture_campaign(tmp_path, monkeypatch, config, 'a')
    frozen = flash.search_only(output, meter_factory=factory)
    other = tmp_path / 'other'; other.mkdir()
    for name in ('selected', 'selection_frozen'):
        (other / f'{name}.json').write_text('{}')
    barrier = tmp_path / 'barrier.json'
    checksum = make_barrier(barrier, {'a': output, 'b': other})
    (other / 'selected.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='frozen selected changed'):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'],
            barrier_path=barrier, barrier_sha256=checksum, meter_factory=factory)
    assert len(physical.requests) == 5 and not (output / 'audit_started.json').exists()


def test_invalid_audit_finish_reason_is_paid_but_cannot_publish_success(tmp_path, monkeypatch):
    output, _, physical, factory = fixture_campaign(tmp_path, monkeypatch)
    frozen = flash.search_only(output, meter_factory=factory)
    original = physical.generate

    def invalid(*args, **kwargs):
        reply = original(*args, **kwargs)
        # Simulate an invalid third-party client object after a physically paid
        # call, bypassing ModelReply's own constructor validation deliberately.
        object.__setattr__(reply, 'finish_reason', 'error')
        return reply
    physical.generate = invalid
    with pytest.raises(ValueError, match='status contract'):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=factory)
    assert not (output / 'summary.json').exists()
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['phase_receipts']['audit']['actual_calls'] == 1
    assert failure['phase_receipts']['audit']['charged_tokens'] == 5
    assert not failure['phase_receipts']['audit']['cost_valid']


def test_barrier_cannot_reuse_one_candidate_as_multiple_method_slots(tmp_path, monkeypatch):
    config = configuration()
    config['comparison'] = {'run_id': 'a', 'planned_run_ids': ['a', 'b']}
    output, _, physical, factory = fixture_campaign(tmp_path, monkeypatch, config)
    frozen = flash.search_only(output, meter_factory=factory)
    barrier = tmp_path / 'barrier.json'
    checksum = make_barrier(barrier, {'a': output, 'b': output})
    with pytest.raises(ValueError, match='reuse the same'):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'],
            barrier_path=barrier, barrier_sha256=checksum, meter_factory=factory)
    assert len(physical.requests) == 5 and not (output / 'audit_started.json').exists()


@pytest.mark.parametrize('target', ['registered_source', 'model_lock'])
def test_source_or_model_change_rejected_before_audit_client_creation(tmp_path, monkeypatch, target):
    output, registration, physical, factory = fixture_campaign(tmp_path, monkeypatch)
    frozen = flash.search_only(output, meter_factory=factory)
    original = flash.file_sha
    selected_path = (flash.ROOT / 'src/specrsi/models/client.py' if target == 'registered_source'
                     else Path(registration['config']['model']['lock']))

    def changed(path):
        return '0' * 64 if Path(path).resolve() == selected_path.resolve() else original(path)
    monkeypatch.setattr(flash, 'file_sha', changed)

    def forbidden_factory(*args):
        pytest.fail('Changed source/model reached audit model construction')
    with pytest.raises(ValueError, match='source or model contract changed'):
        flash.audit_only(output, expected_frozen_sha256=frozen['selection_frozen_sha256'], meter_factory=forbidden_factory)
    assert len(physical.requests) == 5

@pytest.mark.parametrize('role', ['task', 'proposal', 'repair'])
def test_registered_per_role_request_cap_rejects_before_physical_call(tmp_path, role):
    flash.pinned_flash()
    from flashevolve.llm.base import ChatRequest
    config = configuration()
    config['request_token_caps'] = {'task': 10, 'proposal': 10}
    flash.validate_config(config)
    physical = FakeClient()
    m = meter(tmp_path, config, physical)
    with pytest.raises(flash.PhysicalBudgetStop, match='rejected before dispatch'):
        asyncio.run(m.chat(ChatRequest(messages=[{'role': 'user', 'content': 'test'}]), role=role))
    assert not physical.requests and not m.records
    assert m.denied[-1]['request_token_cap'] == 10
    assert m.denied[-1]['requested_tokens'] == 11


def test_per_role_request_cap_preserves_full_output_admission(tmp_path):
    flash.pinned_flash()
    from flashevolve.llm.base import ChatRequest
    config = configuration()
    config['request_token_caps'] = {'task': 11, 'proposal': 10}
    physical = FakeClient()
    m = meter(tmp_path, config, physical)
    asyncio.run(m.chat(ChatRequest(messages=[{'role': 'user', 'content': 'test'}]), role='task'))
    assert physical.requests[0]['settings']['max_tokens'] == 8
    assert physical.requests[0]['settings']['total_token_cap'] == 11
    assert sum(r['actual_tokens'] for r in m.receipt().values()) == 5


def test_invalid_request_token_caps_cannot_register():
    for caps in ({'task': 1}, {'task': 1, 'proposal': 0}, {'task': True, 'proposal': 1}):
        config = configuration()
        config['request_token_caps'] = caps
        with pytest.raises(ValueError, match='request_token_caps'):
            flash.validate_config(config)
