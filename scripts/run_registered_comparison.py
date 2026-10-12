#!/usr/bin/env python3
"""Register, then execute matched prospective local prompt-search comparisons.

All arms share a frozen task backbone and trained proposer for each training seed.
Every task generation, proposal generation, scorer forward and policy forward is
metered. One model call means one generation or one discriminative forward; an
autoregressive generation uses conventional input + output token accounting.
Initialization, optimization and final audit have separately reserved common
caps. All successors are frozen before any audit example is evaluated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import yaml

from specrsi.benchmarks import BenchmarkExample, load_examples, validate_disjoint_splits
from specrsi.core import BudgetLedger, Candidate, EvaluationScope, Ledger, SearchConfig, Task
from specrsi.core.acceptance import accept
from specrsi.core.models import thaw
from specrsi.core.vats import VATSEngine, VATSState
from specrsi.experiments import QwenExecutor
from specrsi.paper_profile import require_role_checkpoint, vats_config
from specrsi.statistics import compare_registered_methods, validate_comparison_registration
from specrsi.learning.frozen_text import semantic_input_cap
from specrsi.tokenization import chat_token_ids

METHODS = ('specrsi', 'initial', 'best_of_n', 'random_search')
METHOD_DEFINITIONS = {
    'specrsi': 'PDF learned VATS with the registered proposer, scorer and independent policy',
    'initial': 'Unmodified common initial prompt; no optimization',
    'best_of_n': 'Independent edits of the common root, complete selection evaluation, best complete prompt',
    'random_search': 'Uniform parent among complete evaluated prompts, complete selection evaluation, best complete prompt',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def prefix_hash(path, size):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while size:
            block = handle.read(min(size, 1 << 20))
            if not block:
                raise ValueError('Frozen physical search prefix was truncated')
            value.update(block)
            size -= len(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def source_hashes():
    paths = set((ROOT / 'src/specrsi').rglob('*.py'))
    paths.update((Path(__file__).resolve(), ROOT / 'scripts/compare_methods.py'))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}


def checkpoint_hashes(path):
    return {str(file.relative_to(path)): file_hash(file) for file in sorted(Path(path).rglob('*')) if file.is_file()}


def example_identities(examples):
    return [{'dataset': e.dataset, 'id': e.id, 'input_hash': e.content_hash} for e in examples]


def prior_audit_identities(path):
    """Read frozen input metadata only, never old outcomes or scores.

    Legacy registered comparisons stored full examples instead of an identity
    manifest. Verify their registered digest, then consult only id/dataset/prompt
    fields; references are neither exposed nor used to construct exclusions.
    """
    files = {}

    def read(source):
        try:
            raw = source.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot read prior audit metadata: {source}') from error
        files[str(source)] = hashlib.sha256(raw).hexdigest()
        return value

    previous = read(path)
    if not isinstance(previous, dict):
        raise ValueError('Prior audit registration must be an object')
    config = previous.get('config', {})
    if isinstance(config, dict) and config.get('schema') == 'inference_performance_v1':
        examples, counts = previous.get('examples'), config.get('counts')
        if not isinstance(examples, dict) or not isinstance(counts, dict):
            raise ValueError('Prior audit registration metadata is incomplete')
        rows = examples.get('audit')
        dataset = config.get('dataset')
        count = counts.get('audit')
    elif previous.get('schema_version') == 1 and 'protocol_sha256' in previous:
        protocol = read(path.parent / 'protocol.json')
        if (not isinstance(protocol, dict) or digest(protocol) != previous['protocol_sha256']
                or digest({k: v for k, v in previous.items() if k != 'protocol_sha256'})
                   != protocol.get('comparison_registration_sha256')):
            raise ValueError('Prior registered comparison protocol identity disagrees')
        config = protocol.get('config', {})
        if (not isinstance(config, dict) or not isinstance(protocol.get('split_hashes'), dict)
                or not isinstance(protocol.get('split_identities', {}), dict)):
            raise ValueError('Prior comparison input manifest is incomplete')
        dataset, count = config.get('dataset'), config.get('audit_limit')
        rows = protocol.get('split_identities', {}).get('audit')
        if rows is None:
            examples = read(path.parent / 'audit_examples.json')
            if not isinstance(examples, list) or digest(examples) != protocol.get('split_hashes', {}).get('audit'):
                raise ValueError('Prior audit examples disagree with their frozen protocol')
            rows = []
            for example in examples:
                if (not isinstance(example, dict) or example.get('dataset') != dataset
                        or example.get('split') != 'audit' or not isinstance(example.get('prompt'), str)):
                    raise ValueError('Prior audit input metadata is incomplete')
                rows.append({'id': example.get('id'), 'input_hash': hashlib.sha256(
                    ' '.join(example['prompt'].split()).casefold().encode()).hexdigest()})
    else:
        raise ValueError('Unsupported prior audit registration; comparable input identities are required')
    if (not isinstance(dataset, str) or not dataset or type(count) is not int or count < 1
            or not isinstance(rows, list) or len(rows) != count):
        raise ValueError('Prior registered audit count or dataset is invalid')
    identities, ids = [], set()
    for row in rows:
        key = row.get('id') if isinstance(row, dict) else None
        checksum = row.get('input_hash') if isinstance(row, dict) else None
        if (not isinstance(key, str) or not key or key in ids
                or row.get('dataset', dataset) != dataset or not isinstance(checksum, str)
                or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum)):
            raise ValueError('Prior audits require unique dataset-qualified IDs and SHA256 input hashes')
        ids.add(key)
        identities.append({'dataset': dataset, 'id': key, 'input_hash': checksum})
    return identities, files


def exclude_prior_audit(config, output, splits):
    """Protect all prior registered audit inputs before taking the new slice."""
    if 'exclude_prior_campaigns' not in config:
        return splits, None
    requested = config['exclude_prior_campaigns']
    if not isinstance(requested, list) or any(not isinstance(p, str) or not p.strip() for p in requested):
        raise ValueError('exclude_prior_campaigns must be a list of registration paths')
    sources, seen, ids, hashes = [], set(), set(), set()
    destination = (Path(output) / 'registration.json').resolve()
    for configured in requested:
        path = (ROOT / configured).resolve()
        if path.is_dir():
            path = path / 'registration.json'
        path = path.resolve()
        if path.name != 'registration.json' or path == destination or path in seen:
            raise ValueError('Prior exclusions require distinct external registration.json files')
        seen.add(path)
        identities, files = prior_audit_identities(path)
        ids.update((r['dataset'], r['id']) for r in identities)
        hashes.update(r['input_hash'] for r in identities)
        sources.append({'configured_path': configured, 'registration_path': str(path),
                        'files_sha256': files, 'registered_audit_count': len(identities),
                        'audit_identities': identities})

    def matched(example):
        return (['dataset_and_id'] if (example.dataset, example.id) in ids else []) + (
            ['input_hash'] if example.content_hash in hashes else [])

    for split in ('development', 'selection'):
        if any(matched(example) for example in splits[split]):
            raise ValueError(f'Prior protected audit overlaps {split}; cannot use it for training or selection')
    kept, removed = [], []
    for example in splits['audit']:
        reasons = matched(example)
        if reasons:
            removed.append({**example_identities([example])[0], 'matched_by': reasons})
        else:
            kept.append(example)
    provenance = {'schema': 'registered_comparison_audit_exclusions_v1', 'sources': sources,
        'policy': 'all prior registered audit inputs, regardless of execution; dataset-qualified ID OR global input hash',
        'input_identity_boundary': 'whitespace/casefold visible-input SHA256; semantic rewrites are not detected',
        'complete_historical_exposure_claimed': False, 'development_selection_unchanged': True,
        'unique_prior_audit_ids': len(ids), 'unique_prior_audit_input_hashes': len(hashes),
        'audit_pool_before_count': len(splits['audit']), 'audit_pool_after_count': len(kept),
        'audit_pool_before_identity_sha256': digest(example_identities(splits['audit'])),
        'audit_pool_after_identity_sha256': digest(example_identities(kept)),
        'development_identity_sha256': digest(example_identities(splits['development'])),
        'selection_identity_sha256': digest(example_identities(splits['selection'])),
        'excluded_count': len(removed), 'excluded_examples': removed}
    verify_exclusion_sources(provenance)
    return {**splits, 'audit': kept}, provenance


def verify_exclusion_sources(provenance):
    for source in provenance['sources'] if provenance is not None else ():
        for path, expected in source['files_sha256'].items():
            try:
                actual = file_hash(path)
            except OSError as error:
                raise ValueError('Prior audit exclusion source disappeared') from error
            if actual != expected:
                raise ValueError('Prior audit exclusion source changed')


def load_backbone(config):
    from transformers import AutoTokenizer
    from specrsi.models import LocalModelClient
    settings = config['model']
    lock = json.loads((ROOT / settings['lock']).read_text())
    client = LocalModelClient(settings['base_url'], settings.get('served_model_name', lock.get('served_model_name', 'Qwen3.5-35B-A3B')))
    health = client.health()
    if (health.get('revision') != lock['revision'] or health.get('protocol_version', 0) < 2
            or not health.get('server_enforces_total_token_cap')):
        raise ValueError('The task server must match the locked revision and enforce token caps')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / lock['local_path'], local_files_only=True,
                                             trust_remote_code=lock.get('trust_remote_code', False))
    return client, tokenizer, lock, health


def load_roles(checkpoint, device):
    from specrsi.learning.hf import HFRoleModel
    model = HFRoleModel.load(checkpoint, device=device)
    model.eval()
    return model


def make_drafter(bridge, settings, validator):
    from specrsi.learning.bridge import HFPromptDrafter
    return HFPromptDrafter(bridge, max_new_tokens=settings.get('draft_max_tokens', 128),
        top_k=settings.get('draft_top_k', 32), temperature=settings.get('draft_temperature', 1.),
        exploration=settings.get('draft_exploration', 0.), proposal_snapshot=bridge.snapshot_id,
        candidate_validator=validator)


def registered_search_config(config, *, learned_query_cap, seed=0, snapshot='frozen-registered-roles'):
    """Use the same checked search contract before registration and execution."""
    base = SearchConfig(execution_cap=config['budget']['execution_cap'], draft_cap=config['budget']['draft_cap'],
        max_candidates=config['search']['max_candidates'], max_actions=config['search']['max_actions'],
        max_children=config['search'].get('max_children', 2), all_u=True, max_sources=0, check_cap=0,
        planning_cost=0, reserve_completion=False, cost_unit='contracted_tokens', seed=seed)
    # A local combinatorial solver does not consume language-model tokens.
    settings = {'algorithm': 'vats_pdf', 'vats': {**config['search'].get('vats', {}), 'solver_call_cost': 0}}
    options = vats_config(base, settings, learned_query_cap=learned_query_cap)
    return replace(options, scorer_snapshot=snapshot, policy_snapshot=snapshot)


def training_run_provenance(checkpoint, seed):
    """Bind a completed replica to its actual standalone run or SFT/GVO chain."""
    checkpoint = Path(checkpoint).resolve()
    run_root = checkpoint.parent
    provenance = {}

    def read(path):
        if not path.is_file():
            raise ValueError(f'Statistical comparison requires completed training provenance: {path}')
        provenance[str(path)] = file_hash(path)
        return json.loads(path.read_text())

    def stage(directory):
        resolved = read(directory / 'resolved_config.json')
        metrics = read(directory / 'metrics.json')
        training = read(directory / 'checkpoint' / 'training_config.json')
        architecture = read(directory / 'checkpoint' / 'model_config.json').get('architecture_version', 2)
        if (resolved.get('training', {}).get('seed') != seed or training.get('seed') != seed
                or metrics.get('algorithm') != 'vats_pdf' or metrics.get('synthetic_demo')
                or metrics.get('source_changed_during_run') is not False or not metrics.get('history')):
            raise ValueError('Training provenance is incomplete, changed, or disagrees with the registered replica')
        if architecture == 3:
            from specrsi.learning.trainer import merge_learning_signals
            signal = merge_learning_signals(row.get('learning_signal') for row in metrics['history'])
            evidence = read(directory / 'checkpoint' / 'training_evidence.json')
            key = 'gvo_learning_signal' if resolved.get('stage') == 'gvo' else 'learning_signal'
            if (not signal['diagnostics_complete'] or evidence.get('schema') != 'role_training_evidence_v1'
                    or evidence.get('architecture_version') != 3 or evidence.get(key) != signal
                    or type(metrics.get('optimizer_updates')) is not int
                    or signal['optimizer_steps'] != metrics['optimizer_updates']):
                raise ValueError('v3 training provenance requires consistent observed role-update evidence')
        return resolved, metrics

    resolved, metrics = stage(run_root)
    pipeline_root = run_root.parent
    registration_path = pipeline_root / 'registration.json'
    if (not registration_path.is_file()
            or json.loads(registration_path.read_text()).get('pipeline') != 'pdf_sft_then_refreshed_gvo_v1'):
        return run_root, provenance
    if run_root.name != 'gvo':
        raise ValueError('Two-stage provenance requires the final GVO checkpoint, not an intermediate stage')
    registration = read(registration_path)
    summary = read(pipeline_root / 'metrics.json')
    if (summary.get('pipeline') != registration['pipeline'] or summary.get('complete') is not True
            or [row.get('stage') for row in summary.get('stages', [])] != ['warmup', 'gvo']
            or not (pipeline_root / 'checkpoint').is_symlink()
            or (pipeline_root / 'checkpoint').resolve() != checkpoint
            or (ROOT / summary.get('checkpoint', '')).resolve() != checkpoint):
        raise ValueError('Two-stage provenance requires the completed pipeline and its final GVO checkpoint link')
    warmup_dir = pipeline_root / 'warmup'
    warmup_resolved, warmup_metrics = stage(warmup_dir)
    stages = {'warmup': (warmup_dir, warmup_resolved, warmup_metrics), 'gvo': (run_root, resolved, metrics)}
    registered_stages = registration.get('stages', [])
    if ([row.get('stage') for row in registered_stages] != ['warmup', 'gvo']
            or any(row.get('training', {}).get('seed') != seed for row in registered_stages)):
        raise ValueError('Two-stage registration must bind both stages to the actual training seed')
    if (not warmup_metrics.get('source_hashes')
            or warmup_metrics['source_hashes'] != metrics.get('source_hashes')
            or (ROOT / resolved.get('resume', '')).resolve() != (warmup_dir / 'checkpoint').resolve()):
        raise ValueError('Two-stage sources or SFT-to-GVO resume provenance disagree')
    for evidence in summary['stages']:
        name = evidence['stage']
        directory, stage_resolved, stage_metrics = stages[name]
        registered_stage = next(row for row in registered_stages if row['stage'] == name)
        if any(stage_resolved.get(key) != value for key, value in registered_stage.items()):
            raise ValueError('Two-stage resolved configuration differs from its prospective registration')
        stage_checkpoint = (directory / 'checkpoint').resolve()
        if (stage_resolved.get('stage') != name
                or (ROOT / stage_resolved.get('output', '')).resolve() != directory
                or (ROOT / evidence.get('metrics', '')).resolve() != directory / 'metrics.json'
                or evidence.get('metrics_sha256') != file_hash(directory / 'metrics.json')
                or (ROOT / evidence.get('checkpoint', '')).resolve() != stage_checkpoint
                or (ROOT / stage_metrics.get('checkpoint', '')).resolve() != stage_checkpoint):
            raise ValueError('Two-stage stage paths or metric hashes disagree with the completed pipeline')
        for name, checksum in checkpoint_hashes(stage_checkpoint).items():
            provenance[str(stage_checkpoint / name)] = checksum
    warmup_steps = warmup_metrics['history'][-1].get('steps')
    gvo_steps = metrics['history'][-1].get('steps')
    v3_gvo = json.loads((checkpoint / 'model_config.json').read_text()).get('architecture_version', 2) == 3
    if (type(warmup_steps) is not int or type(gvo_steps) is not int
            or warmup_steps <= 0 or gvo_steps < warmup_steps or (not v3_gvo and gvo_steps == warmup_steps)):
        raise ValueError('Both SFT and resumed GVO must contain actual optimizer updates')
    for value, initial, final in ((warmup_metrics, 0, warmup_steps), (metrics, warmup_steps, gvo_steps)):
        if (value.get('initial_optimizer_steps', initial) != initial
                or value.get('optimizer_updates', final - initial) != final - initial):
            raise ValueError('Recorded stage optimizer updates disagree with checkpoint continuation')
    return pipeline_root, provenance


def register(config, output):
    """Freeze all inputs and methods before any paid model generation."""
    config = json.loads(canonical(config))
    if config['dataset'] not in {'hotpotqa', 'aime25', 'aime_2025_2026', 'game24', 'ifbench', 'math'}:
        raise ValueError('This runner supports single-prompt native metrics only; structured tasks need separate registered adapters')
    methods = config.get('methods', list(METHODS))
    if (not isinstance(methods, list) or len(set(methods)) != len(methods)
            or 'specrsi' not in methods or len(methods) < 2 or set(methods) - set(METHODS)):
        raise ValueError('Register specrsi and at least one supported local control')
    replicas = config['training_replicas']
    seeds = [replica['seed'] for replica in replicas]
    searches = config['search_seeds']
    if (not seeds or any(type(seed) is not int for seed in seeds + searches)
            or len(set(seeds)) != len(seeds) or not searches or len(set(searches)) != len(searches)):
        raise ValueError('Register unique integer training/search seeds')
    if len(seeds) < 2 and not config.get('engineering_smoke_only', False):
        raise ValueError('Inference needs >=2 independently trained replicas; explicitly mark one-seed runs engineering_smoke_only')
    for field in ('search_tokens', 'search_calls', 'execution_cap', 'draft_cap'):
        if type(config['budget'][field]) is not int or config['budget'][field] < 1:
            raise ValueError(f'Positive integer budget.{field} is required')
    for field in ('selection_limit', 'audit_limit'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    for field in ('selection_offset', 'audit_offset'):
        if type(config.get(field, 0)) is not int or config.get(field, 0) < 0:
            raise ValueError(f'{field} must be a nonnegative integer')
    if not config.get('resource_profile') or not config.get('initial_prompt'):
        raise ValueError('Freeze resource_profile and initial_prompt')
    external = config.get('comparison')
    if external is not None:
        planned = external.get('planned_run_ids') if isinstance(external, dict) else None
        local = {f'seed-{seed}-search-{search}-{method}' for seed in seeds for search in searches for method in methods}
        if (not isinstance(planned, list) or not all(isinstance(x, str) and x for x in planned)
                or len(planned) != len(set(planned)) or not local.issubset(planned)):
            raise ValueError('External comparison must register all local and external planned run IDs')
    for field in ('max_candidates', 'max_actions'):
        if type(config['search'][field]) is not int or config['search'][field] < 1:
            raise ValueError(f'Positive search.{field} required')
    lock_path = ROOT / config['model']['lock']
    lock = json.loads(lock_path.read_text())
    frozen_replicas = []
    distinct_weights = set()
    training_runs = set()
    for replica in replicas:
        path = (ROOT / replica['checkpoint']).resolve()
        require_role_checkpoint({'algorithm': 'vats_pdf', 'controller': 'learned', 'role_checkpoint': str(path)})
        model_config = json.loads((path / 'model_config.json').read_text())
        registered_search_config(config, learned_query_cap=semantic_input_cap(model_config))
        if (model_config.get('revision') != lock['revision']
                or (ROOT / model_config['model_path']).resolve() != (ROOT / lock['local_path']).resolve()):
            raise ValueError('All role/proposer checkpoints must use the same locked task backbone')
        hashes = checkpoint_hashes(path)
        training_path = path / 'training_config.json'
        provenance = {}
        if training_path.is_file():
            if json.loads(training_path.read_text()).get('seed') != replica['seed']:
                raise ValueError('Registered training seed disagrees with checkpoint training_config.json')
        elif not config.get('engineering_smoke_only', False):
            raise ValueError('Statistical comparison requires checkpoint training_config.json with its actual training seed')
        if not config.get('engineering_smoke_only', False):
            run_root, provenance = training_run_provenance(path, replica['seed'])
            if str(run_root) in training_runs:
                raise ValueError('Distinct training replicas require distinct completed training runs')
            training_runs.add(str(run_root))
        weights = {name: checksum for name, checksum in hashes.items()
                   if name == 'heads.pt' or (name.startswith('adapters/') and name.endswith(('.safetensors', '.bin')))}
        weight_identity = digest(weights)
        if weight_identity in distinct_weights:
            raise ValueError('Identical weights cannot be relabeled as independent training replicas')
        distinct_weights.add(weight_identity)
        frozen_replica = {**replica, 'checkpoint': str(path), 'files': hashes, 'snapshot': digest(hashes),
                          'training_provenance': provenance}
        if model_config.get('architecture_version') == 3:
            evidence = json.loads((path / 'training_evidence.json').read_text())
            frozen_replica['training_learning_signal'] = evidence.get('gvo_learning_signal')
            frozen_replica['training_completion_note'] = ('Completed stages do not imply nonzero updates in every role; '
                'legal zero-gradient GVO remains explicitly recorded and must not be promoted into actor-learning evidence.')
        frozen_replicas.append(frozen_replica)
    data_root = ROOT / config.get('data_root', 'data')
    selection_start, audit_start = config.get('selection_offset', 0), config.get('audit_offset', 0)
    selection_pool = load_examples(config['dataset'], 'selection', data_root=data_root)
    audit_pool = load_examples(config['dataset'], 'audit', data_root=data_root, verify=False)
    pools = {'selection': selection_pool, 'audit': audit_pool}
    if 'exclude_prior_campaigns' in config:
        pools['development'] = load_examples(config['dataset'], 'development', data_root=data_root, verify=False)
        validate_disjoint_splits(pools)
    pools, exclusions = exclude_prior_audit(config, output, pools)
    selection = pools['selection'][selection_start:selection_start + config['selection_limit']]
    audit = pools['audit'][audit_start:audit_start + config['audit_limit']]
    if len(selection) != config['selection_limit'] or len(audit) != config['audit_limit']:
        raise ValueError('The requested complete splits are unavailable')
    validate_disjoint_splits({'selection': selection, 'audit': audit})
    budget = config['budget']
    phases = {'initial': {'tokens': len(selection) * budget['execution_cap'], 'calls': len(selection)},
              'search': {'tokens': budget['search_tokens'], 'calls': budget['search_calls']},
              'audit': {'tokens': len(audit) * budget['execution_cap'], 'calls': len(audit)}}
    # Initial and audit allocations cannot be consumed by any optimizer.
    total_cap = sum(value['tokens'] for value in phases.values())
    split_records = {name: [example.to_dict() for example in values] for name, values in (('selection', selection), ('audit', audit))}
    order = []
    rng = random.Random(config.get('order_seed', 0))
    for seed in seeds:
        for search in searches:
            arms = list(methods)
            rng.shuffle(arms)
            order.extend({'training_seed': seed, 'search_seed': search, 'method': method} for method in arms)
    protocol = {'schema_version': 1, 'config': config, 'replicas': frozen_replicas, 'phase_caps': phases,
                'order': order, 'sources': source_hashes(), 'model_lock_sha256': file_hash(lock_path),
                'split_hashes': {name: digest(rows) for name, rows in split_records.items()},
                'method_definitions': {name: METHOD_DEFINITIONS[name] for name in methods},
                'training_compute_included': False, 'scope': 'Prospective local controls; not named paper baselines or SOTA evidence'}
    protocol['split_identities'] = {name: example_identities(values) for name, values in
                                    (('selection', selection), ('audit', audit))}
    protocol['physical_role_admission'] = {
        'v2': 'one discriminative context forward; existing token admission unchanged',
        'v3': 'reserve the registered semantic total input quota; preflight the graph schema and every possible unique-node encoding plus one global forward; charge actual cache misses and final head forward only',
        'exceptions': 'journal before invocation; retain observed receipts and unresolved conservative reservation separately; no automatic retries',
        'search_budget_is_physical_bill': False}
    if exclusions is not None:
        protocol['audit_exclusions'] = exclusions
    verify_exclusion_sources(exclusions)
    contract = {'dataset': config['dataset'], 'model_revision': lock['revision'],
                'scorer': config['dataset'] + '-' + config.get('metric_variant', 'registered-score') + '-v1',
                'permission_band': 'prompt_artifact_only', 'endpoint': 'frozen_successor_audit_mean',
                'evaluation_split': 'audit', 'selection_manifest_sha256': digest(split_records['selection']),
                'evaluation_manifest_sha256': digest(split_records['audit']),
                'initial_artifact_sha256': digest({'system_prompt': config['initial_prompt']}),
                'execution_contract_sha256': digest({'model': config['model'], 'phases': phases,
                    'execution_cap': budget['execution_cap'], 'draft_cap': budget['draft_cap'], 'methods': protocol['method_definitions']}),
                'resource_profile': config['resource_profile'],
                'budget': {'unit': 'physical_prompt_plus_completion_tokens', 'cap': total_cap,
                           'accounting': 'all initial/task/proposal/scorer/policy/audit inference; separate nontransferable phase caps',
                           'phase_caps': phases, 'call_unit': 'one_generation_or_discriminative_forward'}}
    registration = {'schema_version': 1, 'candidate': 'specrsi', 'alpha': config.get('alpha', .05),
                    'bootstrap_resamples': config.get('bootstrap_resamples', 10000), 'random_seed': config.get('analysis_seed', 0),
                    'protocol_sha256': digest(protocol), 'engineering_smoke_only': config.get('engineering_smoke_only', False),
                    'profiles': [{'id': config.get('profile_id', config['dataset']),
                        'baselines': [method for method in methods if method != 'specrsi'],
                        'training_seeds': seeds, 'search_seeds': searches, 'direction': 'higher',
                        'minimum_effect': config.get('minimum_effect', 0.), 'contract': contract}]}
    validate_comparison_registration(registration)
    protocol['comparison_registration_sha256'] = digest({key: value for key, value in registration.items() if key != 'protocol_sha256'})
    registration['protocol_sha256'] = digest(protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'protocol.json', protocol)
    write_new(output / 'registration.json', registration)
    for name, rows in split_records.items():
        write_new(output / f'{name}_examples.json', rows)
    return {'registration': str(output / 'registration.json'), 'planned_runs': len(order), 'paid_model_calls': 0,
            'scope': protocol['scope']}


class PhysicalBudgetStop(RuntimeError):
    """No physical call has started; keep the last complete verified successor."""


def validate_encoder_receipt(row):
    """Cached frozen text saves encoder tokens, never the executed head call."""
    keys = ('encoder_forward_count', 'head_forward_count', 'encoder_cache_hit')
    if not any(key in row for key in keys):
        return
    encoder, head, hit = (row.get(key) for key in keys)
    if (type(encoder) is not int or encoder not in (0, 1) or type(head) is not int or head not in (0, 1) or
            type(hit) is not bool or row.get('completion_tokens') != 0):
        raise ValueError('Invalid encoder/head physical receipt')
    if row.get('role') == 'frozen_node_text':
        if (encoder, head, hit) != (1, 0, False):
            raise ValueError('A node receipt must represent an actual encoder miss')
        ids = row.get('input_token_ids')
    elif row.get('role') in {'scorer', 'policy'}:
        ids = row.get('context_input_token_ids')
        if row.get('status', 'complete') == 'complete' and (encoder, head, hit) not in {(1, 1, False), (0, 1, True)}:
            raise ValueError('Complete role receipt must include its head invocation')
        if hit and (encoder or not row.get('encoder_snapshot')):
            raise ValueError('Cached encoder receipt requires an immutable encoder identity')
        if not encoder and (row.get('prompt_tokens') != 0 or 'input_token_ids' in row):
            raise ValueError('A skipped encoder cannot claim input tokens')
        if encoder and (hit or row.get('input_token_ids') != ids):
            raise ValueError('Encoder miss input IDs differ from role conditioning')
    else:
        raise ValueError('Encoder metadata attached to an incompatible role')
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError('Encoder receipt lacks exact nonempty conditioning token IDs')
    if encoder and row.get('prompt_tokens') != len(ids):
        raise ValueError('Encoder token bill differs from executed token IDs')


class PhysicalMeter:
    def __init__(self, caps, path, *, before_call=None):
        self.caps, self.path, self.phase = caps, Path(path), 'initial'
        self.spent = {phase: {'tokens': 0, 'calls': 0} for phase in caps}
        self.valid = True
        self.records = []
        self.path.touch(exist_ok=False)
        self.journal_path = self.path.with_name(self.path.stem + '_operations.jsonl')
        self.journal_path.touch(exist_ok=False)
        self.pending = None
        self.operation_count = 0
        self.before_call = before_call
        self.unknown_reservations = []

    def check(self, tokens, calls=1):
        if not self.valid:
            raise RuntimeError('An unmetered failure invalidated this run')
        used, caps = self.spent[self.phase], self.caps[self.phase]
        if used['tokens'] + tokens > caps['tokens'] or used['calls'] + calls > caps['calls']:
            raise PhysicalBudgetStop(f'{self.phase} physical token/call cap reached before starting a call')

    def record(self, value):
        row = {'phase': self.phase, **value}
        self.records.append(row)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(canonical(row) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def journal(self, value):
        with self.journal_path.open('a', encoding='utf-8') as handle:
            handle.write(canonical(value) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def begin(self, tokens, calls, role):
        if self.pending is not None:
            raise RuntimeError('An unresolved physical operation cannot be retried')
        if self.before_call is not None:
            self.before_call()
        self.check(tokens, calls)
        self.operation_count += 1
        self.pending = {'operation_id': self.operation_count, 'phase': self.phase, 'role': role,
            'reserved_tokens': tokens, 'reserved_calls': calls,
            'before_tokens': self.spent[self.phase]['tokens'], 'before_calls': self.spent[self.phase]['calls']}
        self.journal({**self.pending, 'status': 'pending', 'created_at_unix': time.time()})

    def settle(self, *, status='complete', unknown=False, error=None):
        if self.pending is None:
            raise RuntimeError('No admitted operation to settle')
        pending = self.pending
        tokens = self.spent[self.phase]['tokens'] - pending['before_tokens']
        calls = self.spent[self.phase]['calls'] - pending['before_calls']
        value = {**pending, 'status': status, 'measured_tokens': tokens, 'measured_calls': calls,
            'unresolved_reserved_tokens': max(0, pending['reserved_tokens'] - tokens) if unknown else 0,
            'unresolved_reserved_calls': max(0, pending['reserved_calls'] - calls) if unknown else 0,
            'created_at_unix': time.time()}
        if error is not None:
            value['error_type'] = type(error).__name__
        self.journal(value)
        if unknown:
            self.unknown_reservations.append(value)
        self.pending = None

    def charge(self, rows, cap):
        valid_rows = [row for row in rows if isinstance(row, dict) and all(
            type(row.get(key)) is int and row[key] >= 0 for key in ('prompt_tokens', 'completion_tokens'))]
        tokens = sum(row['prompt_tokens'] + row['completion_tokens'] for row in valid_rows)
        self.spent[self.phase]['tokens'] += tokens
        self.spent[self.phase]['calls'] += len(valid_rows)
        for row in valid_rows:
            self.record({'cost_kind': 'actual', **row})
        if len(valid_rows) != len(rows):
            self.valid = False
            raise RuntimeError('Invalid physical token receipt; its expense remains unresolved')
        try:
            for row in valid_rows:
                validate_encoder_receipt(row)
        except ValueError as error:
            self.valid = False
            raise RuntimeError(str(error)) from error
        if (tokens > cap or self.spent[self.phase]['tokens'] > self.caps[self.phase]['tokens']
                or self.spent[self.phase]['calls'] > self.caps[self.phase]['calls']):
            self.valid = False
            raise RuntimeError('Physical metering exceeded the admitted token/call contract')
        return tokens

    def failed(self, role, error, *, unknown=True):
        self.valid = False
        self.journal({'phase': self.phase, 'role': role,
            'status': 'unmetered_failure' if unknown else 'metered_failure',
            'error_type': type(error).__name__, 'cost_kind': 'unknown' if unknown else 'failure_metadata'})
        if self.pending is not None:
            self.settle(status='failed', unknown=unknown, error=error)

    @classmethod
    def restore_search(cls, caps, path, frozen, *, before_call=None):
        """Resume only an entirely settled, hash-bound selection phase."""
        meter = cls.__new__(cls)
        meter.caps, meter.path, meter.phase = caps, Path(path), 'audit'
        meter.journal_path = meter.path.with_name(meter.path.stem + '_operations.jsonl')
        if (file_hash(meter.path) != frozen['search_receipts_sha256'] or
                file_hash(meter.journal_path) != frozen['search_operations_sha256']):
            raise ValueError('Frozen physical selection records changed before audit')
        with meter.path.open(encoding='utf-8') as handle:
            meter.records = [json.loads(line) for line in handle if line.strip()]
        meter.spent = {phase: {'tokens': 0, 'calls': 0} for phase in caps}
        for row in meter.records:
            if (row.get('phase') not in {'initial', 'search'} or row.get('cost_kind') != 'actual' or
                    any(type(row.get(k)) is not int or row[k] < 0 for k in ('prompt_tokens', 'completion_tokens'))):
                raise ValueError('Invalid frozen search receipt')
            validate_encoder_receipt(row)
            meter.spent[row['phase']]['tokens'] += row['prompt_tokens'] + row['completion_tokens']
            meter.spent[row['phase']]['calls'] += 1
        if meter.spent != frozen['phase_receipts'] or caps != frozen['phase_caps']:
            raise ValueError('Frozen selection accounting differs from physical receipts or registered caps')
        with meter.journal_path.open(encoding='utf-8') as handle:
            operations = [json.loads(line) for line in handle if line.strip()]
        pending, last_id, receipt_index = None, 0, 0
        settled = {phase: {'tokens': 0, 'calls': 0} for phase in caps}
        identity_keys = ('operation_id', 'phase', 'role', 'reserved_tokens', 'reserved_calls',
                         'before_tokens', 'before_calls')
        for row in operations:
            if row.get('phase') not in {'initial', 'search'}:
                raise ValueError('Audit work already exists or invalid physical phase')
            if row.get('status') == 'pending':
                if pending is not None or row['operation_id'] != last_id + 1:
                    raise ValueError('Invalid frozen physical operation order')
                if (any(type(row.get(k)) is not int or row[k] < 0 for k in
                        ('reserved_tokens', 'reserved_calls', 'before_tokens', 'before_calls')) or
                        row['before_tokens'] != settled[row['phase']]['tokens'] or
                        row['before_calls'] != settled[row['phase']]['calls'] or
                        row['before_tokens'] + row['reserved_tokens'] > caps[row['phase']]['tokens'] or
                        row['before_calls'] + row['reserved_calls'] > caps[row['phase']]['calls']):
                    raise ValueError('Invalid frozen physical admission counters')
                pending = row
            elif row.get('status') == 'complete':
                if pending is None or row['operation_id'] != pending['operation_id']:
                    raise ValueError('Unmatched physical operation settlement')
                if (any(row.get(k) != pending.get(k) for k in identity_keys) or
                        any(type(row.get('measured_' + k)) is not int or
                            not 0 <= row['measured_' + k] <= row['reserved_' + k] for k in ('tokens', 'calls'))):
                    raise ValueError('Physical settlement differs from admission')
                if row.get('unresolved_reserved_tokens') or row.get('unresolved_reserved_calls'):
                    raise ValueError('Unknown selection charges cannot resume into audit')
                actual = meter.records[receipt_index:receipt_index + row['measured_calls']]
                if (len(actual) != row['measured_calls'] or any(r['phase'] != row['phase'] for r in actual) or
                        sum(r['prompt_tokens'] + r['completion_tokens'] for r in actual) != row['measured_tokens']):
                    raise ValueError('Physical settlement differs from inference receipts')
                receipt_index += row['measured_calls']
                for kind in ('tokens', 'calls'):
                    settled[row['phase']][kind] += row['measured_' + kind]
                last_id, pending = row['operation_id'], None
            else:
                raise ValueError('Failed or unmetered selection cannot resume into audit')
        if (pending is not None or last_id != frozen['operation_count'] or
                receipt_index != len(meter.records) or settled != meter.spent):
            raise ValueError('Incomplete physical selection operation')
        for phase, value in meter.spent.items():
            if any(value[k] > caps[phase][k] for k in ('tokens', 'calls')):
                raise ValueError('Frozen selection exceeds registered physical caps')
        meter.valid, meter.pending, meter.unknown_reservations = True, None, []
        meter.operation_count, meter.before_call = last_id, before_call
        return meter


class MeteredClient:
    def __init__(self, client, tokenizer, revision, meter):
        self.client, self.tokenizer, self.revision, self.meter = client, tokenizer, revision, meter

    def generate(self, messages, **kwargs):
        count = len(chat_token_ids(self.tokenizer,messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=kwargs.get('enable_thinking', False), thinking=kwargs.get('enable_thinking', False)))
        cap = count + kwargs.get('max_tokens', 512)
        if cap > kwargs.get('total_token_cap', cap):
            raise ValueError('The complete task generation does not fit its registered execution cap')
        self.meter.begin(cap, 1, 'task')
        try:
            reply = self.client.generate(messages, **{**kwargs, 'total_token_cap': cap})
        except Exception as error:
            self.meter.failed('task', error)
            raise
        metered = False
        try:
            self.meter.charge([{'role': 'task', **asdict(reply)}], cap)
            metered = True
            if reply.model_revision != self.revision or reply.prompt_tokens != count:
                raise RuntimeError('Frozen task model/tokenizer identity changed')
        except Exception as error:
            self.meter.failed('task', error, unknown=not metered)
            raise
        self.meter.settle()
        return reply


def measured_role(bridge, meter, function, cap, role, *, calls=1):
    meter.begin(cap, calls, role)
    before = len(bridge.physical_inferences)
    try:
        result = function()
    except Exception as error:
        # v3 may have completed several real node forwards before a later
        # encoder/head fails. Those receipts must survive the missing return.
        rows = bridge.physical_inferences[before:]
        try:
            meter.charge(rows, cap)
        except Exception as billing_error:
            meter.failed(role, billing_error)
            raise billing_error from error
        meter.failed(role, error)
        if isinstance(error, PhysicalBudgetStop):
            raise RuntimeError('An operation failed after admission; not a safe pre-call budget stop') from error
        raise
    rows = bridge.physical_inferences[before:]
    try:
        meter.charge(rows, cap)
        v3 = getattr(getattr(bridge, 'model', None), 'config', None)
        v3 = getattr(v3, 'architecture_version', 2) == 3 and role in {'scorer', 'policy'}
        valid = (bool(rows) and rows[-1].get('role') == role and
                 all(row.get('role') == 'frozen_node_text' for row in rows[:-1])) if v3 else len(rows) == 1
        no_draft = role == 'draft' and not rows and result.actual_cost == 0
        if not no_draft and (not valid or len(rows) > calls):
            raise RuntimeError('Admitted role operation returned incompatible physical inference receipts')
        if any(row.get('status', 'complete') != 'complete' for row in rows):
            raise RuntimeError('Role returned despite a failed physical inference receipt')
    except Exception as error:
        meter.failed(role, error)
        raise
    meter.settle()
    return result


def role_admission(bridge, state):
    """A conservative no-forward admission bound, independent of cache state.

    v3 splits the registered semantic input cap across common and unique node
    inputs. Repeated candidate-task nodes reuse identical input encodings.
    Reserving every possible miss also bounds the physical call count before
    any cache lookup can execute a backbone forward.
    """
    if getattr(bridge.model.config, 'architecture_version', 2) == 3:
        from specrsi.learning.work_graph import graph_schema, ACTUAL_GRAPH_SCHEMA
        calls = (bridge.semantic_forward_cap(state) if (graph_schema(bridge.model.config) == ACTUAL_GRAPH_SCHEMA or
                 getattr(bridge.model.config, 'role_observation_version', 'legacy_v1') == 'observed_results_v2')
                 else len(state.candidates) + len(state.scope.tasks) + 1)
        return semantic_input_cap(bridge.model.config), calls
    return bridge.inference_cost(bridge.context_text(state)), 1


class MeteredDrafter:
    def __init__(self, inner, bridge, meter):
        self.inner, self.bridge, self.meter = inner, bridge, meter

    def draft(self, parent, state, cap, rng):
        before = self.meter.spent[self.meter.phase]['tokens']
        result = measured_role(self.bridge, self.meter, lambda: self.inner.draft(parent, state, cap, rng), cap, 'draft')
        if result.actual_cost != self.meter.spent[self.meter.phase]['tokens'] - before:
            self.meter.valid = False
            raise RuntimeError('Draft physical records disagree with the actual charged tokens')
        return result


def task_of(example, metric_variant=None):
    return Task(example.id, {'prompt': example.prompt}, example.reference,
                {'dataset': example.dataset, 'content_hash': example.content_hash, 'split': example.split,
                 'metric_variant': metric_variant})


def optimize_arm(config, protocol, replica, row, model, client, tokenizer, lock, selection, directory):
    """Only selection examples enter this function; audit bytes are loaded later."""
    from specrsi.learning.paper_bridge import VATSSearchBridge
    directory.mkdir(parents=True, exist_ok=False)
    meter = PhysicalMeter(protocol['phase_caps'], directory / 'physical_calls.jsonl',
                          before_call=lambda: verify_exclusion_sources(protocol.get('audit_exclusions')))
    metered_client = MeteredClient(client, tokenizer, lock['revision'], meter)
    executor = QwenExecutor(metered_client, tokenizer, config['model'])
    seed = int(digest([row['training_seed'], row['search_seed']])[:8], 16)
    root = Candidate('initial', {'system_prompt': config['initial_prompt']},
                     {'model_revision': lock['revision'], 'seed': seed, 'generation': config['model']}, 'opaque', False)
    tasks = tuple(task_of(example, config.get('metric_variant')) for example in selection)
    scope = EvaluationScope('registered-' + config['dataset'], tasks, 'qwen-capped-generation-v2',
                            config['dataset'] + '-' + config.get('metric_variant', 'registered-score') + '-v1',
                            scored_fields=('answer', 'metrics', 'finish_reason'),
                            replay={'generation': config['model'], 'model_revision': lock['revision']})
    ledger = Ledger()
    execution_cap = config['budget']['execution_cap']
    executor.validate_prompt_candidate(root, tasks, execution_cap)
    initial = []
    for task in tasks:
        acquired = executor.run(root, task, scope, execution_cap)
        if acquired.status != 'complete':
            raise RuntimeError('The common initial candidate must have complete native outcomes')
        ledger.admit_direct(root, task, scope, acquired.outcome, acquired.receipt)
        initial.append({'task': task.uid, 'score': acquired.outcome.score, 'fields': thaw(acquired.outcome.fields)})
    write_new(directory / 'initial.json', initial)
    meter.phase = 'search'
    bridge = VATSSearchBridge(model, cost_unit='tokens', snapshot_id=replica['snapshot'])
    drafter = MeteredDrafter(make_drafter(bridge, config['search'],
        lambda candidate: executor.validate_prompt_candidate(candidate, tasks, execution_cap)), bridge, meter)
    budget = BudgetLedger(config['budget']['search_tokens'])
    options = registered_search_config(config, learned_query_cap=semantic_input_cap(model.config),
                                       seed=seed, snapshot=replica['snapshot'])
    state = VATSState(scope, root, {root.uid: root}, ledger, budget,
                      remaining_attempts=options.max_attempts, initial_attempts=options.max_attempts, workers=options.workers)
    summary, stop = {}, 'initial_control' if row['method'] == 'initial' else 'candidate_limit'
    try:
        if row['method'] == 'specrsi':
            def scorer(state, plan):
                count, calls = role_admission(bridge, state)
                return measured_role(bridge, meter, lambda: bridge.scorer(state, plan), count, 'scorer', calls=calls)

            def policy(state, plans):
                count, calls = role_admission(bridge, state)
                return measured_role(bridge, meter, lambda: bridge.policy_menu(state, plans), count, 'policy', calls=calls)

            engine = VATSEngine(scope, root, ledger, budget, executor, drafter, options, scorer=scorer, policy_menu=policy)
            state = engine.state
            result = engine.run()
            summary = result.to_dict()
            stop = result.stop_reason
        elif row['method'] != 'initial':
            rng = random.Random(seed)
            for _ in range(min(options.max_actions, options.max_candidates - 1)):
                parents = [candidate for candidate in state.candidates.values() if ledger.admissible(candidate, scope)]
                parent = root if row['method'] == 'best_of_n' else rng.choice(parents)
                proposal_state = state
                if row['method'] == 'best_of_n':
                    # A root best-of-N control samples from the same frozen
                    # proposal context instead of conditioning on earlier trials.
                    proposal_state = replace(state, candidates={root.uid: root},
                        budget=BudgetLedger(config['budget']['search_tokens']), history=[], children_drafted={})
                proposal = drafter.draft(parent, proposal_state, options.draft_cap, rng)
                state.budget.spent = meter.spent['search']['tokens']
                state.version += 1
                state.children_drafted[parent.uid] = state.children_drafted.get(parent.uid, 0) + 1
                if proposal.status != 'complete' or proposal.candidate is None:
                    continue
                child = proposal.candidate
                if child.uid in state.candidates:
                    continue
                state.candidates[child.uid] = child
                for task in tasks:
                    acquired = executor.run(child, task, scope, execution_cap)
                    state.budget.spent = meter.spent['search']['tokens']
                    if acquired.status == 'complete':
                        ledger.admit_direct(child, task, scope, acquired.outcome, acquired.receipt)
                    else:
                        break
    except PhysicalBudgetStop:
        # This exception is raised before starting work, so no unknown physical
        # expense is hidden. Return the best candidate whose checks already exist.
        stop = 'physical_budget_exhausted_before_call'
        summary['internal_contract_interrupted'] = True
    outcome = accept(root, state.candidates.values(), ledger, scope)
    summary.update({'method': row['method'], 'stop_reason': stop, 'selected_id': outcome.selected.uid,
                    'initial_score': outcome.incumbent_score, 'selected_score': outcome.selected_score,
                    'physical_search_tokens': meter.spent['search']['tokens'],
                    'physical_search_calls': meter.spent['search']['calls'],
                    'internal_contract_cost_is_physical_bill': False, 'ledger': json.loads(ledger.to_json())})
    write_new(directory / 'search.json', summary)
    write_new(directory / 'selected.json', outcome.selected.manifest())
    return {'row': row, 'directory': directory, 'meter': meter, 'executor': executor,
            'selected': outcome.selected, 'initial': initial, 'scope': scope, 'summary': summary}


def verify_external_barrier(config, directory, *, path=None, expected_sha256=None):
    comparison = config.get('comparison')
    if comparison is None:
        if path is not None or expected_sha256 is not None:
            raise ValueError('External barrier must be prospectively registered')
        return
    if path is None or expected_sha256 is None:
        raise ValueError('Registered cross-method audit requires a complete frozen barrier')
    path = Path(path).resolve()
    if file_hash(path) != expected_sha256:
        raise ValueError('Cross-method barrier identity changed')
    barrier = json.loads(path.read_text())
    planned = comparison['planned_run_ids']
    if (barrier.get('schema') != 'cross_method_selection_barrier_v1' or
            barrier.get('planned_run_ids') != planned or set(barrier.get('runs', {})) != set(planned)):
        raise ValueError('Cross-method barrier does not cover the exact registered run set')
    own_ids = {f"seed-{r['seed']}-search-{s}-{m}" for r in config['training_replicas']
               for s in config['search_seeds'] for m in config.get('methods', list(METHODS))}
    seen_paths = set()
    for run_id, entry in barrier['runs'].items():
        for key in ('selection_frozen', 'selected'):
            artifact = Path(entry[key]['path'])
            if not artifact.is_absolute() or file_hash(artifact) != entry[key]['sha256']:
                raise ValueError('Cross-method frozen candidate or selection receipt changed')
            if artifact.resolve() in seen_paths:
                raise ValueError('Distinct registered runs must have distinct frozen artifact paths')
            seen_paths.add(artifact.resolve())
            own = directory / 'runs' / run_id / (key + '.json')
            if run_id in own_ids:
                if artifact.resolve() != own.resolve():
                    raise ValueError('Barrier substituted a different local run artifact')


def freeze_search_context(context, protocol, input_hashes):
    directory, meter = context['directory'], context['meter']
    if not meter.valid or meter.pending is not None or meter.unknown_reservations:
        raise ValueError('Only complete, metered optimization can freeze a successor')
    frozen = {'schema': 'registered_selection_frozen_v1', 'run_id': directory.name,
        'created_at_unix': time.time(), 'row': context['row'], 'protocol_sha256': digest(protocol),
        'input_hashes': input_hashes, 'selected_sha256': file_hash(directory / 'selected.json'),
        'initial_sha256': file_hash(directory / 'initial.json'), 'search_sha256': file_hash(directory / 'search.json'),
        'search_receipts_sha256': file_hash(meter.path), 'search_operations_sha256': file_hash(meter.journal_path),
        'search_receipts_bytes': meter.path.stat().st_size, 'search_operations_bytes': meter.journal_path.stat().st_size,
        'phase_caps': meter.caps, 'phase_receipts': meter.spent, 'operation_count': meter.operation_count}
    write_new(directory / 'selection_frozen.json', frozen)


def restore_search_context(directory, config, protocol, row, selection, client, tokenizer, lock, input_hashes):
    frozen = json.loads((directory / 'selection_frozen.json').read_text())
    if (frozen.get('schema') != 'registered_selection_frozen_v1' or frozen['run_id'] != directory.name or
            frozen['row'] != row or frozen['protocol_sha256'] != digest(protocol) or frozen['input_hashes'] != input_hashes):
        raise ValueError('Frozen selection context does not match registered inputs')
    for filename, key in [('selected.json', 'selected_sha256'), ('initial.json', 'initial_sha256'), ('search.json', 'search_sha256')]:
        if file_hash(directory / filename) != frozen[key]:
            raise ValueError('Frozen selection artifact changed before audit')
    meter = PhysicalMeter.restore_search(protocol['phase_caps'], directory / 'physical_calls.jsonl', frozen,
        before_call=lambda: verify_exclusion_sources(protocol.get('audit_exclusions')))
    executor = QwenExecutor(MeteredClient(client, tokenizer, lock['revision'], meter), tokenizer, config['model'])
    scope = EvaluationScope('registered-' + config['dataset'],
        tuple(task_of(e, config.get('metric_variant')) for e in selection), 'qwen-capped-generation-v2',
        config['dataset'] + '-' + config.get('metric_variant', 'registered-score') + '-v1',
        scored_fields=('answer', 'metrics', 'finish_reason'),
        replay={'generation': config['model'], 'model_revision': lock['revision']})
    return {'row': row, 'directory': directory, 'meter': meter, 'executor': executor,
        'selected': Candidate(**json.loads((directory / 'selected.json').read_text())), 'scope': scope,
        'initial': json.loads((directory / 'initial.json').read_text()),
        'summary': json.loads((directory / 'search.json').read_text())}


def execute(directory, *, phase='run', expected_frozen_sha256=None, barrier_path=None, barrier_sha256=None):
    if phase not in {'run', 'search-only', 'audit-only'}:
        raise ValueError('Unknown registered execution phase')
    directory = Path(directory).resolve()
    protocol = json.loads((directory / 'protocol.json').read_text())
    registration = json.loads((directory / 'registration.json').read_text())
    input_hashes = {name: file_hash(directory / name) for name in
                    ('protocol.json', 'registration.json', 'selection_examples.json', 'audit_examples.json')}
    config = protocol['config']
    if config.get('comparison') is not None and phase == 'run':
        raise ValueError('Cross-method comparisons must search-only, freeze all methods, then audit-only')
    if phase == 'audit-only':
        if not expected_frozen_sha256 or file_hash(directory / 'all_successors_frozen.json') != expected_frozen_sha256:
            raise ValueError('Audit requires the caller-held frozen successor manifest SHA256')
        verify_external_barrier(config, directory, path=barrier_path, expected_sha256=barrier_sha256)
    if ('exclude_prior_campaigns' in config) != ('audit_exclusions' in protocol):
        raise ValueError('Registered historical audit exclusion contract is missing or inconsistent')
    verify_exclusion_sources(protocol.get('audit_exclusions'))
    registration_body = {key: value for key, value in registration.items() if key != 'protocol_sha256'}
    if (registration['protocol_sha256'] != digest(protocol)
            or digest(registration_body) != protocol['comparison_registration_sha256']
            or source_hashes() != protocol['sources']):
        raise ValueError('Frozen comparison protocol or source changed; create a new prospective registration')
    if file_hash(ROOT / config['model']['lock']) != protocol['model_lock_sha256']:
        raise ValueError('Task backbone lock changed after registration')
    splits = {}
    for name in ('selection', 'audit'):
        raw = json.loads((directory / f'{name}_examples.json').read_text())
        if digest(raw) != protocol['split_hashes'][name]:
            raise ValueError(f'Frozen {name} examples changed')
        # Audit is verified here but is not passed into any optimization call.
        splits[name] = [BenchmarkExample.from_dict(value) for value in raw]
        if ('split_identities' in protocol and
                example_identities(splits[name]) != protocol['split_identities'].get(name)):
            raise ValueError('Frozen input identities disagree with registered examples')
    validate_disjoint_splits(splits)
    for replica in protocol['replicas']:
        if (checkpoint_hashes(Path(replica['checkpoint'])) != replica['files']
                or any(file_hash(path) != checksum for path, checksum in replica['training_provenance'].items())):
            raise ValueError('Role checkpoint changed after registration')
    (directory / ('.audit_started' if phase == 'audit-only' else '.started')).touch(exist_ok=False)
    started = time.monotonic()
    pending = []
    try:
        client, tokenizer, lock, health = load_backbone(config)
        if phase == 'audit-only':
            manifest = json.loads((directory / 'all_successors_frozen.json').read_text())
            expected_ids = {f"runs/seed-{r['training_seed']}-search-{r['search_seed']}-{r['method']}" for r in protocol['order']}
            if set(manifest.get('selection_frozen', {})) != expected_ids or set(manifest.get('selected', {})) != expected_ids:
                raise ValueError('Incomplete frozen optimization set')
            for relative, value in manifest['selection_frozen'].items():
                if file_hash(directory / relative / 'selection_frozen.json') != value:
                    raise ValueError('Frozen optimization receipt changed')
                if file_hash(directory / relative / 'selected.json') != manifest['selected'][relative]:
                    raise ValueError('Selected candidate changed')
            for row in protocol['order']:
                run_id = f"seed-{row['training_seed']}-search-{row['search_seed']}-{row['method']}"
                pending.append(restore_search_context(directory / 'runs' / run_id, config, protocol, row,
                    splits['selection'], client, tokenizer, lock, input_hashes))
        else:
            for replica in protocol['replicas']:
                model = load_roles(replica['checkpoint'], config.get('role_device', 'cuda:1'))
                for row in protocol['order']:
                    if row['training_seed'] != replica['seed']:
                        continue
                    run_id = f"seed-{row['training_seed']}-search-{row['search_seed']}-{row['method']}"
                    run_dir = directory / 'runs' / run_id
                    context = optimize_arm(config, protocol, replica, row, model, client, tokenizer, lock,
                                           splits['selection'], run_dir)
                    freeze_search_context(context, protocol, input_hashes)
                    pending.append(context)
                del model
        # Matching initial artifacts alone does not guarantee matching stochastic
        # outcomes. Refuse different root observations across paired methods.
        roots = {}
        for context in pending:
            key = (context['row']['training_seed'], context['row']['search_seed'])
            observed = digest(context['initial'])
            if key in roots and roots[key] != observed:
                raise ValueError('Paired arms received different initial observations under the same frozen root')
            roots[key] = observed
        if source_hashes() != protocol['sources']:
            raise ValueError('Sources changed during optimization; audit remains untouched')
        verify_exclusion_sources(protocol.get('audit_exclusions'))
        selected_hashes = {str(context['directory'].relative_to(directory)): file_hash(context['directory'] / 'selected.json')
                           for context in pending}
        if phase != 'audit-only':
            write_new(directory / 'all_successors_frozen.json', {'selected': selected_hashes,
                'selection_frozen': {str(c['directory'].relative_to(directory)): file_hash(c['directory'] / 'selection_frozen.json')
                                     for c in pending}})
        frozen_manifest_sha = file_hash(directory / 'all_successors_frozen.json')
        if phase == 'search-only':
            return {'output': str(directory), 'status': 'all_successors_frozen', 'frozen_candidates': len(pending),
                    'selection_frozen_sha256': frozen_manifest_sha, 'heldout_model_calls': 0}
        if phase == 'audit-only':
            for context in pending:
                def before_audit_call():
                    verify_exclusion_sources(protocol.get('audit_exclusions'))
                    verify_external_barrier(config, directory, path=barrier_path, expected_sha256=barrier_sha256)
                context['meter'].before_call = before_audit_call
        records = []
        for context in pending:
            meter, executor, selected = context['meter'], context['executor'], context['selected']
            meter.phase = 'audit'
            tasks = tuple(task_of(example, config.get('metric_variant')) for example in splits['audit'])
            previous = context['scope']
            scope = EvaluationScope('audit-' + config['dataset'], tasks, previous.execution_id, previous.scorer_id,
                                    scored_fields=previous.scored_fields, replay=previous.replay, split='audit')
            audit = []
            for task in tasks:
                acquired = executor.run(selected, task, scope, config['budget']['execution_cap'])
                audit.append({'task': task.uid, 'status': acquired.status,
                              'score': acquired.outcome.score if acquired.outcome else None,
                              'actual_tokens': acquired.actual_cost})
            write_new(context['directory'] / 'audit.json', audit)
            if not meter.valid or any(row['status'] != 'complete' for row in audit):
                raise RuntimeError('Incomplete or unmetered audit; no complete-case comparison is permitted')
            tokens = sum(value['tokens'] for value in meter.spent.values())
            raw = {'status': 'complete', 'audit_mean': sum(row['score'] for row in audit) / len(audit),
                   'physical_tokens': tokens, 'phase_receipts': meter.spent, 'phase_caps': meter.caps,
                   'actual_model_calls': sum(value['calls'] for value in meter.spent.values()),
                   'selected_sha256': selected_hashes[str(context['directory'].relative_to(directory))],
                   'audit_sha256': file_hash(context['directory'] / 'audit.json'),
                   'physical_calls_sha256': file_hash(meter.path), 'model_health': health,
                   'physical_operations_sha256': file_hash(meter.journal_path),
                   'unresolved_physical_reservations': meter.unknown_reservations,
                   'training_compute_included': False, 'scope': protocol['scope']}
            run_path = context['directory'] / 'run.json'
            write_new(run_path, raw)
            records.append({**context['row'], 'profile_id': registration['profiles'][0]['id'],
                'run_id': context['directory'].name, 'status': 'complete', 'contract': registration['profiles'][0]['contract'],
                'evidence': {'score': {'path': str((context['directory'] / 'audit.json').relative_to(directory)),
                    'sha256': file_hash(context['directory'] / 'audit.json'), 'pointer': '', 'item_pointer': '/score', 'reduction': 'mean'},
                    'actual_cost': {'path': str(run_path.relative_to(directory)), 'sha256': file_hash(run_path), 'pointer': '/physical_tokens'}}})
        if source_hashes() != protocol['sources']:
            raise ValueError('Sources changed during audit; this campaign cannot support a frozen comparison')
        verify_exclusion_sources(protocol.get('audit_exclusions'))
        if phase == 'audit-only':
            verify_external_barrier(config, directory, path=barrier_path, expected_sha256=barrier_sha256)
        frozen_all = json.loads((directory / 'all_successors_frozen.json').read_text())
        for context in pending:
            run_dir, meter = context['directory'], context['meter']
            relative = str(run_dir.relative_to(directory))
            if file_hash(run_dir / 'selection_frozen.json') != frozen_all['selection_frozen'][relative]:
                raise ValueError('Frozen search descriptor changed during audit')
            frozen = json.loads((run_dir / 'selection_frozen.json').read_text())
            if (file_hash(run_dir / 'initial.json') != frozen['initial_sha256'] or
                    file_hash(run_dir / 'search.json') != frozen['search_sha256'] or
                    prefix_hash(meter.path, frozen['search_receipts_bytes']) != frozen['search_receipts_sha256'] or
                    prefix_hash(meter.journal_path, frozen['search_operations_bytes']) != frozen['search_operations_sha256']):
                raise ValueError('Frozen search evidence changed during audit')
        if (file_hash(ROOT / config['model']['lock']) != protocol['model_lock_sha256']
                or any(checkpoint_hashes(Path(replica['checkpoint'])) != replica['files']
                       or any(file_hash(path) != checksum for path, checksum in replica['training_provenance'].items())
                       for replica in protocol['replicas'])):
            raise ValueError('Frozen model/checkpoint files changed during the campaign')
        if (any(file_hash(directory / name) != checksum for name, checksum in input_hashes.items())
                or file_hash(directory / 'all_successors_frozen.json') != frozen_manifest_sha
                or any(file_hash(directory / name / 'selected.json') != checksum for name, checksum in selected_hashes.items())):
            raise ValueError('Registered inputs or frozen selected artifacts changed during the campaign')
        with (directory / 'runs.jsonl').open('x', encoding='utf-8') as handle:
            for row in records:
                handle.write(canonical(row) + '\n')
        spec = importlib.util.spec_from_file_location('registered_metric_reader', ROOT / 'scripts/compare_methods.py')
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        observed, verified = reader.load_observed_records(directory / 'runs.jsonl')
        if config.get('engineering_smoke_only', False):
            report = {'all_registered_comparisons_supported': False, 'engineering_smoke_only': True,
                      'scope': 'Engineering execution check only; no statistical superiority or SOTA conclusion',
                      'observed_runs': len(observed)}
        else:
            report = compare_registered_methods(registration, observed)
        ranking = []
        for method in config.get('methods', list(METHODS)):
            arm = [row for row in observed if row['method'] == method]
            ranking.append({'method': method, 'audit_mean': sum(row['score'] for row in arm) / len(arm),
                            'mean_physical_tokens': sum(row['actual_cost'] for row in arm) / len(arm),
                            'runs': len(arm)})
        ranking.sort(key=lambda row: (-row['audit_mean'], row['method']))
        for row in ranking:
            row['rank'] = 1 + sum(other['audit_mean'] > row['audit_mean'] for other in ranking)
        report['descriptive_ranking'] = ranking
        report['ranking_interpretation'] = 'Observed audit means only; nested searches are not independent training seeds and this table is not a significance claim.'
        report.update(verified_source_hashes=verified, registration_sha256=file_hash(directory / 'registration.json'),
                      protocol_sha256=digest(protocol), registered_input_hashes=input_hashes,
                      frozen_successors_sha256=frozen_manifest_sha, elapsed_seconds=time.monotonic() - started)
        write_new(directory / 'comparison.json', report)
        return {'output': str(directory), 'completed_runs': len(records),
                'all_registered_comparisons_supported': report['all_registered_comparisons_supported'], 'scope': report['scope']}
    except Exception as error:
        write_new(directory / 'failure.json', {'complete_campaign': False, 'error_type': type(error).__name__,
                  'error': str(error), 'finished_optimizations': len(pending), 'elapsed_seconds': time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('register', help='Freeze inputs before any model generation')
    plan.add_argument('--config', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run', help='Execute an unchanged registration once')
    run.add_argument('directory', type=Path)
    search = sub.add_parser('search-only', help='Optimize every local arm and freeze all successors without heldout calls')
    search.add_argument('directory', type=Path)
    audit = sub.add_parser('audit-only', help='Audit a complete frozen optimization set without repeating search')
    audit.add_argument('directory', type=Path)
    audit.add_argument('--expected-frozen-sha256', required=True)
    audit.add_argument('--barrier', type=Path)
    audit.add_argument('--barrier-sha256')
    args = parser.parse_args()
    if args.action == 'register':
        result = register(yaml.safe_load(args.config.read_text()), args.output)
    else:
        result = execute(args.directory, phase=args.action,
            expected_frozen_sha256=getattr(args, 'expected_frozen_sha256', None),
            barrier_path=getattr(args, 'barrier', None), barrier_sha256=getattr(args, 'barrier_sha256', None))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
