"""Pinned official FlashEvolve GEPA runtime with a metered local task adapter.

The scheduler, minibatch gate, Pareto pool, reflection formatter, proposal
template/parser and optional stale-repair stages are imported from FlashEvolve.
The adapter connects this runtime to HotpotQA execution, native task scoring and
explicit generation budgets.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from specrsi.benchmarks.base import BenchmarkExample
from specrsi.benchmarks.data import load_examples, validate_disjoint_splits
from specrsi.benchmarks.scorers import hotpotqa_score
from specrsi.core.budget import BudgetLedger
from specrsi.output_contracts import generation_messages, output_contract_instruction, draft_interface_guidance

ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCK = ROOT / 'configs/baselines/flash_evolve.lock.json'
PHASE = ContextVar('flash_phase', default='search')
STAGE = ContextVar('flash_stage', default='unknown')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sampling_contract(config, role):
    """Resolve and register task randomness independently of proposal randomness."""
    default = ({'mode': 'example_identity_v1', 'seed': config['seed']}
               if role == 'audit_sampling' else {'mode': 'legacy_call_index'})
    value = dict(config.get(role, default))
    mode = value.get('mode')
    if mode == 'legacy_call_index':
        if set(value) != {'mode'}:
            raise ValueError(f'{role}: legacy_call_index accepts no independent seed')
    elif mode in {'fixed', 'example_identity_v1'}:
        if (set(value) != {'mode', 'seed'} or type(value['seed']) is not int
                or not 0 <= value['seed'] < 2**32):
            raise ValueError(f'{role}: register a uint32 seed for {mode}')
    else:
        raise ValueError(f'Unknown {role} mode: {mode}')
    return value


def example_generation_seed(example, contract):
    """Public cross-method seed rule; no candidate, request index, or label input."""
    if contract['mode'] == 'legacy_call_index':
        return None
    if contract['mode'] == 'fixed':
        return contract['seed']
    if contract['mode'] != 'example_identity_v1':
        raise ValueError('Unknown task seed rule')
    return int(sha(['specrsi-evaluation-seed-v1', contract['seed'], example.dataset,
                    example.id, example.content_hash])[:8], 16)


def original_source_split(example, storage_split):
    """Preserve original provenance even when development is stored as audit."""
    metadata = example.metadata
    origins = [metadata[k] for k in ('source_split', 'original_split', 'flash_source_split')
               if isinstance(metadata.get(k), str) and metadata[k]]
    # Any development origin wins over an intermediate, renamed storage role.
    if storage_split == 'development' or any(v.lower() in {'development', 'dev', 'train', 'training'} for v in origins):
        return 'development'
    if origins:
        return origins[0] if len(set(origins)) == 1 else 'ambiguous'
    return storage_split


def evaluation_eligibility(splits):
    """A native filename never overrides an explicit development-only scope."""
    rows = splits['audit']
    reasons = []
    origins = {e.metadata['flash_source_split'] for e in rows}
    if origins != {'audit'}:
        reasons.append('final_evaluation_is_not_entirely_original_audit')
    for e in rows:
        scope = str(e.metadata.get('evaluation_scope', '')).lower()
        if (e.metadata.get('not_benchmark_audit') is True or 'development' in scope
                or 'diagnostic' in scope or 'not_paper_audit' in scope):
            reasons.append('explicit_development_or_diagnostic_only_scope')
            break
    return {'eligible': not reasons, 'reasons': reasons,
            'audit_original_source_splits': sorted(origins),
            'historical_exposure_verified': False, 'sota_claim': False}


def pinned_flash():
    lock = json.loads(SOURCE_LOCK.read_text())
    directory = ROOT / lock['local_path']
    for name, expected in lock['files'].items():
        if not (directory / name).is_file() or file_sha(directory / name) != expected:
            raise ValueError(f'Pinned official FlashEvolve source changed or missing: {name}')
    extra = {str(p.relative_to(directory)) for p in directory.rglob('*.py')} - set(lock['files'])
    if extra:
        raise ValueError(f'Unregistered Python files in official FlashEvolve source: {sorted(extra)}')
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module('flashevolve')
    if not Path(module.__file__).resolve().is_relative_to(directory.resolve()):
        raise ValueError('Another FlashEvolve installation shadows the fixed source')
    # Isolate the upstream generic examples namespace from unrelated packages.
    name = '_specrsi_pinned_flash_gepa'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / 'examples/gepa/__init__.py',
            submodule_search_locations=[str(directory / 'examples/gepa')])
        examples = importlib.util.module_from_spec(spec)
        sys.modules[name] = examples
        spec.loader.exec_module(examples)
    return sys.modules[name], lock


def validate_config(config):
    if config.get('schema') != 'flash_evolve_local_v1' or config.get('dataset') != 'hotpotqa':
        raise ValueError('This adapter supports only flash_evolve_local_v1 / HotpotQA answer F1')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(config.get('seed_prompt'), str) or not config['seed_prompt'].strip():
        raise ValueError('Register a nonempty seed_prompt')
    alg = config['algorithm']
    if alg.get('runtime') not in {'async_gepa', 'sync_gepa'}:
        raise ValueError('Use official async_gepa or sync_gepa; optional-spec is not exposed by the pinned source')
    if alg.get('staleness_mode') not in {'none', 'hard', 'reflective', 'meta'}:
        raise ValueError('Unsupported official staleness mode; optional-spec is not implemented')
    if alg['runtime'] == 'sync_gepa' and alg['staleness_mode'] != 'none':
        raise ValueError('The synchronous control has no staleness repair mode')
    for key in ('proposal_budget', 'minibatch', 'pipeline_depth', 'eval_workers',
                'rollout_workers', 'propose_workers', 'repair_workers', 'meta_stale_n'):
        if type(alg.get(key)) is not int or alg[key] < 1:
            raise ValueError(f'algorithm.{key} must be a positive integer')
    if alg.get('meta_mode') not in {'serial', 'parallel'}:
        raise ValueError('meta_mode must be serial or parallel')
    if alg.get('selection_strategy') not in {'pareto_front', 'greedy', 'random', 'latest'}:
        raise ValueError('Unknown official parent selection strategy')
    if set(config['budget']) != {'initial', 'search', 'audit'}:
        raise ValueError('Register separate initial/search/audit budgets')
    for role, budget in config['budget'].items():
        if set(budget) != {'tokens', 'calls'} or any(type(v) is not int or v < 1 for v in budget.values()):
            raise ValueError(f'{role} requires positive integer token/call limits')
    for role in ('task', 'proposal'):
        gen = config['generation'][role]
        optional = {'output_contract'} if role == 'task' else set()
        if (not {'max_tokens', 'temperature', 'top_p'} <= set(gen)
                or set(gen) - {'max_tokens', 'temperature', 'top_p'} - optional or type(gen['max_tokens']) is not int
                or gen['max_tokens'] < 1 or type(gen['temperature']) not in (float, int)
                or not 0 <= gen['temperature'] <= 2 or type(gen['top_p']) not in (float, int)
                or not 0 < gen['top_p'] <= 1):
            raise ValueError(f'Invalid fixed generation contract: {role}')
        output_contract_instruction(gen)
    data = config['data']
    if data.get('mode') not in {'native_splits', 'development_smoke'}:
        raise ValueError('Choose native_splits or explicitly development-only engineering smoke')
    for role in ('development', 'selection', 'audit'):
        part = data[role]
        if (set(part) != {'offset', 'count'} or type(part['offset']) is not int or part['offset'] < 0
                or type(part['count']) is not int or part['count'] < 1):
            raise ValueError(f'Invalid registered data slice: {role}')
    if alg['minibatch'] > data['development']['count']:
        raise ValueError('Minibatch cannot exceed development set')
    if config['budget']['initial']['calls'] < data['selection']['count']:
        raise ValueError('Initial budget cannot complete the root selection evaluation')
    if config['budget']['audit']['calls'] < data['audit']['count']:
        raise ValueError('Audit budget cannot complete all registered audit items')
    sampling_contract(config, 'task_sampling')
    sampling_contract(config, 'audit_sampling')
    if 'request_token_caps' in config:
        caps = config['request_token_caps']
        if (not isinstance(caps, dict) or set(caps) != {'task', 'proposal'}
                or any(type(value) is not int or value < 1 for value in caps.values())):
            raise ValueError('request_token_caps requires positive integer task/proposal caps')
    if 'comparison' in config:
        comparison = config['comparison']
        planned = comparison.get('planned_run_ids') if isinstance(comparison, dict) else None
        if (not isinstance(comparison, dict) or set(comparison) != {'run_id', 'planned_run_ids'} or not isinstance(planned, list)
                or len(planned) < 2 or any(not isinstance(x, str) or not x for x in planned)
                or len(set(planned)) != len(planned) or comparison['run_id'] not in planned):
            raise ValueError('comparison requires a unique planned_run_ids set containing run_id')


def own_sources():
    names = ['src/specrsi/baselines/flash_evolve.py', 'scripts/run_flash_evolve.py',
             'src/specrsi/benchmarks/base.py', 'src/specrsi/benchmarks/data.py',
             'src/specrsi/benchmarks/scorers.py', 'src/specrsi/models/client.py',
             'src/specrsi/output_contracts.py',
             'src/specrsi/core/budget.py', 'configs/baselines/flash_evolve.lock.json']
    return {name: file_sha(ROOT / name) for name in names}


def register(config, output):
    validate_config(config)
    _, lock = pinned_flash()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new prospective FlashEvolve campaign directory')
    source_roles = {r: ('development' if config['data']['mode'] == 'development_smoke' else r)
                    for r in ('development', 'selection', 'audit')}
    raw = {r: load_examples('hotpotqa', r, data_root=ROOT / config.get('data_root', 'data'))
           for r in set(source_roles.values())}
    splits = {}
    for role, source in source_roles.items():
        part = config['data'][role]
        rows = raw[source][part['offset']:part['offset'] + part['count']]
        if len(rows) != part['count']:
            raise ValueError(f'Registered {role} data slice is unavailable')
        splits[role] = [replace(row, split=role, metadata={**row.metadata,
                        'flash_storage_split': source, 'flash_source_split': original_source_split(row, source)})
                        for row in rows]
    validate_disjoint_splits(splits)
    model_lock = ROOT / config['model']['lock']
    identity = json.loads(model_lock.read_text())
    files = own_sources()
    files[str(model_lock.resolve())] = file_sha(model_lock)
    # Freeze tokenizer/config bytes used for token admission, not just a model name.
    model_dir = ROOT / identity['local_path']
    tokenizer_files = [p for p in model_dir.iterdir() if p.is_file() and
                       (p.name.endswith('.json') or p.name.endswith('.jinja') or p.name.endswith('.model'))]
    for path in tokenizer_files:
        files[str(path.resolve())] = file_sha(path)
    registration = {'schema': 'flash_evolve_registration_v1', 'config': config,
        'upstream': {k: lock[k] for k in ('repository', 'commit', 'source_tree_sha256', 'license_declaration')},
        'model_identity': identity, 'source_sha256': files,
        'data_storage_splits': source_roles,
        'data_source_splits': {r: next(iter(origins)) if len(origins) == 1 else 'mixed'
            for r, rows in splits.items() for origins in [{e.metadata['flash_source_split'] for e in rows}]},
        'data_identity': {r: [{'id': e.id, 'input_hash': e.content_hash, 'source_split': e.metadata['flash_source_split']}
                              for e in rows] for r, rows in splits.items()},
        'split_sha256': {r: sha([e.to_dict() for e in rows]) for r, rows in splits.items()},
        'method': 'FlashEvolve official GEPA runtime with local HotpotQA adapter',
        'author_exact_replication': False, 'optional_spec_implemented': False,
        'comparison_to_trained_drafter_controls_valid': False,
        'model_service_parallelism': 'LocalModelClient serializes the server; native scheduling remains asynchronous',
        'performance_claim_eligible': evaluation_eligibility(splits)['eligible'],
        'evaluation_eligibility': evaluation_eligibility(splits),
        'cost_contract': 'actual input+output tokens and one call per generation; nontransferable phase caps',
        'sampling_contracts': {role: sampling_contract(config, role) for role in ('task_sampling', 'audit_sampling')},
        'sampling_reproducibility': 'Registered task/audit rules; proposals retain seed + admitted request index'}
    output.mkdir(parents=True, exist_ok=False)
    for role, rows in splits.items():
        write_new(output / f'{role}_examples.json', [e.to_dict() for e in rows])
    write_new(output / 'registration.json', registration)
    (output / 'registration.sha256').write_text(file_sha(output / 'registration.json') + '\n')
    return registration


def verify_registration(output):
    output = Path(output)
    if file_sha(output / 'registration.json') != (output / 'registration.sha256').read_text().strip():
        raise ValueError('FlashEvolve registration changed')
    registration = json.loads((output / 'registration.json').read_text())
    validate_config(registration['config'])
    if registration.get('sampling_contracts') != {role: sampling_contract(registration['config'], role)
                                                 for role in ('task_sampling', 'audit_sampling')}:
        raise ValueError('Registered sampling contracts changed or are missing; legacy artifacts must remain archived')
    for name, expected in registration['source_sha256'].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError(f'Registered source or model contract changed: {name}')
    splits = {}
    for role, expected in registration['split_sha256'].items():
        raw = json.loads((output / f'{role}_examples.json').read_text())
        if sha(raw) != expected:
            raise ValueError(f'Registered {role} data changed')
        splits[role] = [BenchmarkExample.from_dict(e) for e in raw]
    eligibility = evaluation_eligibility(splits)
    if (registration.get('evaluation_eligibility') != eligibility
            or registration['performance_claim_eligible'] != eligibility['eligible']):
        raise ValueError('Registered evaluation provenance/eligibility changed')
    pinned_flash()
    return registration


class PhysicalBudgetStop(RuntimeError):
    """Rejected before the physical model call; no fabricated stage output."""


class FlashMeter:
    """Event-loop atomic admission; shield/drain physical requests on cancellation.

    Async runtime work is concurrent, while the existing LocalModelClient's
    shared lock serializes the local server. Outstanding full request bounds
    are reserved before dispatch and unused output tokens refunded on receipt.
    """
    def __init__(self, client, tokenizer, *, revision, budgets, generation, seed,
                 context_length, journal, request_token_caps=None):
        self.client, self.tokenizer, self.revision = client, tokenizer, revision
        self.budgets, self.generation, self.seed = budgets, generation, seed
        self.context_length, self.journal = context_length, Path(journal)
        self.request_token_caps = request_token_caps or {}
        self.ledgers = {p: BudgetLedger(b['tokens']) for p, b in budgets.items()}
        self.counts = {p: 0 for p in budgets}
        self.records, self.physical_tasks, self.denied = [], set(), []
        self.search_closed = False

    def close_search(self):
        self.search_closed = True

    def token_count(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, thinking=False, return_dict=False))

    async def chat(self, request, *, role, generation_seed=None):
        from flashevolve.llm.base import ChatResponse
        phase, stage = PHASE.get(), STAGE.get()
        if request.model not in {None, self.client.model}:
            raise ValueError('Official stage requested an unregistered model')
        generation = dict(self.generation['task' if role == 'task' else 'proposal'])
        if set(request.extra) - {'max_tokens', 'temperature'}:
            raise ValueError('Unregistered upstream generation parameters')
        for name, value in request.extra.items():
            if generation.get(name) != value:
                raise ValueError('Official stage generation settings differ from registration')
        count = self.token_count(request.messages)
        cap = count + generation['max_tokens']
        request_cap = min(self.context_length, self.request_token_caps.get(
            'task' if role == 'task' else 'proposal', self.context_length))
        ledger = self.ledgers[phase]
        if (phase == 'search' and self.search_closed or cap > request_cap
                or cap > ledger.available or self.counts[phase] >= self.budgets[phase]['calls']):
            denial = {'phase': phase, 'stage': stage, 'role': role, 'requested_tokens': cap,
                      'request_token_cap': request_cap,
                      'available_tokens': ledger.available, 'admitted_calls': self.counts[phase]}
            self.denied.append(denial)
            raise PhysicalBudgetStop(f'Physical request rejected before dispatch: {canonical(denial)}')
        index = sum(self.counts.values())
        if generation_seed is not None and (role != 'task' or type(generation_seed) is not int
                                             or not 0 <= generation_seed < 2**32):
            raise ValueError('Only task requests accept a registered uint32 generation seed')
        seed = self.seed + index if generation_seed is None else generation_seed
        key = f'call-{index:06d}'
        ledger.reserve(key, cap, role)
        self.counts[phase] += 1
        record = {'call_id': key, 'phase': phase, 'stage': stage, 'role': role,
                  'input_sha256': sha(request.messages), 'messages': request.messages,
                  'reserved_tokens': cap, 'sampling': {**generation, 'seed': seed,
                                                      'enable_thinking': False}}

        async def physical():
            started = time.monotonic()
            try:
                reply = await asyncio.to_thread(self.client.generate, request.messages,
                    **{k: v for k, v in generation.items() if k != 'output_contract'},
                    seed=seed, enable_thinking=False, total_token_cap=cap)
            except BaseException as error:
                ledger.settle_unknown(key)
                record.update(status='unmetered_failure', cost_kind='conservative_bound',
                              error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
                raise
            else:
                record.update(asdict(reply), status='complete', cost_kind='actual')
                # Settle paid work before checking identity/format/timeout.
                ledger.settle(key, reply.prompt_tokens + reply.completion_tokens)
                if (reply.prompt_tokens != count or reply.model_revision != self.revision
                        or reply.finish_reason not in {'stop', 'length'}):
                    record['status'] = 'invalid_measured_response'
                    ledger.cost_valid = False
                    raise ValueError('Measured FlashEvolve response violates frozen tokenizer/model/status contract')
                return ChatResponse(content=reply.text, raw={'call_id': key,
                    'prompt_tokens': reply.prompt_tokens, 'completion_tokens': reply.completion_tokens})
            finally:
                self.records.append(record)
                with self.journal.open('a', encoding='utf-8') as handle:
                    handle.write(canonical(record) + '\n')

        task = asyncio.create_task(physical())
        self.physical_tasks.add(task)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self.physical_tasks.discard(task)

    async def drain(self):
        tasks = list(self.physical_tasks)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.physical_tasks.difference_update(tasks)
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    def receipt(self):
        return {p: {'token_cap': self.budgets[p]['tokens'], 'call_cap': self.budgets[p]['calls'],
                    'actual_tokens': b.spent if b.cost_valid else None, 'charged_tokens': b.spent,
                    'cost_kind': 'actual' if b.cost_valid else 'invalid_or_conservative_bound',
                    'actual_calls': self.counts[p], 'pending_tokens': b.pending,
                    'cost_valid': b.cost_valid, 'budget_breached': b.breached}
                for p, b in self.ledgers.items()}


class ModelRole:
    def __init__(self, meter, role, sampling=None, *, task_generation=None):
        self.meter, self.role, self.sampling = meter, role, sampling
        self.task_description = draft_interface_guidance(task_generation or {})

    async def chat(self, request, *, example=None):
        if self.task_description:
            # Extend only the local adapter's public task description. The
            # upstream proposal/repair template and parser remain unchanged.
            messages = [dict(message) for message in request.messages]
            user_indices = [i for i, message in enumerate(messages) if message.get('role') == 'user']
            if not user_indices or not isinstance(messages[user_indices[-1]].get('content'), str):
                raise ValueError('A proposal output contract requires a textual user task description')
            messages[user_indices[-1]]['content'] += '\n\n' + self.task_description
            request = replace(request, messages=messages)
        seed = example_generation_seed(example, self.sampling) if self.sampling is not None else None
        return await self.meter.chat(request, role=self.role, generation_seed=seed)


async def optimize(config, development, selection, meter):
    """Use the official scheduler; only task rendering/metric and logging adapt."""
    reference, _ = pinned_flash()
    from flashevolve.agents.base import Agent, AgentResult
    from flashevolve.artifacts import PromptArtifact
    from flashevolve.llm.base import ChatRequest
    from flashevolve.pools import AppendOnlyPool
    from flashevolve.runtime import AsyncGEPARuntime, GEPARuntime
    from flashevolve.samplers import FullSampler, RandomSampler
    from flashevolve.stages.reflect_patch import ReflectivePatch

    class LocalAgent(Agent):
        async def run(self, artifact, samples, *, llm):
            async def one(sample):
                reply = await llm.chat(ChatRequest(messages=generation_messages(artifact.text, sample.prompt,
                    config['generation']['task'], dataset=sample.dataset)), example=sample)
                return AgentResult(output=reply.content, signals={'example_id': sample.id,
                    'split': sample.split, 'answer': reply.content, **reply.raw})
            return await asyncio.gather(*(one(s) for s in samples))

    evaluations, stage_events = [], []

    class RecordedStage:
        def __init__(self, original, name):
            self.original, self.name, self.workers = original, name, original.workers

        async def process(self, item):
            bootstrap = self.name == 'evaluate' and item.critique.text == '<bootstrap>'
            token1 = PHASE.set('initial' if bootstrap else 'search')
            token2 = STAGE.set(self.name)
            event = {'stage': self.name, 'phase': PHASE.get(), 'status': 'started'}
            stage_events.append(event)
            try:
                result = await self.original.process(item)
                if result is None:
                    raise ValueError(f'Official stage returned no result: {self.name}')
                if self.name in {'propose', 'repair'} and not result.artifact.text.strip():
                    raise ValueError('Paid proposal/repair produced an empty instruction')
                event['status'] = 'complete'
                if self.name == 'evaluate':
                    evaluations.append({'system_prompt': result.candidate.artifact.text,
                        'score': result.score, 'signals': result.signals, 'bootstrap': bootstrap})
                return result
            except BaseException as error:
                event.update(status='interrupted' if isinstance(error, (PhysicalBudgetStop, asyncio.CancelledError))
                             else 'failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                PHASE.reset(token1)
                STAGE.reset(token2)

    def metric(sample, answer):
        return hotpotqa_score(answer, sample.reference).score

    def feedback(sample, answer, score):
        if sample.split != 'development':
            raise ValueError('Only development references may enter proposal feedback')
        return canonical({'answer_f1': score, 'reference': sample.reference})

    alg = config['algorithm']
    agent, task_model = LocalAgent(), ModelRole(meter, 'task', sampling_contract(config, 'task_sampling'))
    pool = AppendOnlyPool(seed=config['seed'])
    rollout = RecordedStage(reference.make_gepa_rollout(agent=agent, metric=metric, feedback_fn=feedback,
        llm=task_model, workers=alg['rollout_workers']), 'rollout')
    reflect = RecordedStage(reference.GEPAReflect(sample_renderer=lambda s: s.prompt), 'reflect')
    proposal_settings = config['generation']['proposal']
    propose = RecordedStage(reference.make_gepa_propose(llm=ModelRole(meter, 'proposal',
        task_generation=config['generation']['task']),
        workers=alg['propose_workers'], max_tokens=proposal_settings['max_tokens'],
        temperature=proposal_settings['temperature']), 'propose')
    evaluate = RecordedStage(reference.make_gepa_evaluate(agent=agent, metric=metric,
        llm=task_model, sampler=FullSampler(selection)), 'evaluate')
    kwargs = dict(pool=pool, sampler=RandomSampler(development, alg['minibatch'], seed=config['seed']),
        rollout=rollout, reflect=reflect, propose=propose, evaluate=evaluate,
        initial_artifact=PromptArtifact(text=config['seed_prompt']), budget=alg['proposal_budget'],
        selection_strategy=alg['selection_strategy'])
    if alg['runtime'] == 'async_gepa':
        repair = RecordedStage(ReflectivePatch(ModelRole(meter, 'repair',
            task_generation=config['generation']['task']), workers=alg['repair_workers']), 'repair')
        runtime = AsyncGEPARuntime(**kwargs, pipeline_depth=alg['pipeline_depth'],
            eval_workers=alg['eval_workers'], staleness_mode=alg['staleness_mode'], repair=repair,
            meta_mode=alg['meta_mode'], meta_stale_n=alg['meta_stale_n'])
    else:
        runtime = GEPARuntime(**kwargs)
    # Upstream gather has no exception-time sibling cleanup. Own and drain the
    # tasks created by this isolated invocation before freezing a successor.
    before = asyncio.all_tasks()
    stop = None
    try:
        await runtime.run()
    except PhysicalBudgetStop as error:
        stop = str(error)
        if not len(pool):
            raise ValueError('Budget could not complete initial candidate evaluation') from error
    finally:
        meter.close_search()
        pending = asyncio.all_tasks() - before - meter.physical_tasks
        for task in pending:
            task.cancel()
        cleanup = await asyncio.gather(*pending, return_exceptions=True)
        await meter.drain()
        errors = [e for e in cleanup if isinstance(e, BaseException)
                  and not isinstance(e, (asyncio.CancelledError, PhysicalBudgetStop))]
        if errors:
            raise errors[0]
    _, best = await pool.select_parent(strategy='greedy')
    # Read completed selection records for the exact chosen pool member.
    matching = [r for r in evaluations if r['system_prompt'] == best.text]
    if not matching:
        raise ValueError('Official pool selected an artifact without complete evaluation')
    selected = max(matching, key=lambda r: r['score'])
    return {'system_prompt': best.text, 'selection_score': selected['score']}, {
        'runtime_class': f'{type(runtime).__module__}.{type(runtime).__name__}',
        'budget_truncated': stop is not None, 'budget_stop': stop, 'admitted_candidates': len(pool),
        'evaluations': evaluations, 'stages': stage_events,
        'native_stats': {k: getattr(runtime, k) for k in ('stale_seen', 'stale_discarded', 'stale_repaired',
            'gated_out', 'meta_syntheses', 'meta_buffer_max', 'accepted_iterations', 'rejected_iterations')
                         if hasattr(runtime, k)}}


def create_meter(registration, output):
    from transformers import AutoTokenizer
    from specrsi.models import LocalModelClient
    config, identity = registration['config'], registration['model_identity']
    client = LocalModelClient(config['model']['base_url'], config['model']['served_model_name'])
    health = client.health()
    if (health.get('revision') != identity['revision'] or health.get('model') != client.model
            or health.get('protocol_version', 0) < 2
            or not health.get('server_enforces_total_token_cap')):
        raise ValueError('FlashEvolve requires the frozen local model and hard server token caps')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / identity['local_path'], local_files_only=True,
                                              trust_remote_code=identity.get('trust_remote_code', False))
    return FlashMeter(client, tokenizer, revision=identity['revision'], budgets=config['budget'],
        generation=config['generation'], seed=config['seed'], context_length=health.get('context_length', 32768),
        journal=Path(output) / 'physical_calls.jsonl', request_token_caps=config.get('request_token_caps'))


def _rows(output, role):
    return [BenchmarkExample.from_dict(e) for e in json.loads((Path(output) / f'{role}_examples.json').read_text())]


def _valid_receipts(receipts):
    return all(r['cost_valid'] and not r['budget_breached'] and not r['pending_tokens'] for r in receipts.values())


def _record_failure(output, error, meter, started, stage):
    write_new(Path(output) / 'failure.json', {'status': 'failed', 'stage': stage,
        'error': f'{type(error).__name__}: {error}',
        'phase_receipts': meter.receipt() if meter else {}, 'denied_requests': meter.denied if meter else [],
        'elapsed_seconds': time.monotonic() - started})


def search_only(output, *, meter_factory=create_meter):
    """Optimize and freeze; never issue a heldout request or publish a score."""
    output = Path(output)
    registration = verify_registration(output)
    registration_sha = file_sha(output / 'registration.json')
    write_new(output / 'started.json', {'registration_sha256': registration_sha,
        'stage': 'search', 'started_unix': time.time()})
    meter, started = None, time.monotonic()
    try:
        meter = meter_factory(registration, output)
        selected, search = asyncio.run(optimize(registration['config'], _rows(output, 'development'),
                                                _rows(output, 'selection'), meter))
        if file_sha(output / 'registration.json') != registration_sha:
            raise ValueError('FlashEvolve registration changed during search')
        verify_registration(output)
        receipts = meter.receipt()
        if not _valid_receipts(receipts) or meter.physical_tasks or receipts['audit']['actual_calls']:
            raise ValueError('Search cannot freeze incomplete physical work or heldout requests')
        write_new(output / 'search.json', search)
        write_new(output / 'selected.json', selected)
        write_new(output / 'selection_frozen.json', {
            'schema': 'flash_evolve_selection_frozen_v2', 'registration_sha256': registration_sha,
            'selected_sha256': file_sha(output / 'selected.json'),
            'search_sha256': file_sha(output / 'search.json'),
            'search_journal_sha256': file_sha(output / 'physical_calls.jsonl'),
            'search_journal_bytes': (output / 'physical_calls.jsonl').stat().st_size,
            'search_receipt': receipts, 'pending_physical_calls': 0,
            'denied_requests': meter.denied, 'frozen_at_unix': time.time(),
            'search_elapsed_seconds': time.monotonic() - started})
        return {'status': 'search_complete', 'selected_sha256': file_sha(output / 'selected.json'),
                'selection_frozen_sha256': file_sha(output / 'selection_frozen.json'),
                'phase_receipts': receipts, 'audit_model_calls': 0,
                'performance_claim_eligible': registration['performance_claim_eligible']}
    except BaseException as error:
        _record_failure(output, error, meter, started, 'search')
        raise


def verify_comparison_barrier(registration, output, path=None, expected_sha256=None):
    """Check the entire prospectively registered run set before heldout dispatch."""
    comparison = registration['config'].get('comparison')
    if comparison is None:
        if path is not None or expected_sha256 is not None:
            raise ValueError('An external barrier requires a prospectively registered comparison')
        return None
    if path is None or expected_sha256 is None:
        raise ValueError('This comparison requires the cross-method barrier and its frozen SHA256')
    path = Path(path).resolve()
    if file_sha(path) != expected_sha256:
        raise ValueError('Cross-method barrier SHA256 changed')
    barrier = json.loads(path.read_text())
    planned, entries = barrier.get('planned_run_ids'), barrier.get('runs')
    expected = comparison['planned_run_ids']
    if (barrier.get('schema') != 'cross_method_selection_barrier_v1' or not isinstance(planned, list)
            or len(planned) != len(set(planned)) or set(planned) != set(expected)
            or not isinstance(entries, dict) or set(entries) != set(expected)):
        raise ValueError('Cross-method barrier does not cover the exact registered run set')
    paths_seen = {'selected': set(), 'selection_frozen': set()}
    for run_id, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {'selected', 'selection_frozen'}:
            raise ValueError('Each barrier run must bind selected and selection_frozen artifacts')
        for field in ('selected', 'selection_frozen'):
            value = entry[field]
            if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
                raise ValueError('Malformed cross-method artifact binding')
            artifact = Path(value['path'])
            artifact = artifact if artifact.is_absolute() else path.parent / artifact
            artifact = artifact.resolve()
            if artifact in paths_seen[field]:
                raise ValueError(f'Cross-method runs cannot reuse the same {field} path')
            paths_seen[field].add(artifact)
            if file_sha(artifact) != value['sha256']:
                raise ValueError(f'Cross-method frozen {field} changed: {run_id}')
            if run_id == comparison['run_id'] and artifact != (Path(output) / f'{field}.json').resolve():
                raise ValueError('Barrier binds a different artifact directory for this run')
    return {'path': str(path), 'sha256': expected_sha256, 'run_id': comparison['run_id'],
            'planned_run_ids': planned}


def _frozen_search(output, registration, expected_frozen_sha256):
    output = Path(output)
    if not isinstance(expected_frozen_sha256, str) or file_sha(output / 'selection_frozen.json') != expected_frozen_sha256:
        raise ValueError('Frozen search receipt differs from the caller-provided SHA256')
    frozen = json.loads((output / 'selection_frozen.json').read_text())
    started = json.loads((output / 'started.json').read_text())
    if (frozen.get('schema') != 'flash_evolve_selection_frozen_v2'
            or frozen['registration_sha256'] != file_sha(output / 'registration.json')
            or started['registration_sha256'] != frozen['registration_sha256']
            or frozen['pending_physical_calls'] != 0):
        raise ValueError('Frozen search is incomplete or belongs to another registration')
    for name, key in (('selected.json', 'selected_sha256'), ('search.json', 'search_sha256'),
                      ('physical_calls.jsonl', 'search_journal_sha256')):
        if file_sha(output / name) != frozen[key]:
            raise ValueError(f'Frozen search artifact changed before audit: {name}')
    journal = (output / 'physical_calls.jsonl').read_bytes()
    if len(journal) != frozen['search_journal_bytes']:
        raise ValueError('Frozen search physical journal length changed')
    records = [json.loads(line) for line in journal.split(b'\n') if line.strip()]
    caps = registration['config']['budget']
    totals = {phase: {'tokens': 0, 'calls': 0} for phase in caps}
    for record in records:
        phase = record.get('phase')
        if (phase not in {'initial', 'search'} or record.get('status') != 'complete'
                or record.get('cost_kind') != 'actual' or record.get('finish_reason') not in {'stop', 'length'}
                or record.get('role') not in {'task', 'proposal', 'repair'}):
            raise ValueError('Frozen search contains incomplete, unmetered, or heldout work')
        usage = [record.get(key) for key in ('prompt_tokens', 'completion_tokens', 'reserved_tokens')]
        if any(type(x) is not int or x < 0 for x in usage) or sum(usage[:2]) > usage[2]:
            raise ValueError('Frozen search physical usage violates its reservation')
        totals[phase]['tokens'] += sum(usage[:2])
        totals[phase]['calls'] += 1
    if {r['call_id'] for r in records} != {f'call-{i:06d}' for i in range(len(records))}:
        raise ValueError('Frozen search request indices are missing or duplicated')
    if totals['initial']['calls'] != registration['config']['data']['selection']['count']:
        raise ValueError('Frozen search root evaluation is incomplete')
    for phase, spent in totals.items():
        expected = {'token_cap': caps[phase]['tokens'], 'call_cap': caps[phase]['calls'],
            'actual_tokens': spent['tokens'], 'charged_tokens': spent['tokens'], 'cost_kind': 'actual',
            'actual_calls': spent['calls'], 'pending_tokens': 0, 'cost_valid': True, 'budget_breached': False}
        if (frozen['search_receipt'][phase] != expected or spent['tokens'] > caps[phase]['tokens']
                or spent['calls'] > caps[phase]['calls']):
            raise ValueError('Frozen search receipt differs from its complete physical journal')
    return frozen, records


def _restore_search_meter(meter, frozen, records):
    if meter.records or any(meter.counts.values()) or any(b.spent or b.pending for b in meter.ledgers.values()):
        raise ValueError('Audit-only requires a fresh physical meter')
    for record in records:
        if record['model_revision'] != meter.revision:
            raise ValueError('Frozen search and audit use different model revisions')
        phase = record['phase']
        meter.ledgers[phase].charge(record['call_id'], record['prompt_tokens'] + record['completion_tokens'], record['role'])
        meter.counts[phase] += 1
    meter.records.extend(records)
    meter.denied.extend(frozen['denied_requests'])
    meter.close_search()
    if meter.receipt() != frozen['search_receipt']:
        raise ValueError('Restored meter does not match the frozen search bill')


def audit_only(output, *, expected_frozen_sha256, barrier_path=None, barrier_sha256=None,
               meter_factory=create_meter):
    """Evaluate only after validating the exact search and optional all-arm barrier."""
    output = Path(output)
    registration = verify_registration(output)
    if any((output / name).exists() for name in ('failure.json', 'audit_started.json', 'audit.json', 'summary.json')):
        raise FileExistsError('Audit was already attempted; preserve existing evidence')
    frozen, records = _frozen_search(output, registration, expected_frozen_sha256)
    barrier = verify_comparison_barrier(registration, output, barrier_path, barrier_sha256)
    registration_sha = file_sha(output / 'registration.json')
    selected = json.loads((output / 'selected.json').read_text())
    search = json.loads((output / 'search.json').read_text())
    frozen_hashes = {name: file_sha(output / name) for name in ('search.json', 'selected.json', 'selection_frozen.json')}

    def verify_unchanged():
        if file_sha(output / 'registration.json') != registration_sha:
            raise ValueError('FlashEvolve registration changed during audit')
        if any(file_sha(output / name) != expected for name, expected in frozen_hashes.items()):
            raise ValueError('Selected artifact or frozen search evidence changed during audit')
        with (output / 'physical_calls.jsonl').open('rb') as handle:
            prefix = handle.read(frozen['search_journal_bytes'])
        if hashlib.sha256(prefix).hexdigest() != frozen['search_journal_sha256']:
            raise ValueError('Frozen physical search journal changed during audit')
        verify_registration(output)
        verify_comparison_barrier(registration, output, barrier_path, barrier_sha256)

    write_new(output / 'audit_started.json', {'registration_sha256': registration_sha,
        'selection_frozen_sha256': expected_frozen_sha256, 'external_barrier': barrier,
        'started_unix': time.time()})
    meter, started = None, time.monotonic()
    try:
        meter = meter_factory(registration, output)
        _restore_search_meter(meter, frozen, records)
        config = registration['config']

        async def audit():
            from flashevolve.llm.base import ChatRequest
            token1, token2 = PHASE.set('audit'), STAGE.set('audit')
            audit_rows = []
            try:
                for example in _rows(output, 'audit'):
                    verify_comparison_barrier(registration, output, barrier_path, barrier_sha256)
                    seed = example_generation_seed(example, registration['sampling_contracts']['audit_sampling'])
                    response = await meter.chat(ChatRequest(messages=generation_messages(
                        selected['system_prompt'], example.prompt, config['generation']['task'],
                        dataset=example.dataset)), role='task', generation_seed=seed)
                    score = hotpotqa_score(response.content, example.reference)
                    audit_rows.append({'id': example.id, 'answer': response.content,
                        'score': score.score, 'metrics': dict(score.metrics), **response.raw})
            finally:
                PHASE.reset(token1)
                STAGE.reset(token2)
                await meter.drain()
            return audit_rows
        verify_unchanged()
        audit_rows = asyncio.run(audit())
        verify_unchanged()
        receipts = meter.receipt()
        if not _valid_receipts(receipts):
            raise ValueError('Incomplete or invalid physical billing')
        write_new(output / 'audit.json', audit_rows)
        audit_elapsed = time.monotonic() - started
        summary = {'status': 'complete', 'method': registration['method'], 'upstream': registration['upstream'],
            'runtime': config['algorithm'], 'data_mode': config['data']['mode'],
            'performance_claim_eligible': registration['performance_claim_eligible'],
            'evaluation_eligibility': registration['evaluation_eligibility'],
            'sampling_contracts': registration['sampling_contracts'], 'external_barrier': barrier,
            'comparison_to_trained_drafter_controls_valid': False,
            'sota_claim': False, 'optional_spec_implemented': False, 'budget_truncated': search['budget_truncated'],
            'selected_sha256': frozen['selected_sha256'], 'selection_score': selected['selection_score'],
            'audit_score': sum(r['score'] for r in audit_rows) / len(audit_rows),
            'audit_count': len(audit_rows), 'phase_receipts': receipts,
            'physical_tokens': sum(r['actual_tokens'] for r in receipts.values()),
            'physical_calls': sum(r['actual_calls'] for r in receipts.values()),
            'denied_requests': meter.denied, 'search_elapsed_seconds': frozen['search_elapsed_seconds'],
            'audit_elapsed_seconds': audit_elapsed, 'elapsed_seconds': frozen['search_elapsed_seconds'] + audit_elapsed,
            'raw_artifacts': {n: file_sha(output / n) for n in ('registration.json', 'search.json',
                'selected.json', 'selection_frozen.json', 'audit_started.json', 'audit.json', 'physical_calls.jsonl')}}
        write_new(output / 'summary.json', summary)
        return summary
    except BaseException as error:
        _record_failure(output, error, meter, started, 'audit')
        raise


def execute(output, *, meter_factory=create_meter):
    """Legacy single-run entry: freeze then audit immediately, without a campaign."""
    registration = verify_registration(output)
    if 'comparison' in registration['config']:
        raise ValueError('Registered cross-method comparisons require search-only, then audit-only with a barrier')
    result = search_only(output, meter_factory=meter_factory)
    return audit_only(output, expected_frozen_sha256=result['selection_frozen_sha256'], meter_factory=meter_factory)
