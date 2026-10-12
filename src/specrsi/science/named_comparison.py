"""Explicit scientific-parameter adaptations of existing named search methods.

The Janus, HGM and AFlow algorithms are imported, not replaced with random
search. Their bounded allocation statistics retain the existing science
backend's atan scores. Every final candidate is selected by the same complete
native-utility ledger as SpecRSI. These are prospective parameter-space
adaptations, not claims of reproducing the authors' workflow/task experiments.
EMAS's graph-operation/correct-count contract and the existing HotpotQA-only
FlashEvolve runtime cannot be silently relabelled as numeric science methods.
"""
from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import random
import time
import uuid

from specrsi.baselines.extended_common import (Design, Evaluation, ExperimentBudget,
    Exhausted, Generated, ItemResult, RunContext)
from specrsi.core.acceptance import accept
from specrsi.core.budget import BudgetLedger
from specrsi.core.models import thaw
from specrsi.paper_profile import REFERENCE_BACKBONE_REPO, REFERENCE_BACKBONE_REVISION
from .baselines import ScienceBackend, numeric_features
from .executor import make_candidate


NAMED_METHODS = ('janus_science', 'hgm_science', 'aflow_science')
UNAVAILABLE_METHODS = {
    'emas': 'Existing EMAS requires typed workflow graph edits, disjoint evolve items and correct-count paired acceptance; numeric parameter JSON does not implement that contract.',
    'flash_evolve': 'The pinned official FlashEvolve adapter supports HotpotQA prompt optimization; a scientific-parameter/native-utility adapter has not been implemented.',
    'metarsi': 'No callable scientific MetaRSI outer-learning implementation/checkpoint is present in this comparison adapter.',
    'gepa': 'Existing prompt/reflection interface has not been registered for the scientific numeric program space.',
    'mipro': 'Existing prompt/few-shot optimization interface has not been registered for the scientific numeric program space.',
    'grpo': 'GRPO is a separate training baseline and has no trained scientific checkpoint in this comparison registration.',
}


class SearchDeadline:
    """One per-method search clock, excluding root acquisition/model loading.

    New work is admitted only if its whole declared slice fits. Backends remain
    responsible for their physical cancellation contracts. Actual overrun is
    measured and invalidates equal-budget reporting; no receipt is clipped.
    """
    def __init__(self, seconds=None):
        self.started = time.monotonic()
        self.limit = seconds
        if seconds is not None and (type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('search.wall_limit_seconds must be finite and positive')

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def fits(self, cap_ms=0):
        return self.limit is None or self.elapsed + cap_ms / 1000 <= self.limit

    def report(self):
        elapsed = self.elapsed
        return {'wall_limit_seconds': self.limit, 'metered_search_wall_seconds': elapsed,
                'wall_limit_respected': self.limit is None or elapsed <= self.limit,
                'wall_admission': 'whole declared operation/plan slice must fit; actual overruns remain recorded'}


def validate_named_config(config):
    """Read-only preflight; no network/model call before registration succeeds."""
    methods = set(config.get('methods', ())) & set(NAMED_METHODS)
    if not methods:
        return None
    options = config.get('named_baselines', {})
    model = options.get('model', {})
    if (model.get('base_url') != 'http://127.0.0.1:18085'
            or model.get('name') != 'Qwen3.5-35B-A3B' or not model.get('lock')):
        raise ValueError('Named science baselines require the explicit Qwen3.5-35B-A3B BF16 model on http://127.0.0.1:18085 and its lock')
    lock_path = Path(model['lock'])
    identity = json.loads(lock_path.read_text())
    if (identity.get('repo_id') != REFERENCE_BACKBONE_REPO
            or identity.get('revision') != REFERENCE_BACKBONE_REVISION
            or identity.get('dtype') != 'bfloat16'
            or identity.get('format') != 'publisher_bfloat16_unquantized'):
        raise ValueError('Named science model lock must pin the official unquantized BF16 reference snapshot')
    if 'wall_limit_seconds' not in config['search']:
        raise ValueError('Named comparisons require one common search.wall_limit_seconds for every method')
    SearchDeadline(config['search']['wall_limit_seconds'])
    for name in ('token_cap', 'request_token_cap'):
        value = options.get(name, 1000000 if name == 'token_cap' else 16384)
        if type(value) is not int or value < 1:
            raise ValueError(f'named_baselines.{name} must be a positive integer')
    return identity


class MatchedScienceBackend(ScienceBackend):
    """Existing real proposer/proxy interface with a common physical meter.

    Initial root evidence was acquired once outside B by the enclosing runner.
    Its first algorithm request is a read of that paid evidence. Any additional
    repeat is fresh paid work. All subsequent native calls, proposals, diagnoses
    and proxy proposals share B and the primitive-attempt allowance N.
    """
    def __init__(self, config, scope, root, ledger, executor, options, journal, deadline):
        self.config, self.scope, self.root, self.ledger = config, scope, root, ledger
        self.options, self.journal, self.deadline = options, journal, deadline
        self.budget = BudgetLedger(config['search']['budget_ms'])
        self.attempts = self.native_attempts = self.generation_attempts = 0
        self.initial = Design(thaw(root.artifacts), label=root.name)
        self.candidates = {self.initial.uid: root}
        self.initial_read = True
        model = config['named_baselines']['model']
        super().__init__(config['domain'], options.execution_cap, model,
                         case_ids=config['search_cases'], proposal_space='global_and_case_routes')
        self.executor = executor
        if (self.health.get('revision') != REFERENCE_BACKBONE_REVISION
                or self.health.get('model') != model['name'] or self.health.get('protocol_version') != 2):
            raise ValueError('Named baseline endpoint does not report the registered BF16 model/revision/protocol')

    def _admit(self, cap, attempts=1, wall_cap=None):
        if self.attempts + attempts > self.options.max_attempts:
            raise Exhausted('Shared primitive-attempt allowance exhausted')
        if cap > self.budget.available:
            raise Exhausted('Shared search expense cannot reserve the requested work')
        if not self.deadline.fits(cap if wall_cap is None else wall_cap):
            raise Exhausted('Shared wall deadline cannot reserve the requested work')

    def _candidate(self, design):
        if design.uid not in self.candidates:
            if len(self.candidates) >= self.options.max_candidates:
                raise Exhausted('Shared immutable-candidate cap reached')
            payload = thaw(design.payload)
            if set(payload) != {'parameters', 'scenario_overrides'}:
                raise ValueError('Named science design changed the registered parameter program schema')
            parent = self.candidates.get(design.parent_id)
            if parent is None:
                raise ValueError('Named science design does not identify an actual retained parent')
            self.candidates[design.uid] = make_candidate(self.domain, payload['parameters'],
                design.label, parent.uid, payload['scenario_overrides'])
        return self.candidates[design.uid]

    def evaluate(self, design, items, seed, stage, token_cap):
        candidate = self._candidate(design)
        by_id = {task.uid: task for task in self.scope.tasks}
        if any(item['id'] not in by_id for item in items):
            raise ValueError('Named baseline attempted evaluation outside the frozen development case registry')
        reuse_root = self.initial_read and design.uid == self.initial.uid
        self.initial_read = False
        physical = {}
        if not reuse_root:
            waves = math.ceil(len(items) / self.options.workers)
            self._admit(len(items) * self.options.execution_cap, len(items),
                        waves * self.options.execution_cap)
            jobs = {item['id']: 'named-evaluate-' + uuid.uuid4().hex for item in items}
            self.budget.reserve_many({job: (self.options.execution_cap, 'native_execution')
                                      for job in jobs.values()})
            self.attempts += len(items)
            self.native_attempts += len(items)

            def execute(item):
                try:
                    return self.executor.run(candidate, by_id[item['id']], self.scope, self.options.execution_cap)
                except BaseException as error:
                    return error

            # Like the VATS synchronous block, worker calls only produce
            # receipts; one coordinator settles and admits after the full block.
            if self.options.workers > 1 and len(items) > 1:
                with ThreadPoolExecutor(max_workers=min(self.options.workers, len(items))) as pool:
                    executed = list(pool.map(execute, items))
            else:
                executed = [execute(item) for item in items]
            first_error = None
            for item, result in zip(items, executed):
                job = jobs[item['id']]
                if isinstance(result, BaseException):
                    self.budget.settle_unknown(job)
                    first_error = first_error or result
                else:
                    try:
                        self.budget.settle(job, result.actual_cost)
                    except BaseException as error:
                        first_error = first_error or error
                    physical[item['id']] = result
            if first_error is not None:
                raise first_error
        results = []
        for item in items:
            task = by_id[item['id']]
            if reuse_root:
                record = self.ledger.get(self.root, task, self.scope)
                outcome, receipt, status, cost = record.outcome, record.receipt, 'complete', 0
            else:
                result = physical[task.uid]
                outcome, receipt, status, cost = result.outcome, result.receipt, result.status, result.actual_cost
                if status == 'complete' and outcome is not None:
                    self.ledger.admit_direct(candidate, task, self.scope, outcome, receipt)
            results.append(ItemResult(task.uid, outcome.score if outcome else 0., 0,
                complete=status == 'complete' and outcome is not None,
                valid=bool(outcome and outcome.admissible),
                trace=({'status': status, 'wall_cost_ms': cost, 'receipt': thaw(receipt),
                        'native_score': outcome.fields['native_score'] if outcome else None,
                        'initial_paid_evidence_read': reuse_root},)))
        return Evaluation(design.uid, tuple(results), seed, stage)

    def _generate_json(self, prompt, seed, token_cap):
        cap = self.options.draft_cap
        if cap <= 0:
            raise Exhausted('No model-generation slice is available')
        self._admit(cap)
        registered = self.config['named_baselines']
        request_cap = min(token_cap, registered.get('request_token_cap', 16384))
        job = 'named-generate-' + uuid.uuid4().hex
        self.budget.reserve(job, cap, 'named_model_generation')
        self.attempts += 1
        self.generation_attempts += 1
        self.journal.append({'event': 'start', 'kind': 'named_model_generation',
                             'call_id': job, 'cap_ms': cap, 'token_cap': request_cap})
        started = time.monotonic()
        try:
            reply = self.client.generate([
                {'role': 'system', 'content': 'Return only the requested JSON object. Propose parameter data; never Python code.'},
                {'role': 'user', 'content': prompt}],
                max_tokens=min(self.config['search'].get('draft_max_tokens', 256), max(1, request_cap - 1)),
                temperature=self.config['search'].get('temperature', 1.),
                total_token_cap=request_cap, seed=seed, max_seconds=cap / 1000,
                enable_thinking=False)
            elapsed = time.monotonic() - started
            actual = max(1, math.ceil(elapsed * 1000))
            self.journal.append({'event': 'finish', 'kind': 'named_model_generation', 'call_id': job,
                'cost_ms': actual, 'elapsed_seconds': elapsed, 'status': reply.finish_reason})
            self.journal.append({'event': 'physical_inference', 'kind': 'named_baseline',
                'role': 'proposal_diagnosis_or_proxy', **asdict(reply),
                'client_elapsed_seconds': elapsed})
            self.budget.settle(job, actual)
            self.calls.append(asdict(reply))
            if reply.model_revision != REFERENCE_BACKBONE_REVISION:
                raise ValueError('Named baseline generation changed model revision during the run')
            if reply.prompt_tokens + reply.completion_tokens > request_cap:
                raise ValueError('Named baseline generation exceeded its registered token slice')
            if reply.finish_reason == 'timeout':
                return None, reply.prompt_tokens + reply.completion_tokens
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', reply.text.strip())
            try:
                value = json.loads(text)
            except (ValueError, TypeError):
                value = None
            return value, reply.prompt_tokens + reply.completion_tokens
        except BaseException as error:
            if job in self.budget.reserved:
                self.budget.settle_unknown(job)
                self.journal.append({'event': 'error', 'kind': 'named_model_generation',
                    'call_id': job, 'error': repr(error), 'actual_cost_unknown': True,
                    'elapsed_seconds': time.monotonic() - started})
            raise

    def propose(self, parent, feedback, seed, token_cap):
        if len(self.candidates) >= self.options.max_candidates:
            raise Exhausted('Shared immutable-candidate cap reached')
        # Preserve ScienceBackend's registered finite-edit parameter program,
        # with the SAME catalog size as the HFParameterDrafter controls. The
        # existing backend's fixed default size=12 would change candidate
        # support when the common experiment registers a different size.
        from .proposals import parameter_catalog
        payload = thaw(parent.payload)
        parameters, overrides = payload['parameters'], payload.get('scenario_overrides', {})
        catalog = parameter_catalog(self.domain, parameters,
            [overrides.get('case:' + str(case), parameters) for case in self.case_ids],
            random.Random(seed), self.config['search'].get('catalog_size', 12))
        value, tokens = self._generate_json(
            'Choose a promising feasible scientific edit. Return ONLY JSON {"edit": integer}. '
            'task_index=-1 changes global parameters; other indices select registered stable case routes. '
            'Every edit is evaluated by the unchanged complete native simulator.\n' + json.dumps({
                'case_ids': self.case_ids, 'parent': payload, 'observed_feedback': str(feedback)[:2000],
                'choices': [{'edit': i, 'task_index': item['task_index'], 'changes': item['changes']}
                            for i, item in enumerate(catalog)]}), seed, token_cap)
        if not isinstance(value, dict) or set(value) != {'edit'} or type(value['edit']) is not int or not 0 <= value['edit'] < len(catalog):
            return Generated(None, tokens, 'Invalid catalog choice retained as a paid failed edit')
        chosen = catalog[value['edit']]
        if chosen['task_index'] < 0:
            parameters = chosen['parameters']
        else:
            overrides['case:' + str(self.case_ids[chosen['task_index']])] = chosen['parameters']
        generated = Generated(Design({'parameters': parameters, 'scenario_overrides': overrides},
            parent.uid, f'catalog-edit-{seed}'), tokens, 'Registered common numeric catalog; actual model selection')
        if generated.value is not None:
            self._candidate(generated.value)
        return generated


def run_named_method(method, config, scope, root, ledger, executor, options, journal, deadline):
    """Run a real existing algorithm, then apply common native final selection."""
    if method not in NAMED_METHODS:
        raise ValueError('Unsupported named scientific method')
    backend = MatchedScienceBackend(config, scope, root, ledger, executor, options, journal, deadline)
    items = [{'id': task.uid, 'domain': config['domain'], 'scenario_set': 'development',
              'case_id': task.inputs['case_id'], 'evaluator_identity': task.metadata['evaluator_identity']}
             for task in scope.tasks]
    extra = config['named_baselines']
    context = RunContext(backend, items, seed=options.seed,
        budget=ExperimentBudget(max_tokens=extra.get('token_cap', 1000000),
                                max_real_items=options.max_attempts + len(items)))
    settings = dict(extra.get(method, {}))
    if method == 'janus_science':
        from specrsi.baselines.janus import JanusConfig, run_janus
        settings.setdefault('real_evaluations', max(1, options.max_actions))
        algorithm_config = JanusConfig(**settings)
        invoke = lambda: run_janus(context, backend.initial, algorithm_config,
                                  features=lambda design: numeric_features(design, config['search_cases']))
    elif method == 'hgm_science':
        from specrsi.baselines.hgm import HGMConfig, run_hgm
        settings.setdefault('max_steps', max(1, options.max_actions))
        algorithm_config = HGMConfig(**settings)
        invoke = lambda: run_hgm(context, backend.initial, algorithm_config)
    else:
        from specrsi.baselines.aflow import AFlowConfig, run_aflow
        settings.setdefault('rounds', max(1, options.max_actions))
        algorithm_config = AFlowConfig(**settings)
        invoke = lambda: run_aflow(context, backend.initial, algorithm_config)
    try:
        algorithm_report = invoke()
    except Exhausted as error:
        # Initialization repeats may consume the cap before an algorithm's own
        # catch. Existing complete evidence still permits common ACCEPT.
        algorithm_report = {'status': 'budget_stop', 'reason': str(error), 'events': context.events}
    selected = accept(root, backend.candidates.values(), ledger, scope).selected
    initial, final = ledger.score(root, scope), ledger.score(selected, scope)
    return selected, {'method': method, 'initial_native_utility': initial,
        'selected_native_utility': final, 'selection_native_gain': final - initial,
        'search_budget_ms': backend.budget.total, 'search_cost_ms': backend.budget.spent,
        'budget_remaining_ms': backend.budget.available,
        'native_attempts': backend.native_attempts, 'draft_diagnosis_proxy_attempts': backend.generation_attempts,
        'primitive_attempts': backend.attempts, 'remaining_attempts': options.max_attempts - backend.attempts,
        'native_workers': options.workers, 'catalog_size': config['search'].get('catalog_size', 12),
        'charges': [asdict(charge) for charge in backend.budget.charges],
        'algorithm_configuration': asdict(algorithm_config), 'algorithm_report': algorithm_report,
        'search_elapsed_seconds': deadline.elapsed, 'model_identity': backend.health,
        'implementation': 'prospective_scientific_parameter_adaptation', 'author_exact': False,
        'algorithm_source': {'janus_science': 'src/specrsi/baselines/janus.py',
                             'hgm_science': 'src/specrsi/baselines/hgm.py',
                             'aflow_science': 'src/specrsi/baselines/aflow.py'}[method],
        'allocation_statistic': 'Existing bounded per-item atan scores; their mean is an allocation statistic, not native J.',
        'final_selection': 'Common maximum complete admissible native utility, root preferred only if tied.',
        'proposal_space': 'Same global parameters and registered stable case routes; existing ScienceBackend numeric catalog.',
        'proposal_model': 'Frozen official Qwen3.5-35B-A3B BF16 endpoint; no SpecRSI-trained Drafter adapter.',
        'cost_accounting': 'Shared actual native/model wall-ms, attempts and per-method elapsed cap; failed work is retained.'}
