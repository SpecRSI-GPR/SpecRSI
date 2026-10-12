"""Shared local-only metering and held-out boundary for secondary optimizers."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from specrsi.benchmarks.data import validate_disjoint_splits
from specrsi.core.budget import BudgetLedger, BudgetExceeded
from specrsi.models import LocalModelClient
from specrsi.tokenization import chat_token_ids
from .gepa import MeteredLocalModel

ROOT = Path(__file__).resolve().parents[3]


def pinned_import(name):
    lock = json.loads((ROOT / f"configs/baselines/secondary_{name}.lock.json").read_text())
    directory = ROOT / lock['local_path']
    for filename, expected in lock['files'].items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned original {name} source changed: {filename}")
    sys.path.insert(0, str(directory)) if str(directory) not in sys.path else None
    module = importlib.import_module('opto.trace' if name == 'trace' else name)
    if not Path(module.__file__).resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Another {name} installation shadows the pinned original source")
    return module, {key: lock[key] for key in ('repository', 'commit', 'archive_sha256', 'local_path')}


class CappedClient:
    """Charge every physical task/judge/optimizer call, with a separate audit cap."""
    def __init__(self, client, tokenizer, *, revision, search_tokens, audit_tokens, context_length=32768):
        self.client, self.tokenizer, self.model = client, tokenizer, client.model
        self.revision, self.context_length = revision, context_length
        self.ledgers = {'optimization': BudgetLedger(search_tokens), 'audit': BudgetLedger(audit_tokens)}
        self.phase = 'optimization'
        self.calls = []

    def health(self):
        return self.client.health()

    def generate(self, messages, **kwargs):
        ledger = self.ledgers[self.phase]
        thinking = kwargs.get('enable_thinking', False)
        count = len(chat_token_ids(self.tokenizer,messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=thinking, thinking=thinking, return_dict=False))
        cap = min(ledger.available, kwargs.get('total_token_cap') or self.context_length, self.context_length)
        maximum = kwargs.get('max_tokens', 512)
        if count + maximum > cap:
            raise BudgetExceeded('Full secondary model request does not fit its remaining enforced token cap')
        job = f'{self.phase}-{len(self.calls)}'
        ledger.reserve(job, count + maximum, 'physical-model-call')
        started = time.monotonic()
        try:
            reply = self.client.generate(messages, **{**kwargs, 'total_token_cap': cap})
        except Exception:
            ledger.settle_unknown(job)
            self.calls.append({'phase': self.phase, 'status': 'unmetered_failure', 'cost_kind': 'conservative_bound',
                               'elapsed_seconds': time.monotonic() - started})
            raise
        actual = reply.prompt_tokens + reply.completion_tokens
        ledger.settle(job, actual)
        self.calls.append({'phase': self.phase, 'status': 'complete', 'cost_kind': 'actual',
            'request_id': reply.request_id, 'prompt_tokens': reply.prompt_tokens,
            'completion_tokens': reply.completion_tokens, 'elapsed_seconds': reply.elapsed_seconds,
            'gpu_count': reply.gpu_count, 'revision': reply.model_revision})
        if reply.prompt_tokens != count or reply.model_revision != self.revision:
            ledger.cost_valid = False
            raise ValueError('Secondary frozen tokenizer/model identity changed after a measured call')
        return reply

    def receipt(self):
        return {role: {'total_cap': b.total, 'spent': b.spent, 'remaining': b.available,
                       'cost_valid': b.cost_valid, 'budget_breached': b.breached}
                for role, b in self.ledgers.items()}


def create_model(config):
    from transformers import AutoTokenizer
    settings = config['model']
    lock = json.loads((ROOT / settings['lock']).read_text())
    native = LocalModelClient(settings['base_url'], lock.get('served_model_name', 'Qwen3.5-35B-A3B'))
    health = native.health()
    if (health.get('revision') != lock['revision'] or health.get('protocol_version') != 2
            or not health.get('server_enforces_total_token_cap')):
        raise ValueError('Secondary baseline requires the pinned SpecRSI local model protocol')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / lock['local_path'], local_files_only=True,
                                              trust_remote_code=lock.get('trust_remote_code', False))
    budgets = config.get('budget', {})
    client = CappedClient(native, tokenizer, revision=lock['revision'],
        search_tokens=int(budgets.get('search_tokens', 3000000)),
        audit_tokens=int(budgets.get('audit_tokens', max(131072, config['data']['audit'] * settings.get('execution_cap', 131072)))),
        context_length=health.get('context_length', 32768))
    model = MeteredLocalModel(client, max_tokens=settings.get('max_tokens', 1024),
        temperature=settings.get('temperature', 0.), seed=config.get('seed', 0), model_revision=lock['revision'])
    if config['dataset'] in {'hover', 'pupa', 'polyglot', 'livebench_math', 'swe_verified', 'swe_lite'}:
        from .tasks import attach_pipeline
        attach_pipeline(model, lock, dataset=config['dataset'], native_config=settings)
    return model


def prepare_run(config, splits, model):
    if set(splits) != {'development', 'selection', 'audit'} or any(not rows for rows in splits.values()):
        raise ValueError('Secondary optimization requires nonempty development, selection and audit splits')
    validate_disjoint_splits(splits)
    if any(example.dataset != config['dataset'] for rows in splits.values() for example in rows):
        raise ValueError('Secondary study requires one native dataset/metric')
    if model.records:
        raise ValueError('Use a fresh physical model ledger for each secondary baseline')
    output = Path(config['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve prior secondary run; choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    model.journal = output / 'model_calls.jsonl'
    manifest = {'method': config['method'], 'protocol': 'prospective_pinned_original_optimizer_local_native_adapter',
        'author_exact_replication': False, 'reuse_matched': False, 'opaque_results_all_U': True,
        'dataset': config['dataset'], 'model_revision': model.model_revision,
        'splits': {role: [e.id for e in rows] for role, rows in splits.items()},
        'task_protocol': getattr(model, 'task_protocol', {'metric_variant': 'task_native'}),
        'budget_endpoint': 'actual prompt plus generated tokens, separate optimization and audit ledgers',
        'algorithm': config.get('algorithm', {}), 'seed': config.get('seed', 0)}
    (output / 'resolved_config.json').write_text(json.dumps(config, indent=2) + '\n')
    return output, manifest


def evaluate_prompt(model, instruction, examples):
    rows = []
    for example in examples:
        answer, score = model.evaluate_example(instruction, example)
        rows.append({'id': example.id, 'answer': answer, 'score': score.score, 'metrics': dict(score.metrics)})
    return sum(row['score'] for row in rows) / len(rows), rows


def finish_run(config, output, manifest, model, selected, audit_rows, *, started, **extra):
    text = json.dumps(selected, sort_keys=True, ensure_ascii=False, default=str)
    frozen = (output / 'selected_candidate.json').read_text().rstrip('\n')
    if frozen != text:
        raise ValueError('Selected artifact changed during independent audit')
    (output / 'audit.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in audit_rows))
    summary = {'status': 'complete', 'method': config['method'], 'protocol': manifest['protocol'],
        'selected_candidate_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'audit_score': sum(row['score'] for row in audit_rows) / len(audit_rows),
        'model_revision': model.model_revision, 'meter': model.summary(),
        'wall_seconds': time.monotonic() - started, **extra}
    if isinstance(model.client, CappedClient):
        summary['token_budget_receipts'] = model.client.receipt()
        if any(not b['cost_valid'] or b['budget_breached'] for b in summary['token_budget_receipts'].values()):
            raise ValueError('Secondary run has invalid measured costs')
        (output / 'physical_budget_calls.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in model.client.calls))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def freeze_selection(output, selected, model):
    text = json.dumps(selected, sort_keys=True, ensure_ascii=False, default=str)
    (output / 'selected_candidate.json').write_text(text + '\n')
    if isinstance(model.client, CappedClient):
        model.client.phase = 'audit'


def record_failure(output, model, exception):
    report = {'status': 'failed', 'error': f'{type(exception).__name__}: {exception}', 'meter': model.summary()}
    if isinstance(model.client, CappedClient):
        report['token_budget_receipts'] = model.client.receipt()
        report['physical_budget_calls'] = model.client.calls
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
