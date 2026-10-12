#!/usr/bin/env python3
"""Bounded, development-only native 35B resource validation; no model updates.

Time the complete v3 scorer/policy APIs, including record and frozen semantic
encodings. Separately backpropagate through a maximum-length computational
fixture. No benchmark metric, trained checkpoint, or heldout observation is
produced. Fresh randomly initialized role heads measure the registered shapes.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import yaml
from specrsi.core import BudgetLedger
from specrsi.core.models import thaw
from specrsi.core.vats import VATSEngine, VATSPlan
from specrsi.learning.frozen_text import node_text_cache
from specrsi.learning.hf import HFModelConfig, HFRoleModel
from specrsi.learning.paper_bridge import VATSSearchBridge
from specrsi.science.executor import (ScienceExecutor, ParameterDrafter, make_scope,
    make_candidate, make_science_ledger, mutate)
from specrsi.science_comparison import _options
from train_smoke_qwen import REVISION, loaded_software, native_kernel_dispatch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def development_state(config, emit):
    """Run only the registered root on development 0/1; never score the child."""
    collection = config['collection']
    domain = collection['domain']
    scope = make_scope(domain, 'development', (0, 1), native_utility=True)
    root = make_candidate(domain)
    ledger, executor = make_science_ledger(), ScienceExecutor()
    for task in scope.tasks:
        started = time.monotonic()
        emit({'event': 'native_development_start', 'domain': domain, 'case_id': task.inputs['case_id'],
              'scope_id': scope.uid, 'cap_ms': collection['execution_cap_ms']})
        result = executor.run(root, task, scope, collection['execution_cap_ms'])
        emit({'event': 'native_development_finish', 'case_id': task.inputs['case_id'],
              'elapsed_seconds': time.monotonic() - started, 'status': result.status,
              'actual_cost_ms': result.actual_cost, 'error': result.error,
              'receipt': thaw(result.receipt), 'admitted': bool(result.outcome and result.outcome.admissible)})
        if result.status != 'complete' or not result.outcome or not result.outcome.admissible:
            raise RuntimeError('Development root incomplete/inadmissible; cannot fabricate role observations')
        ledger.admit_direct(root, task, scope, result.outcome, result.receipt)
    options = _options({'search': collection, 'seed': config['training']['seed']}, 'resource-validation-only')
    engine = VATSEngine(scope, root, ledger, BudgetLedger(collection['budget_ms']), executor,
        ParameterDrafter(), options)
    # A real legal parameter mutation gives a missing-result candidate. Its
    # outcomes remain unobserved: only the development root was executed.
    child_ids = []
    for index in range(collection['max_candidates'] - 1):
        parent = next(candidate for candidate in engine.state.candidates.values()
            if engine.state.children_drafted.get(candidate.uid, 0) < collection['max_children'])
        child = make_candidate(domain, mutate(domain, thaw(parent.artifacts['parameters']), random.Random(index)),
            f'resource-profile-unexecuted-child-{index}', parent.uid)
        ledger.register(child, scope)
        engine.state.candidates[child.uid] = child
        child_ids.append(child.uid)
        engine.state.children_drafted[parent.uid] = engine.state.children_drafted.get(parent.uid, 0) + 1
    state = engine._conservative_state()
    state.job_catalog = engine._catalog(state)
    jobs = state.job_catalog.jobs
    if not jobs or not all(key[0] in child_ids for job in jobs for key in job.keys):
        raise AssertionError('Expected only actually missing child verification jobs')
    work_cap = sum(job.cap for job in jobs) + sum(state.job_catalog.setup_costs.values())
    verify = VATSPlan('verify', jobs=jobs, work_cap=work_cap, attempt_cap=sum(j.attempts for j in jobs))
    # Saturate the registered tensor menu cap with distinct feasible bundles.
    # This is a head-resource pressure menu, not a claim that Algorithm 2's
    # registered seed quotas would emit every singleton simultaneously.
    from itertools import combinations
    plans = [verify]
    for size in range(1, len(jobs)):
        for selected in combinations((job.uid for job in jobs), size):
            if len(plans) >= options.max_menu_options - 1:
                break
            if state.job_catalog.feasible(selected):
                plans.append(engine._verify_plan(selected, state.job_catalog, 'resource-pressure-feasible-menu'))
        if len(plans) >= options.max_menu_options - 1:
            break
    plans = (*plans, VATSPlan('accept'))
    emit({'event': 'development_fixture_ready', 'domain': domain, 'scenario_set': 'development',
          'case_ids': [0, 1], 'scope_id': scope.uid, 'root_id': root.uid, 'child_ids': child_ids,
          'candidate_count': len(state.candidates), 'menu_size': len(plans),
          'menu_contract': 'registered cap saturated by feasible work bundles; resource pressure only',
          'children_executed': False, 'job_catalog': asdict(state.job_catalog),
          'plans': [plan.manifest() for plan in plans]})
    return state, plans, options


def computational_nll(model, input_ids, supervised_start):
    """Raw full-vocabulary CE pressure, with no EditExample or verified label.

    This mirrors only the numerical kernel of Eq. (18), not demonstration
    admission: a resource fixture cannot assert that an edit was verified.
    Position j-1 predicts observed label j; the loss sums full-vocabulary NLL.
    """
    if not 1 <= supervised_start < len(input_ids):
        raise ValueError('Computational loss needs a nonempty prompt and label suffix')
    model._activate('draft')
    tokens = model._tokens(input_ids=tuple(input_ids))
    positions = tuple(range(supervised_start - 1, len(input_ids) - 1))
    logits = model._token_position_logits(tokens, positions).float()
    labels = torch.tensor(input_ids[supervised_start:], dtype=torch.long, device=model.device)
    return torch.nn.functional.cross_entropy(logits, labels, reduction='sum')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training/qwen35_paper33_science.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='reports/qwen35_paper33_20261010/registered_context_profile')
    parser.add_argument('--timeout-seconds', type=int, default=1200)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    events = []
    started = time.monotonic()

    def emit(row):
        row = thaw({'elapsed_total_seconds': time.monotonic() - started, **row})
        events.append(row)
        with (output / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    def timeout(signum, frame):
        raise TimeoutError('Bounded registered-context resource profile timed out')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    report = {'validation_only': True, 'benchmark_run': False, 'optimizer_steps': 0,
              'checkpoint_written': False, 'heldout_accessed': False, 'status': 'running',
              'config_path': args.config, 'config_sha256': sha(args.config),
              'script_sha256': sha(__file__), 'timeout_seconds': args.timeout_seconds,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'role_initialization': 'fresh seeded random LoRA/heads; frozen official BF16 backbone'}
    try:
        config = yaml.safe_load(Path(args.config).read_text())
        raw = dict(config['model'])
        assert raw.pop('backend') == 'hf'
        raw['device'] = args.device
        model_config = HFModelConfig(**raw)
        assert model_config.max_length == 2048
        assert model_config.revision == REVISION
        assert model_config.require_unquantized_backbone and model_config.language_model_only
        assert model_config.experts_implementation == 'eager'
        assert model_config.gradient_checkpointing and model_config.frozen_global_cache
        report['model_config'] = asdict(model_config)
        report['software'] = loaded_software()
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(config['training']['seed'])
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        report['gpu_name'] = torch.cuda.get_device_name(device)
        report['gpu_total_bytes'] = torch.cuda.get_device_properties(device).total_memory
        state, plans, options = development_state(config, emit)
        report['scorer_query_cap_ms'] = options.scorer_query_cap
        report['policy_prepaid_quota_ms'] = options.policy_call_cost
        emit({'event': 'model_load_start'})
        load_started = time.monotonic()
        roles = HFRoleModel(model_config).eval()
        torch.cuda.synchronize(device)
        emit({'event': 'model_load_finish', 'elapsed_seconds': time.monotonic() - load_started,
              'allocated_bytes': torch.cuda.memory_allocated(device),
              'native_class': type(roles.backbone.get_base_model()).__name__,
              'experts_implementation': roles.backbone.get_base_model().config._experts_implementation,
              'native_kernel_dispatch': native_kernel_dispatch()})
        frozen_versions = {name: (id(p), p._version) for name, p in roles.backbone.named_parameters() if 'lora_' not in name}
        bridge = VATSSearchBridge(roles, cost_unit='wall-ms', snapshot_id='resource-validation-only')
        for role in ('scorer', 'policy'):
            for repetition in range(4):
                cold = repetition in (0, 2)
                if cold:
                    node_text_cache(roles)._cache.clear()
                cache_label = ('cold_semantic_cache' if repetition == 0 else
                    'cold_semantic_cache_after_kernel_warmup' if cold else 'warm_semantic_cache')
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                first_receipt = len(bridge.physical_inferences)
                emit({'event': 'full_role_call_start', 'role': role, 'cache': cache_label, 'repetition': repetition})
                call_started = time.monotonic()
                result = bridge.scorer(state, plans[0]) if role == 'scorer' else bridge.policy_menu(state, plans)
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - call_started
                row = {'event': 'full_role_call_finish', 'role': role, 'cache': cache_label, 'repetition': repetition,
                    'complete_call_elapsed_seconds': elapsed, 'includes_record_node_global_and_head': True,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                    'physical_inferences': bridge.physical_inferences[first_receipt:]}
                if role == 'scorer':
                    row.update(actual_search_charge_ms=result.actual_cost, scorer_cap_ms=options.scorer_query_cap,
                        would_exceed_registered_cap=result.actual_cost > options.scorer_query_cap)
                else:
                    row.update(prepaid_policy_quota_ms=options.policy_call_cost,
                        actual_physical_seconds=elapsed, menu_size=len(plans))
                emit(row)
        # Deliberately computational pressure data, never an admitted verified
        # edit or trainer demonstration. No optimizer or checkpoint is created.
        seed_tokens = roles.tokenizer.encode('Resource validation only. This token sequence is not a training demonstration.\n', add_special_tokens=False)
        ids = (seed_tokens * (2048 // len(seed_tokens) + 1))[:2048]
        eos = roles.backbone.generation_config.eos_token_id
        ids[-1] = eos[0] if isinstance(eos, list) else eos
        for repetition in range(2):
            roles.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            emit({'event': 'context_backward_start', 'repetition': repetition, 'input_tokens': 2048,
                  'supervised_tokens': 1920, 'gradient_checkpointing': True, 'optimizer_step': False})
            step_started = time.monotonic()
            loss = computational_nll(roles, ids, 128)
            torch.cuda.synchronize(device)
            forward_seconds = time.monotonic() - step_started
            backward_started = time.monotonic()
            loss.backward()
            torch.cuda.synchronize(device)
            active = [p.grad for n, p in roles.named_parameters() if '.draft.' in n and p.grad is not None]
            if not active or not all(torch.isfinite(g).all().item() for g in active) or not any(torch.count_nonzero(g).item() for g in active):
                raise AssertionError('Maximum-context backward lacks finite nonzero Draft gradients')
            violations = [name for name, p in roles.backbone.named_parameters() if 'lora_' not in name and
                (p.requires_grad or p.grad is not None or frozen_versions[name] != (id(p), p._version))]
            if violations:
                raise AssertionError('Frozen backbone changed: ' + repr(violations[:5]))
            emit({'event': 'context_backward_finish', 'repetition': repetition,
                  'forward_seconds': forward_seconds, 'backward_and_gradient_check_seconds': time.monotonic() - backward_started,
                  'input_tokens': 2048, 'supervised_tokens': 1920, 'finite_nonzero_draft_gradient': True,
                  'frozen_backbone_violations': violations, 'optimizer_step': False,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
            del loss, active
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        emit({'event': 'profile_error', 'error': repr(error), 'traceback': report['traceback']})
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - started
        report['events'] = events
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
