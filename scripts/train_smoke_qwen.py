#!/usr/bin/env python3
"""Validate real BF16 Qwen3.5 SFT, separate v3 roles and exact checkpoint replay.

The two-item integer fixture is solely an implementation validation. Its
legal demonstration is constructed and executed here, never a benchmark run.
"""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from specrsi.core import AcquisitionCaps, BudgetLedger, Candidate, EvaluationScope, Ledger, PureIntegerExecutor, ResultAcquirer, Task
from specrsi.core.models import thaw
from specrsi.core.vats import VATSPlan, VATSState
from specrsi.learning.hf import HFModelConfig, HFRoleModel
from specrsi.learning.objectives import mixture_probabilities
from specrsi.learning.paper_bridge import VATSSearchBridge
from specrsi.learning.records import EditExample, SearchTrajectory, TransitionRecord
from specrsi.learning.trainer import SpecRSITrainer, TrainingConfig, _parameter_role
from specrsi.planning.completion import CompletionProblem, VerificationJob

REVISION = '59d61f3ce65a6d9863b86d2e96597125219dc754'


def native_kernel_dispatch():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as native
    result = {}
    for name in ('torch_chunk_gated_delta_rule', 'causal_conv1d_fn'):
        function = getattr(native, name)
        while function is not None:
            implementation = inspect.getclosurevars(function).nonlocals.get('implementation')
            if implementation is not None:
                result[name] = implementation.__module__ + '.' + implementation.__name__
                break
            function = getattr(function, '__wrapped__', None)
        else:
            result[name] = 'unidentified'
    return result


def loaded_software():
    import transformers
    import peft
    return {name: {'version': module.__version__, 'path': module.__file__}
            for name, module in (('torch', torch), ('transformers', transformers), ('peft', peft))}


def verified_fixture():
    """Execute the legal demonstration before constructing its verified label."""
    tasks = (Task('read', {'route': 'read', 'x': 2}, 2),
             Task('write', {'route': 'write', 'x': 2}, 3))
    scope = EvaluationScope('qwen35-training-validation-only', tasks, 'pure-int-v1',
        'integer-equality-v1', ('answer', 'primitives'), split='dev')
    parent = Candidate('fixture-parent', {'routes': {'read': ('x',), 'write': ('x',)}},
        {'contract': 'pure-int-v1'}, 'pure_integer', True)
    child = Candidate('fixture-repair', {'routes': {'read': ('x',),
        'write': ('add', ('x',), ('const', 1))}}, parent.runtime, 'pure_integer', True, parent_id=parent.uid)
    executor, ledger = PureIntegerExecutor(), Ledger()
    acquirer = ResultAcquirer(ledger, BudgetLedger(100), executor)
    for task in tasks:
        acquirer.acquire(parent, task, scope, (), AcquisitionCaps(10))
    ledger.register(child, scope)
    state = VATSState(scope, parent, {parent.uid: parent, child.uid: child}, ledger,
        BudgetLedger(1000), remaining_attempts=10, initial_attempts=10, workers=1)
    jobs = tuple(VerificationJob('fixture-check-' + task.uid, ((child.uid, task.uid),), 10) for task in tasks)
    # The v3 graph observes the fixed physical catalog before any value/policy
    # call, exactly as VATSEngine._prepare does. A list of chosen jobs alone
    # omits this required conditioning object and cannot form a v3 record.
    state.job_catalog = CompletionProblem(jobs, {'execution': 0}, state.budget.available, state.remaining_attempts)
    plans = (VATSPlan('verify', jobs=jobs, work_cap=20, attempt_cap=2), VATSPlan('accept'))
    observed = [executor.run(child, task, scope, 10) for task in tasks]
    if any(result.status != 'complete' or result.outcome.score != 1. for result in observed):
        raise RuntimeError('Smoke demonstration did not pass its actual integer checks')
    return state, plans, child, sum(result.actual_cost for result in observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-35B-A3B')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--output', default='artifacts/training/qwen35_paper33_validation')
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()
    if args.max_length < 256:
        parser.error('--max-length must be at least 256 for the registered observation fixture')
    directory = Path(args.model).resolve()
    lock_path = ROOT / 'configs/models/qwen35_35b_a3b_bf16.lock.json'
    lock = json.loads(lock_path.read_text())
    if lock['revision'] != REVISION or directory != (ROOT / lock['local_path']).resolve():
        raise ValueError('Validation requires the pinned official BF16 model lock and its local directory')
    raw = json.loads((directory / 'config.json').read_text())
    if raw.get('model_type') != 'qwen3_5_moe' or raw.get('quantization_config'):
        raise ValueError('Expected the official unquantized Qwen3.5 MoE descriptor')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    start = time.monotonic()
    print('Loading the pinned BF16 native text backbone for validation.', flush=True)
    model_config = HFModelConfig(model_path=str(directory), revision=REVISION, architecture_version=3,
        option_dim=12, feature_dim=8, head_hidden_dim=256, lora_rank=16, lora_alpha=32.,
        device=args.device, dtype='bfloat16', max_length=args.max_length, gradient_checkpointing=True,
        language_model_only=True, require_unquantized_backbone=True, lora_target_profile='qwen35_hybrid_attention',
        semantic_budget_mode='redistribute', semantic_global_tokens=128, semantic_node_min_tokens=16,
        semantic_node_max_tokens=128, semantic_total_tokens=1024, semantic_truncation='head_tail',
        graph_schema='actual_jobs_v2', role_observation_version='observed_results_v2', frozen_global_cache=True)
    roles = HFRoleModel(model_config).eval()
    assert all(parameter.dtype == torch.bfloat16 for name, parameter in roles.backbone.named_parameters()
               if 'lora_' not in name and parameter.is_floating_point())
    frozen_versions = {name: (id(parameter), parameter._version) for name, parameter in roles.backbone.named_parameters()
                       if 'lora_' not in name}
    state, plans, child, actual_cost = verified_fixture()
    prompt = 'Return only the repaired JSON artifact. Preserve the read route and add one to x in the write route.\nParent: ' + json.dumps(thaw(state.incumbent.artifacts)) + '\nRepair: '
    prompt_ids = roles.tokenizer.encode(prompt, add_special_tokens=True)
    continuation = roles.tokenizer.encode(json.dumps(thaw(child.artifacts)), add_special_tokens=False)
    eos = roles.backbone.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    if eos is None:
        raise ValueError('Pinned model must declare an EOS for this supervised continuation')
    continuation.append(eos)
    edit = EditExample('executed-integer-validation-only', tuple(prompt_ids + continuation),
        tuple([-100] * len(prompt_ids) + continuation), True)
    print('Encoding the v3 fixture, checking frozen backbone identity, and taking one SFT/value step.', flush=True)
    bridge = VATSSearchBridge(roles, cost_unit='tokens')
    plan = bridge._record(state, plans)
    probabilities = mixture_probabilities(roles.decision_logits(plan), plan.temperature, plan.exploration).detach()
    plan = replace(plan, selected_index=0, deterministic=False, behavior_logprob=float(probabilities[0].log()),
        behavior_probabilities=tuple(probabilities.tolist()))
    # F is improvement relative to the fully evaluated root, not raw task score.
    # The fixture's two actual checks establish gain .5 for this full VERIFY.
    trajectory = SearchTrajectory('validation-only', 'fixture-parent', 'executed-repair', 0,
        (TransitionRecord(0., .5, float(actual_cost)),), (plan,), initial_f=0., final_f=.5,
        demonstration=True, collection_snapshot='validation-initial', contract_id=state.scope.uid,
        initial_budget=1000., workers=1)
    training = TrainingConfig(optimizer_rule='pdf_gradient_v1', learning_rate=2e-5,
        actor_learning_rate=2e-5, value_learning_rate=2e-5, weight_decay=0., cost_coefficient=0.,
        batch_size=3, surface_weight=0., max_grad_norm=0.)
    trainer = SpecRSITrainer(roles, training)
    update = trainer.warmup_epoch(edits=(edit,), plans=(plan,), trajectories=(trajectory,))
    assert update['update_schedule']['actual_optimizer_steps'] == 1
    for role in ('draft', 'policy', 'value'):
        assert update['learning_signal']['roles'][role]['signal_parameter_change_steps'] == 1, role
    violations = [name for name, parameter in roles.backbone.named_parameters() if 'lora_' not in name and
        (parameter.requires_grad or parameter.grad is not None or (id(parameter), parameter._version) != frozen_versions[name])]
    assert not violations
    counts = {}
    for name, parameter in roles.named_parameters():
        if parameter.requires_grad:
            role = _parameter_role(name)
            counts[role] = counts.get(role, 0) + parameter.numel()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        expected = torch.cat((roles.decision_values(plan), roles.decision_logits(plan), roles.edit_nll(edit).reshape(1))).cpu()
    trainer.save(output / 'checkpoint')
    report = {'validation_only': True, 'benchmark_run': False, 'fixture': 'executed_two_item_integer_repair',
        'model_path': str(directory), 'model_revision': REVISION, 'model_lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        'model_config': asdict(model_config), 'training_config': asdict(training),
        'native_class': type(roles.backbone.get_base_model()).__name__,
        'experts_implementation': roles.backbone.get_base_model().config._experts_implementation,
        'native_kernel_dispatch': native_kernel_dispatch(),
        'resolved_lora_targets': list(roles.resolved_lora_targets), 'trainable_parameters_by_role': counts,
        'frozen_backbone_parameters': sum(p.numel() for name, p in roles.backbone.named_parameters() if 'lora_' not in name),
        'frozen_backbone_violations': violations, 'learning_signal': update['learning_signal'],
        'update_schedule': update['update_schedule'], 'supervised_tokens_including_eos': len(continuation),
        'gpu': torch.cuda.get_device_name(roles.device), 'peak_gpu_memory_allocated_bytes': torch.cuda.max_memory_allocated(roles.device),
        'peak_gpu_memory_reserved_bytes': torch.cuda.max_memory_reserved(roles.device),
        'software': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'tokenizers', 'huggingface-hub')},
        'loaded_software': loaded_software(), 'torch_cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
        'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    # Drop every owner before reloading: the 35B base must never exist twice on
    # the training GPU. The frozen semantic cache and bridge both own the model.
    del trainer, roles, bridge, probabilities
    gc.collect()
    torch.cuda.empty_cache()
    print('Reloading the single native backbone and saved roles for exact replay.', flush=True)
    reloaded = HFRoleModel.load(output / 'checkpoint', device=args.device).eval()
    with torch.no_grad():
        actual = torch.cat((reloaded.decision_values(plan), reloaded.decision_logits(plan), reloaded.edit_nll(edit).reshape(1))).cpu()
    difference = float((expected - actual).abs().max())
    report['save_reload_max_abs_difference'] = difference
    report['save_reload_exact'] = torch.equal(expected, actual)
    assert difference < 1e-5
    report['elapsed_seconds'] = time.monotonic() - start
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
