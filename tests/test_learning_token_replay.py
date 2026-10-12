"""Causal edit replay must preserve the recorded GVO objective and gradients."""
from copy import deepcopy
from dataclasses import replace
import random

import pytest
import torch

from specrsi.learning import SpecRSITrainer, TrainingConfig
from specrsi.learning.records import SearchTrajectory, TransitionRecord


def tiny_roles():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from specrsi.learning.hf import HFModelConfig, HFRoleModel

    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8, eos_token_id=31))
    return HFRoleModel(HFModelConfig(model_path="__token_replay_test__", device="cpu", dtype="float32",
        lora_rank=2, lora_alpha=4, gradient_checkpointing=False, max_length=32), base_model=base).eval()


def sampled_edit(model, seed=0):
    from specrsi.learning.sampler import sample_edit_tokens

    # The first and last grammar tokens are deterministic. The two sampled
    # middle choices still use their complete original prefixes and supports.
    grammar = lambda prefix: (5,) if not prefix else (31,) if len(prefix) == 3 else (6, 7, 8)
    return sample_edit_tokens(model, (1, 2, 3), max_new_tokens=4, rng=random.Random(seed), top_k=0,
        temperature=.7, exploration=.2, proposal_snapshot="fixed-collection", token_grammar=grammar)


def test_sft_projects_only_supervised_positions_with_exact_full_vocabulary_gradient():
    from specrsi.learning.records import EditExample
    roles = tiny_roles()
    example = EditExample('root', (1, 2, 3, 4, 5, 6), (-100, -100, 3, -100, 5, 6), True)
    roles._activate('draft')
    tokens = roles._tokens(input_ids=example.input_ids)
    full = roles.backbone(**tokens, use_cache=False, return_dict=True).logits[0, :-1].float()
    expected = torch.nn.functional.cross_entropy(full, torch.tensor(example.labels[1:]), reduction='sum')
    expected.backward()
    gradients = {name: p.grad.clone() for name, p in roles.named_parameters() if p.grad is not None}
    roles.zero_grad(set_to_none=True)
    positions = []
    hook = roles.backbone.register_forward_pre_hook(
        lambda module, args, kwargs: positions.append(kwargs['logits_to_keep'].tolist()), with_kwargs=True)
    actual = roles.edit_nll(example)
    hook.remove()
    actual.backward()
    assert positions == [[1, 3, 4]]
    assert torch.allclose(actual, expected, atol=1e-6)
    assert any(torch.count_nonzero(value) for value in gradients.values())
    for name, parameter in roles.named_parameters():
        if name in gradients:
            assert torch.allclose(parameter.grad, gradients[name], atol=2e-6, rtol=2e-5), name


def test_causal_batch_preserves_fixed_support_probabilities_and_token_gradients():
    roles = tiny_roles()
    sample = sampled_edit(roles)
    trainer = SpecRSITrainer(roles, TrainingConfig(kl_weight=.3))
    advantage = torch.tensor(-.7)
    reference = []
    for decision in sample.decisions:
        reference.append(roles.decision_logits(decision).detach())
        if not decision.deterministic:
            actor, kl = trainer._actor_terms(decision, advantage)
            (actor + .3 * kl).backward()
    expected_grads = {name: p.grad.clone() for name, p in roles.named_parameters() if p.grad is not None}
    roles.zero_grad(set_to_none=True)

    calls = []
    hook = roles.backbone.register_forward_pre_hook(lambda module, args, kwargs:
        calls.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True)
    rows = roles.edit_token_logits(sample.decisions)
    hook.remove()
    assert calls == [(1, len(sample.decisions[-1].input_ids))]
    assert all(torch.allclose(row, original, atol=1e-6) for row, original in zip(rows, reference))
    losses = []
    for decision, row in zip(sample.decisions, rows):
        if not decision.deterministic:
            actor, kl = trainer._actor_terms(decision, advantage, row)
            losses.append(actor + .3 * kl)
    torch.stack(losses).sum().backward()
    actual_grads = {name: p.grad for name, p in roles.named_parameters() if p.grad is not None}
    assert expected_grads.keys() == actual_grads.keys()
    assert any(torch.count_nonzero(value) for value in actual_grads.values())
    for name, expected in expected_grads.items():
        assert torch.allclose(actual_grads[name], expected, atol=2e-6, rtol=2e-5), name
    assert all(p.grad is None for p in roles.value_head.parameters())


def test_token_batch_rejects_cross_edit_prefixes_and_trainer_keeps_them_separate():
    roles = tiny_roles()
    sample = sampled_edit(roles)
    trainer = SpecRSITrainer(roles, TrainingConfig())
    with pytest.raises(ValueError, match="contiguous"):
        roles.edit_token_logits((sample.decisions[0], sample.decisions[2]))
    two_edits = sample.decisions + sample.decisions
    assert tuple(trainer._replay_units(two_edits)) == (sample.decisions, sample.decisions)
    next_work = tuple(replace(d, transition_index=1) for d in sample.decisions)
    assert tuple(trainer._replay_units(sample.decisions + next_work)) == (sample.decisions, next_work)


def test_gvo_batched_replay_matches_prefix_optimizer_update_and_uses_one_forward_per_edit():
    roles = tiny_roles()
    samples = [sampled_edit(roles, seed) for seed in (1, 2)]
    groups = [[SearchTrajectory("group", "root", f"search-{i}", i,
        (TransitionRecord(0., float(i), 4.),), sample.decisions, final_f=float(i))
        for i, sample in enumerate(samples)]]
    reference = deepcopy(roles)

    class PrefixReplay:
        # Expose only the original single-decision role interface.
        device = reference.device
        parameters = reference.parameters
        eval = reference.eval
        decision_logits = reference.decision_logits
        decision_values = reference.decision_values

    config = TrainingConfig(value_weight=0., kl_weight=.1, learning_rate=1e-3, weight_decay=0.)
    calls = []
    hook = roles.backbone.register_forward_pre_hook(lambda module, args, kwargs:
        calls.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True)
    actual = SpecRSITrainer(roles, config).gvo_epoch(groups)
    hook.remove()
    expected = SpecRSITrainer(PrefixReplay(), config).gvo_epoch(groups)
    assert len(calls) == 2
    assert actual["stochastic_decisions"] == expected["stochastic_decisions"] == 4
    assert actual["actor_loss"] == pytest.approx(expected["actor_loss"], abs=1e-6)
    assert actual["kl"] == pytest.approx(expected["kl"], abs=1e-6)
    for (name, actual_parameter), (_, expected_parameter) in zip(roles.named_parameters(), reference.named_parameters()):
        assert torch.allclose(actual_parameter, expected_parameter, atol=2e-6, rtol=2e-5), name
