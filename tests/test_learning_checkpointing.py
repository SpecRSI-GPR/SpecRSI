"""Real tiny HF blocks must checkpoint deterministic adapter training, not just advertise it."""
import random
from dataclasses import replace

import pytest
import torch

from specrsi.learning import SpecRSITrainer, TrainingConfig
from specrsi.learning.hf import HFModelConfig, HFRoleModel
from specrsi.learning.records import EditExample, SearchTrajectory, TransitionRecord


def roles(checkpointing):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(718)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        attention_dropout=.45, eos_token_id=31))
    model = HFRoleModel(HFModelConfig(model_path="__checkpoint_eval__", device="cpu", dtype="float32",
        lora_rank=2, lora_alpha=4, lora_dropout=.55, gradient_checkpointing=checkpointing,
        max_length=32), base_model=base).eval()
    # Nonzero and different adapter weights make a wrong recomputation role visible.
    with torch.no_grad():
        for name, parameter in model.backbone.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=.05 if ".draft." in name else -.04, std=.03)
    return model


def block_calls(model):
    from transformers.modeling_layers import GradientCheckpointingLayer

    calls = []
    handles = [module.register_forward_pre_hook(lambda module, args: calls.append(module.training))
               for module in model.backbone.modules() if isinstance(module, GradientCheckpointingLayer)]
    return calls, handles


def grad_snapshot(model):
    return {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}


def assert_gradients_equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert any(torch.count_nonzero(value) for value in actual.values())
    for name in actual:
        assert torch.allclose(actual[name], expected[name], atol=2e-6, rtol=3e-5), name


@pytest.mark.parametrize("role", ["draft", "verify"])
def test_eval_recomputes_each_block_and_matches_noncheckpoint_gradients(role):
    reference, model = roles(False), roles(True)
    calls, handles = block_calls(model)
    example = EditExample("root", (1, 2, 3, 4), (-100, -100, 3, 4), True)

    def loss(item):
        return item.edit_nll(example) if role == "draft" else item.surface_head(item.encode("", "verify", (1, 2, 3)))

    expected = loss(reference)
    expected.backward()
    actual = loss(model)
    assert torch.equal(actual.detach(), loss(model).detach())  # eval dropout stays disabled.
    calls.clear()
    # Count a fresh graph's forward and backward, not the preceding comparison forwards.
    actual = loss(model)
    actual.backward()
    assert len(calls) == 4  # two blocks, each actually recomputed once.
    assert calls == [False] * 4  # real block bodies execute in eval mode.
    assert torch.allclose(actual, expected, atol=1e-6)
    assert_gradients_equal(grad_snapshot(model), grad_snapshot(reference))
    assert all(not module.training for module in model.modules())
    assert all(parameter.grad is None for name, parameter in model.backbone.named_parameters() if "lora_" not in name)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_delayed_backward_restores_forward_adapter_and_callers_current_state(checkpointing):
    reference, model = roles(False), roles(checkpointing)
    example = EditExample("root", (1, 2, 3, 4), (-100, -100, 3, 4), True)
    expected = reference.edit_nll(example)
    expected.backward()
    actual = model.edit_nll(example)
    model._activate("verify")
    with torch.no_grad():
        model.encode("", "value", (1, 2, 3))
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    actual.backward()
    assert_gradients_equal(grad_snapshot(model), grad_snapshot(reference))
    assert model.backbone.active_adapters == ["verify"]
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == before
    assert all(not module.training for module in model.modules())


def test_two_pending_role_graphs_recompute_with_their_own_adapter():
    reference, model = roles(False), roles(True)

    def combined(item):
        draft = item.edit_nll(EditExample("root", (1, 2, 3, 4), (-100, -100, 3, 4), True))
        verify = item.surface_head(item.encode("", "verify", (1, 2, 3)))
        return draft + verify

    combined(reference).backward()
    combined(model).backward()
    assert_gradients_equal(grad_snapshot(model), grad_snapshot(reference))
    assert any(".draft." in name for name in grad_snapshot(model))
    assert any(".verify." in name for name in grad_snapshot(model))


def sampled_edit(model, seed):
    from specrsi.learning.sampler import sample_edit_tokens

    grammar = lambda prefix: (5,) if not prefix else (31,) if len(prefix) == 3 else (6, 7, 8)
    return sample_edit_tokens(model, (1, 2, 3), max_new_tokens=4, rng=random.Random(seed), top_k=0,
        temperature=.7, exploration=.2, proposal_snapshot="checkpoint-test", token_grammar=grammar)


def test_no_grad_collection_and_frozen_features_keep_eval_path_and_exact_replay():
    from specrsi.learning.objectives import mixture_probabilities

    model = roles(True)
    calls, handles = block_calls(model)
    with torch.no_grad():
        features = model.encode("", "value", (1, 2, 3))
    assert not features.requires_grad
    assert calls == [False, False]
    sample = sampled_edit(model, 7)
    for decision, row in zip(sample.decisions, model.edit_token_logits(sample.decisions)):
        if not decision.deterministic:
            probability = mixture_probabilities(row, decision.temperature, decision.exploration)[decision.selected_index]
            assert float(probability.log().detach()) == pytest.approx(decision.behavior_logprob, abs=2e-6)
    assert all(not module.training for module in model.modules())
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("stage", ["warmup", "gvo"])
def test_real_trainer_recomputes_and_preserves_optimizer_update(stage):
    reference, model = roles(False), roles(True)
    config = TrainingConfig(value_weight=0., kl_weight=.1, learning_rate=1e-3, weight_decay=0.)
    calls, handles = block_calls(model)
    if stage == "warmup":
        args = {"edits": [EditExample("root", (1, 2, 3, 4), (-100, -100, 3, 4), True)]}
        expected = SpecRSITrainer(reference, config).warmup_epoch(**args)
        actual = SpecRSITrainer(model, config).warmup_epoch(**args)
        assert len(calls) == 4
        assert actual["edit_loss"] == pytest.approx(expected["edit_loss"], abs=1e-6)
    else:
        samples = [sampled_edit(reference, seed) for seed in (1, 2)]
        groups = [[SearchTrajectory("group", "root", f"search-{i}", i,
            (TransitionRecord(0., float(i), 4.),), sample.decisions, final_f=float(i))
            for i, sample in enumerate(samples)]]
        expected = SpecRSITrainer(reference, config).gvo_epoch(groups)
        actual = SpecRSITrainer(model, config).gvo_epoch(groups)
        assert len(calls) == 8
        assert actual["actor_loss"] == pytest.approx(expected["actor_loss"], abs=1e-6)
    for (name, parameter), (_, original) in zip(model.named_parameters(), reference.named_parameters()):
        assert torch.allclose(parameter, original, atol=2e-6, rtol=3e-5), name
    assert all(not module.training for module in model.modules())
    for handle in handles:
        handle.remove()


def test_failed_forward_restores_eval_mode():
    from transformers.modeling_layers import GradientCheckpointingLayer

    model = roles(True)
    layer = next(module for module in model.modules() if isinstance(module, GradientCheckpointingLayer))

    def fail(module, args):
        raise RuntimeError("injected forward failure")

    handle = layer.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        model.edit_nll(EditExample("root", (1, 2, 3), (-100, -100, 3), True))
    handle.remove()
    assert all(not module.training for module in model.modules())
    assert not layer._specrsi_evaluation_gate


def test_failed_recomputation_restores_callers_adapter_and_eval_mode():
    from transformers.modeling_layers import GradientCheckpointingLayer

    model = roles(True)
    loss = model.edit_nll(EditExample("root", (1, 2, 3), (-100, -100, 3), True))
    model._activate("verify")
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    layer = next(module for module in model.modules() if isinstance(module, GradientCheckpointingLayer))

    def fail(module, args):
        raise RuntimeError("injected recomputation failure")

    handle = layer.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected recomputation failure"):
        loss.backward()
    handle.remove()
    assert model.backbone.active_adapters == ["verify"]
    assert all(not module.training for module in model.modules())
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == flags


def test_explicit_training_mode_retains_seeded_dropout_semantics():
    reference, model = roles(False).train(), roles(True).train()
    example = EditExample("root", (1, 2, 3, 4), (-100, -100, 3, 4), True)
    torch.manual_seed(73)
    expected = reference.edit_nll(example)
    expected.backward()
    torch.manual_seed(73)
    actual = model.edit_nll(example)
    actual.backward()
    assert torch.allclose(actual, expected, atol=1e-6)
    assert_gradients_equal(grad_snapshot(model), grad_snapshot(reference))
    assert all(module.training for module in model.modules())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_replay_projects_only_recorded_positions_with_identical_fixed_support_gradients(checkpointing):
    reference, model = roles(checkpointing), roles(checkpointing)
    reference._supports_logit_positions = False
    sample = sampled_edit(reference, 7)
    projected = []
    handle = model.backbone.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: projected.append(tuple(args[0].shape)))
    expected = reference.edit_token_logits(sample.decisions)
    actual = model.edit_token_logits(sample.decisions)
    assert projected == [(1, len(sample.decisions), 16)]
    assert len(sample.decisions) < len(sample.decisions[-1].input_ids)
    assert sample.generated_ids[-1] == 31  # EOS position is still replayed.
    for left, right in zip(actual, expected):
        assert torch.allclose(left, right, atol=1e-6)
    sum(row.logsumexp(0) for row in expected).backward()
    sum(row.logsumexp(0) for row in actual).backward()
    assert_gradients_equal(grad_snapshot(model), grad_snapshot(reference))
    handle.remove()


def full_vocabulary_sft_reference(model, example):
    from specrsi.learning.objectives import mixture_probabilities

    model._activate("draft")
    tokens = model._tokens(input_ids=example.input_ids)
    with model._gradient_forward():
        logits = model.backbone(**tokens, use_cache=False).logits[0, :-1].float()
    losses = []
    for decision in example.token_decisions:
        if not decision.deterministic:
            ids = torch.tensor([int(token) for token in decision.support])
            row = logits[len(decision.input_ids) - 1, ids]
            exploration = row.new_tensor(decision.exploration_probabilities) if decision.exploration_probabilities else None
            probabilities = mixture_probabilities(row, decision.temperature, decision.exploration, exploration)
            losses.append(-probabilities[decision.selected_index].log())
    return torch.stack(losses).sum() if losses else logits.sum() * 0.


@pytest.mark.parametrize("checkpointing", [False, True])
def test_observed_sft_projects_sampled_positions_before_float_conversion(checkpointing):
    reference, model = roles(checkpointing), roles(checkpointing)
    sample = sampled_edit(reference, 1)
    # Exercise the stored nonuniform exploration distribution as well.
    decisions = tuple(replace(decision, exploration_probabilities=(.2, .3, .5))
                      if not decision.deterministic else decision for decision in sample.decisions)
    example = EditExample("root", sample.prompt_ids + sample.generated_ids,
        (-100,) * len(sample.prompt_ids) + sample.generated_ids, True, token_decisions=decisions)
    projected = []
    handle = model.backbone.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: projected.append(tuple(args[0].shape)))
    expected, actual = full_vocabulary_sft_reference(reference, example), model.edit_nll(example)
    assert projected == [(1, 2, 16)]  # only the two non-forced positions.
    assert torch.allclose(actual, expected, atol=1e-6)
    expected.backward()
    actual.backward()
    assert_gradients_equal(grad_snapshot(model), grad_snapshot(reference))
    handle.remove()


def test_offline_ce_remains_full_vocab_and_frozen_context_avoids_lm_head():
    model = roles(True)
    projected = []
    handle = model.backbone.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: projected.append(tuple(args[0].shape)))
    model.encode("", "value", (1, 2, 3))
    model.encode("", "policy", (1, 2, 3))
    model.encode("", "verify", (1, 2, 3))
    assert not projected
    loss = model.edit_nll(EditExample("root", (1, 2, 3, 4), (-100, -100, 3, 4), True))
    assert projected == [(1, 4, 16)]
    assert torch.isfinite(loss)
    loss.backward()
    handle.remove()


def test_all_forced_sft_still_has_zero_connected_loss_with_one_projected_position():
    model = roles(True)
    sample = sampled_edit(model, 2)
    decisions = tuple(replace(decision, support=(decision.support[decision.selected_index],), selected_index=0,
        deterministic=True, behavior_logprob=None, behavior_probabilities=(1.,),
        exploration_probabilities=(1.,)) for decision in sample.decisions)
    example = EditExample("root", sample.prompt_ids + sample.generated_ids,
        (-100,) * len(sample.prompt_ids) + sample.generated_ids, True, token_decisions=decisions)
    projected = []
    handle = model.backbone.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: projected.append(tuple(args[0].shape)))
    loss = model.edit_nll(example)
    assert projected == [(1, 1, 16)]
    assert loss.requires_grad and loss.item() == 0.
    loss.backward()
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters())
    handle.remove()
