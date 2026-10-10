"""Analytic Eq23 updates and independent Eq21 reference, without a real LLM."""
from dataclasses import asdict, replace
import json
import math

import pytest
import torch

from specrsi.learning.objectives import mixture_probabilities
from specrsi.learning.records import DecisionRecord, SearchTrajectory, TransitionRecord
from specrsi.learning.trainer import SpecRSITrainer, TrainingConfig
from specrsi.learning.reference_policy import FrozenActorReference

torch.set_num_threads(2)


class ScalarRoles(torch.nn.Module):
    device = torch.device("cpu")
    config = {"base_revision": "scalar-fixture-v1", "tuple_roundtrip": ("draft", "policy")}

    def __init__(self):
        super().__init__()
        self.policy_head = torch.nn.Linear(1, 1, bias=False)
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        self.draft_adapter = torch.nn.Linear(1, 1, bias=False)
        for parameter in self.parameters():
            torch.nn.init.zeros_(parameter)

    def decision_logits(self, decision):
        head = self.draft_adapter if decision.kind == "edit_token" else self.policy_head
        weight = head.weight.flatten()[0]
        return torch.stack((weight, -weight))

    def decision_values(self, decision):
        return self.value_head.weight.flatten()[0].expand(2)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "weights.pt")


def decision(model, selected=0, *, temperature=1., exploration=0.):
    record = DecisionRecord("plan", ("draft", "verify"), selected, math.log(.5),
                            temperature=temperature, exploration=exploration)
    probabilities = mixture_probabilities(model.decision_logits(record), temperature, exploration).detach()
    return replace(record, behavior_probabilities=tuple(probabilities.tolist()),
                   behavior_logprob=float(probabilities[selected].log()))


def groups(model, returns=(1., 0.), **kwargs):
    return [[SearchTrajectory("root-group", "same-root", f"search-{i}", i,
        (TransitionRecord(0., gain, 0.),), (decision(model, i, **kwargs),), final_f=gain,
        collection_snapshot="current-collection", contract_id="fixed-contract", initial_budget=100., workers=1)
        for i, gain in enumerate(returns)]]


def config(**kwargs):
    return TrainingConfig(optimizer_rule="pdf_gradient_v1", weight_decay=0.,
                          learning_rate=.1, cost_coefficient=0., **kwargs)


def test_eq23_exact_separate_step_sizes_and_no_implicit_clipping():
    model = ScalarRoles()
    trainer = SpecRSITrainer(model, config(actor_learning_rate=.2, value_learning_rate=.3, value_weight=1e6))
    result = trainer.gvo_epoch(groups(model))
    assert model.policy_head.weight.item() == pytest.approx(.2)
    assert model.value_head.weight.item() == pytest.approx(300000.)
    assert result['update_schedule']['gradient_clipping'] == 'disabled'
    assert result['update_schedule']['actual_optimizer_steps'] == 1
    assert all(not state for state in trainer.optimizer.state.values())


def test_critic_gradient_cannot_rescale_actor_with_registered_separate_clipping():
    values = []
    for critic_weight in (1., 1e8):
        model = ScalarRoles()
        trainer = SpecRSITrainer(model, config(value_weight=critic_weight,
            actor_max_grad_norm=.25, value_max_grad_norm=.5))
        trainer.gvo_epoch(groups(model))
        values.append(model.policy_head.weight.item())
        assert model.value_head.weight.item() == pytest.approx(.05, abs=1e-6)
    assert values == pytest.approx([.025, .025], abs=1e-6)


def test_nonzero_then_zero_advantages_have_no_momentum_or_decay_drift():
    model = ScalarRoles()
    trainer = SpecRSITrainer(model, config(value_weight=0.))
    trainer.gvo_epoch(groups(model))
    previous = model.policy_head.weight.detach().clone()
    result = trainer.gvo_epoch(groups(model, returns=(0., 0.)))
    assert torch.equal(previous, model.policy_head.weight)
    assert result['learning_signal']['roles']['policy']['nonzero_gradient_steps'] == 0
    assert result['learning_signal']['roles']['policy']['parameter_change_steps'] == 0


def test_sft_reference_is_distinct_from_behavior_and_has_analytic_kl_gradient():
    model = ScalarRoles()
    trainer = SpecRSITrainer(model, config(value_weight=0., kl_weight=.7, kl_reference='frozen_sft_actor'))
    identity = trainer.bind_actor_reference(provenance={'kind': 'sft_fixture'})
    with torch.no_grad():
        model.policy_head.weight.fill_(.5)
    probability = torch.sigmoid(torch.tensor(1.)).item()
    expected = .5 - .1 * .7 * 4 * .5 * probability * (1 - probability)
    result = trainer.gvo_epoch(groups(model, returns=(0., 0.)))
    assert result['kl'] > .1
    assert model.policy_head.weight.item() == pytest.approx(expected)
    assert result['learning_signal']['roles']['policy']['nonzero_gradient_steps'] == 1
    assert result['update_schedule']['kl_reference_identity'] == identity
    assert trainer.actor_reference.weights['policy_head.weight'].item() == 0.
    assert model.value_head.weight.item() == 0.
    assert 'value_head.weight' not in trainer.actor_reference.weights


def test_reference_replays_exact_temperature_exploration_support_without_live_mutation():
    model = ScalarRoles()
    with torch.no_grad():
        model.policy_head.weight.fill_(.4)
    reference = FrozenActorReference(model)
    with torch.no_grad():
        model.policy_head.weight.fill_(1.2)
    d = decision(model, temperature=.6, exploration=.3)
    versions = [(id(p), p._version, p.requires_grad) for p in model.parameters()]
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    actual = reference.probabilities(model, (d,))[0]
    assert torch.allclose(actual, mixture_probabilities(torch.tensor([.4, -.4]), .6, .3))
    assert versions == [(id(p), p._version, p.requires_grad) for p in model.parameters()]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert all(p.grad is None for p in model.parameters())


def test_reference_and_update_rule_survive_checkpoint_and_detect_tampering(tmp_path):
    model = ScalarRoles()
    training = config(kl_weight=.1, kl_reference='frozen_sft_actor')
    trainer = SpecRSITrainer(model, training)
    identity = trainer.bind_actor_reference(provenance={'kind': 'sft_fixture'})
    trainer.gvo_epoch(groups(model))
    trainer.save(tmp_path)
    restored_model = ScalarRoles()
    restored_model.load_state_dict(torch.load(tmp_path/'weights.pt', weights_only=True))
    restored = SpecRSITrainer(restored_model, replace(training))
    restored.restore_optimizer(tmp_path)
    assert restored.actor_reference.identity == identity
    assert isinstance(restored.optimizer, torch.optim.SGD)
    assert restored.actor_reference.manifest() == trainer.actor_reference.manifest()
    recorded = json.loads((tmp_path/'training_evidence.json').read_text())
    assert recorded['actor_reference']['identity'] == identity
    weights = torch.load(tmp_path/'actor_reference.pt', weights_only=True)
    weights['policy_head.weight'].add_(1)
    torch.save(weights, tmp_path/'actor_reference.pt')
    with pytest.raises(ValueError, match='identity'):
        FrozenActorReference.load(tmp_path, restored_model)


def test_old_adam_checkpoint_stays_legacy_until_explicit_rule_override(tmp_path):
    model = ScalarRoles()
    legacy = SpecRSITrainer(model, TrainingConfig(learning_rate=.1, weight_decay=0., value_weight=0.))
    legacy.gvo_epoch(groups(model))
    legacy.save(tmp_path)
    saved = json.loads((tmp_path/'training_config.json').read_text())
    saved.pop('optimizer_rule'); saved.pop('kl_reference')
    (tmp_path/'training_config.json').write_text(json.dumps(saved))
    state = torch.load(tmp_path/'trainer.pt', weights_only=True);state.pop('optimizer_rule')
    torch.save(state,tmp_path/'trainer.pt')
    inherited = SpecRSITrainer(model, config())
    inherited.restore_optimizer(tmp_path)
    assert inherited.config.optimizer_rule == 'adamw_legacy_v1'
    assert isinstance(inherited.optimizer, torch.optim.AdamW)
    migrated = SpecRSITrainer(model, config(value_weight=0.))
    migrated.restore_optimizer(tmp_path, optimizer_rule_override='pdf_gradient_v1')
    assert migrated.optimizer_restore_history[-1]['state_transfer'] == 'discarded_on_registered_rule_change'
    assert not migrated.optimizer.state
    before = model.policy_head.weight.detach().clone()
    migrated.gvo_epoch(groups(model, returns=(0., 0.)))
    assert torch.equal(before, model.policy_head.weight)


def test_invalid_reference_and_update_settings_fail_before_training():
    with pytest.raises(ValueError, match='weight_decay'):
        TrainingConfig(optimizer_rule='pdf_gradient_v1')
    with pytest.raises(ValueError, match='clipping'):
        config(actor_max_grad_norm=-1.)
    with pytest.raises(ValueError, match='frozen-collection'):
        config(gvo_update_schedule='legacy_group_then_forced_v1')
    model = ScalarRoles()
    trainer = SpecRSITrainer(model, config(kl_reference='frozen_sft_actor', kl_weight=1.))
    with pytest.raises(ValueError, match='identified reference'):
        trainer.gvo_epoch(groups(model))
    assert trainer.steps == 0


def test_disabled_clipping_still_rejects_nonfinite_gradients():
    model = ScalarRoles()
    trainer = SpecRSITrainer(model, config())
    model.policy_head.weight.grad = torch.full_like(model.policy_head.weight, float('inf'))
    with pytest.raises(ValueError, match='Nonfinite gradient'):
        trainer._step()
    assert model.policy_head.weight.item() == 0. and trainer.steps == 0


def test_stateless_reference_on_injected_tiny_hf_preserves_lora_and_live_backward(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from specrsi.learning.hf import HFModelConfig, HFRoleModel
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
    model = HFRoleModel(HFModelConfig(model_path='__fixture__', device='cpu', dtype='float32',
        lora_rank=2, lora_alpha=4, option_dim=12, head_hidden_dim=8, gradient_checkpointing=True), base_model=base)
    model.eval()
    reference = FrozenActorReference(model)
    decision = DecisionRecord('edit_token', ('3','4'), 0, math.log(.5), input_ids=(1,2), exploration=.2)
    old_adapter = model.backbone.active_adapter
    versions = [(id(p), p._version, p.requires_grad) for p in model.parameters()]
    first = reference.probabilities(model, (decision,))[0]
    assert model.backbone.active_adapter == old_adapter
    assert versions == [(id(p), p._version, p.requires_grad) for p in model.parameters()]
    assert all('lora_' in key or key.startswith(('policy_head.', 'edit_head.')) for key in reference.weights)
    current = mixture_probabilities(model.decision_logits(decision), decision.temperature, decision.exploration)
    assert torch.allclose(first, current, atol=1e-6)
    (-current[0].log()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for name,p in model.named_parameters() if '.draft.' in name)
    assert all(p.grad is None for p in reference.weights.values())
    reference.save(tmp_path)
    restored = FrozenActorReference.load(tmp_path, model)
    assert restored.identity == reference.identity


def test_sft_reference_and_live_heads_share_only_frozen_global_cache(tmp_path):
    from specrsi.core.vats import VATSPlan
    from test_learning_observed_context_cache import roles
    from test_learning_semantic_graph import state_and_plans
    from specrsi.learning.paper_bridge import VATSSearchBridge
    model = roles(tmp_path)
    state, plans = state_and_plans()
    plans = (*plans, VATSPlan("accept"))
    bridge = VATSSearchBridge(model)
    record = bridge._record(state, plans)
    reference = FrozenActorReference(model)
    before = reference.probabilities(model, (record,))[0]
    frozen_identity = reference.contract['frozen_text_snapshot']
    with torch.no_grad():
        # ACCEPT stays fixed zero, so changing only non-ACCEPT preference bias
        # changes the distribution, not a removable shared softmax offset.
        model.policy_head.output[-1].bias.add_(2.)
    versions = [(id(p), p._version) for p in model.parameters()]
    live = mixture_probabilities(model.decision_logits(record), record.temperature, record.exploration)
    replay = reference.probabilities(model, (record,))[0]
    again = mixture_probabilities(model.decision_logits(record), record.temperature, record.exploration)
    assert not torch.allclose(live, before)
    assert torch.allclose(before, replay, atol=1e-6)
    assert torch.allclose(live, again, atol=1e-6)
    assert versions == [(id(p), p._version) for p in model.parameters()]
    assert reference.contract['frozen_text_snapshot'] == frozen_identity
    assert model._last_global_encoding['encoder_cache_hit'] is True
