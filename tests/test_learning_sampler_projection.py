"""Draft prefill projects only consumed logits without changing its policy."""
import random

import pytest
import torch

from specrsi.learning.objectives import mixture_probabilities
from specrsi.learning.sampler import sample_edit_tokens


@pytest.mark.parametrize("top_k", [0, 7])
def test_last_position_projection_preserves_actual_sampling_and_replay(top_k):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from specrsi.learning.hf import HFModelConfig, HFRoleModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        backbone = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=8, eos_token_id=63))
        model = HFRoleModel(HFModelConfig(model_path="__sampler_projection__", device="cpu",
            dtype="float32", lora_rank=2, lora_alpha=4, gradient_checkpointing=False,
            max_length=32), base_model=backbone)
    assert model._supports_logit_positions
    prompt = tuple(range(1, 14))
    projected_lengths = []
    hook = backbone.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: projected_lengths.append(args[0].shape[-2]))
    settings = dict(max_new_tokens=4, top_k=top_k, temperature=.8, exploration=.1,
                    proposal_snapshot="fixed-projection-comparison", eos_token_ids=())
    try:
        model._supports_logit_positions = False
        reference = sample_edit_tokens(model, prompt, rng=random.Random(31), **settings)
        reference_lengths = tuple(projected_lengths)
        projected_lengths.clear()
        model._supports_logit_positions = True
        actual = sample_edit_tokens(model, prompt, rng=random.Random(31), **settings)
        actual_lengths = tuple(projected_lengths)
    finally:
        hook.remove()
    assert reference_lengths == (len(prompt), 1, 1, 1)
    assert actual_lengths == (1, 1, 1, 1)
    assert actual.generated_ids == reference.generated_ids
    assert actual.prompt_ids == reference.prompt_ids == prompt
    assert actual.token_cost == reference.token_cost == len(prompt) + 4
    assert actual.stop_reason == reference.stop_reason == "output_cap"
    for decision, previous in zip(actual.decisions, reference.decisions):
        assert decision.support == previous.support
        assert decision.selected_index == previous.selected_index
        assert decision.input_ids == previous.input_ids
        assert decision.behavior_probabilities == pytest.approx(previous.behavior_probabilities, abs=1e-7)
        assert decision.behavior_logprob == pytest.approx(previous.behavior_logprob, abs=1e-6)
        replay = mixture_probabilities(model.decision_logits(decision), decision.temperature, decision.exploration)
        assert replay.detach().tolist() == pytest.approx(decision.behavior_probabilities, abs=1e-7)


def test_backbone_without_projection_capability_gets_no_new_keyword():
    from types import SimpleNamespace

    class StrictLegacyBackbone:
        config = SimpleNamespace(eos_token_id=2)

        def __call__(self, input_ids, attention_mask, past_key_values, use_cache, return_dict):
            logits = torch.tensor([[[0., 1., 100.]]])
            return SimpleNamespace(logits=logits, past_key_values=None)

    class LegacyRole:
        backend = "hf"
        device = "cpu"
        config = SimpleNamespace(max_length=16)
        tokenizer = None
        backbone = StrictLegacyBackbone()

        def eval(self):
            return self

        def _activate(self, role):
            assert role == "draft"

    result = sample_edit_tokens(LegacyRole(), (0, 1), max_new_tokens=3,
        rng=random.Random(0), top_k=0, exploration=0., proposal_snapshot="legacy-projection")
    assert result.generated_ids == (2,)
    assert result.stop_reason == "eos" and result.token_cost == 3
