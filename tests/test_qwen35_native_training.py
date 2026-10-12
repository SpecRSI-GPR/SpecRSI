"""Exercise the reference backbone's native hybrid MoE classes on CPU.

The randomly initialized, deliberately small fixture tests software contracts;
it is neither an experimental backbone nor a substitute for the pinned 35B
publisher snapshot. Both Gated DeltaNet and full-attention layers are present.
No dataset, pretrained weight download, GPU, or experiment result is required.
"""
import json
import inspect
import random

import pytest
import torch

from specrsi.learning.hf import HFModelConfig, HFRoleModel
from specrsi.learning.objectives import mixture_probabilities
from specrsi.learning.records import EditExample
from specrsi.learning.sampler import sample_edit_tokens


@pytest.fixture(autouse=True)
def cpu_reference_kernels(monkeypatch):
    """Use the native Torch paths even on a host with CUDA extensions installed.

    Transformers selects installed causal-convolution/FLA packages at import
    time. Their CUDA kernels cannot execute CPU fixtures; unwrap only these
    decorated kernel functions to exercise the library's own CPU reference
    implementations. Production model loading and training are unchanged.
    """
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5MoeForCausalLM"):
        pytest.skip("Install the pinned optional model dependencies for native Qwen3.5 coverage")
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as native
    for name in ("causal_conv1d_fn", "causal_conv1d_update",
                 "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        monkeypatch.setattr(native, name, inspect.unwrap(getattr(native, name)))


def native_fixture(directory, *, checkpointing=False):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if not hasattr(transformers, "Qwen3_5MoeForCausalLM"):
        pytest.skip("Install the pinned optional model dependencies for native Qwen3.5 coverage")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(345)
        text = Qwen3_5MoeTextConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            moe_intermediate_size=32, shared_expert_intermediate_size=32,
            num_experts=4, num_experts_per_tok=2, eos_token_id=31,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]})
        base = Qwen3_5MoeForCausalLM(text)
        # Persist the unmodified tiny native weights before PEFT registration.
        # This lets checkpoint restoration use the production local HF loader.
        base.save_pretrained(directory, safe_serialization=True)
        vocabulary = {"[UNK]": 0, **{str(index): index for index in range(1, 31)}, "[EOS]": 31}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            unk_token="[UNK]", eos_token="[EOS]", model_max_length=32)
        tokenizer.save_pretrained(directory)
        config = HFModelConfig(model_path=str(directory), device="cpu", dtype="float32",
            architecture_version=3, option_dim=12, head_hidden_dim=16,
            lora_rank=2, lora_alpha=4, gradient_checkpointing=checkpointing,
            max_length=32, language_model_only=True, require_unquantized_backbone=True,
            lora_target_profile="qwen35_hybrid_attention", experts_implementation="eager")
        model = HFRoleModel(config, base_model=base, tokenizer=tokenizer).eval()
    return model


@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_hybrid_lora_covers_both_layer_types_and_freezes_experts(tmp_path, checkpointing):
    model = native_fixture(tmp_path / "native", checkpointing=checkpointing)
    assert set(model.resolved_lora_targets) == {
        "model.layers.0.linear_attn." + name for name in
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    } | {"model.layers.1.self_attn." + name for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    assert all("mlp" not in target and "lm_head" not in target for target in model.resolved_lora_targets)
    loss = model.edit_nll(EditExample("native", (1, 2, 3, 4), (-100, -100, 3, 4), True))
    assert torch.isfinite(loss)
    loss.backward()
    named = list(model.backbone.named_parameters())
    for layer in ("linear_attn", "self_attn"):
        assert any(layer in name and ".draft." in name and parameter.grad is not None
            and torch.count_nonzero(parameter.grad) for name, parameter in named)
    assert all(parameter.grad is None for name, parameter in named if "lora_" not in name or ".verify." in name)
    assert all(not parameter.requires_grad for name, parameter in named if "mlp" in name or ".verify." in name)
    assert all(parameter.grad is None for parameter in model.policy_head.parameters())
    assert all(parameter.grad is None for parameter in model.value_head.parameters())
    assert all(not module.training for module in model.modules())


def test_native_cached_sampler_replays_its_exact_behavior_support(tmp_path):
    model = native_fixture(tmp_path / "native")
    result = sample_edit_tokens(model, (1, 2, 3), max_new_tokens=3,
        top_k=5, rng=random.Random(91), proposal_snapshot="native-fixture",
        temperature=.8, exploration=.1, eos_token_ids=())
    assert result.token_cost == 6 and len(result.decisions) == 3
    for decision in result.decisions:
        replay = mixture_probabilities(model.decision_logits(decision), decision.temperature, decision.exploration)
        assert replay.detach().tolist() == pytest.approx(decision.behavior_probabilities, abs=1e-6)
        assert float(replay[decision.selected_index].log().detach()) == pytest.approx(decision.behavior_logprob, abs=1e-6)


def test_native_checkpoint_restores_registered_hybrid_adapters_and_frozen_identity(tmp_path):
    model = native_fixture(tmp_path / "native")
    with torch.no_grad():
        for name, parameter in model.backbone.named_parameters():
            if "lora_B" in name and ".draft." in name:
                parameter.fill_(.02)
    example = EditExample("native", (1, 2, 3, 4), (-100, -100, 3, 4), True)
    before = model.edit_nll(example).detach()
    checkpoint = tmp_path / "checkpoint"
    model.save(checkpoint)
    manifest = json.loads((checkpoint / "model_config.json").read_text())
    assert manifest["lora_target_profile"] == "qwen35_hybrid_attention"
    assert manifest["experts_implementation"] == "eager"
    assert manifest["resolved_lora_targets"] == list(model.resolved_lora_targets)
    restored = HFRoleModel.load(checkpoint, device="cpu").eval()
    assert restored.resolved_lora_targets == model.resolved_lora_targets
    assert torch.allclose(restored.edit_nll(example).detach(), before, atol=1e-6)
    assert all(not parameter.requires_grad for name, parameter in restored.backbone.named_parameters()
               if "lora_" not in name or ".verify." in name)
