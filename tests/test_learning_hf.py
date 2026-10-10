"""A tiny real transformer verifies PEFT integration without model downloads."""
import importlib.util

import pytest
import torch

torch.set_num_threads(2)


@pytest.mark.skipif(not importlib.util.find_spec("peft"), reason="Install optional training dependencies")
def test_qwen3_separate_adapters_frozen_backbone_and_token_backward():
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        import peft
    except ImportError as exc:
        pytest.skip(f"Optional HF stack unavailable: {exc}")
    from specrsi.learning.hf import HFModelConfig, HFRoleModel
    from specrsi.learning.records import EditExample
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    base = Qwen3ForCausalLM(config)
    roles = HFRoleModel(HFModelConfig(model_path="__injected_tiny_test__", device="cpu", dtype="float32",
        lora_rank=2, lora_alpha=4, option_dim=2, head_hidden_dim=8, gradient_checkpointing=False), base_model=base)
    roles.eval()
    roles.edit_nll(EditExample("root", (1,2,3,4), (-100,-100,3,4), True)).backward()
    named = list(roles.backbone.named_parameters())
    assert any("lora_" in n and ".draft." in n and p.grad is not None for n,p in named)
    assert not any(".verify." in n and p.grad is not None for n,p in named)
    assert not any("lora_" not in n and p.grad is not None for n,p in named)
    roles.zero_grad(set_to_none=True)
    hidden = roles.encode("", "verify", input_ids=(1,2,3))
    roles.surface_head(hidden).backward()
    assert any(".verify." in n and p.grad is not None for n,p in named)
    assert not any(".draft." in n and p.grad is not None for n,p in named)
    roles.zero_grad(set_to_none=True)
    from specrsi.learning.records import DecisionRecord
    plan = DecisionRecord("plan", ("draft", "ACCEPT"), 0, 0., input_ids=(1, 2, 3),
        state_features=(0.,) * 8, option_features=((1., 0.), (0., 1.)))
    assert roles.decision_values(plan)[1].item() == 0.
    logits = roles.decision_logits(plan)
    assert logits[1].item() == 0.
    logits[0].backward()
    assert any(p.grad is not None for p in roles.policy_head.parameters())
    assert all(p.grad is None for p in roles.value_head.parameters())
    assert all(p.grad is None for p in roles.backbone.parameters())


def test_real_transformer_token_collection_support_replay_and_actor_gradients():
    try:
        from transformers import Qwen3Config,Qwen3ForCausalLM
        import peft
    except ImportError as exc:
        pytest.skip(f"Optional HF stack unavailable: {exc}")
    import random
    from specrsi.learning.hf import HFModelConfig,HFRoleModel
    from specrsi.learning.sampler import sample_edit_tokens
    from specrsi.learning.objectives import mixture_probabilities,clipped_surrogate
    cfg = Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=8,eos_token_id=31)
    roles = HFRoleModel(HFModelConfig(model_path="__tiny_sampler_test__",device="cpu",dtype="float32",
        lora_rank=2,lora_alpha=4,gradient_checkpointing=False,max_length=16),base_model=Qwen3ForCausalLM(cfg))
    sample = sample_edit_tokens(roles,(1,2,3),max_new_tokens=4,rng=random.Random(0),top_k=5,
        proposal_snapshot="frozen-test",eos_token_ids=())
    assert len(sample.generated_ids)==4 and sample.token_cost==7
    assert all(len(d.support)==5 and len(d.input_ids)==3+i for i,d in enumerate(sample.decisions))
    for decision in sample.decisions:
        values = roles.decision_values(decision)
        probs = mixture_probabilities(values,decision.temperature,decision.exploration)
        new_logprob = probs[decision.selected_index].log()
        assert abs(float(new_logprob.detach())-decision.behavior_logprob)<2e-5
        loss = -clipped_surrogate(new_logprob,new_logprob.new_tensor(decision.behavior_logprob),new_logprob.new_tensor(1.))
        loss.backward()
    assert any(".draft." in n and p.grad is not None for n,p in roles.backbone.named_parameters())
    assert not any("lora_" not in n and p.grad is not None for n,p in roles.backbone.named_parameters())
    with pytest.raises(ValueError,match="context length"):
        sample_edit_tokens(roles,(1,)*15,max_new_tokens=4,rng=random.Random(0),proposal_snapshot="test")


def test_constrained_numeric_token_support_marks_forced_syntax():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from specrsi.learning.hf import HFModelConfig,HFRoleModel
    from specrsi.learning.sampler import sample_edit_tokens
    import random
    cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=8,eos_token_id=31)
    roles=HFRoleModel(HFModelConfig(model_path='__tiny_grammar__',device='cpu',dtype='float32',
        lora_rank=2,lora_alpha=4,gradient_checkpointing=False,max_length=16),base_model=Qwen3ForCausalLM(cfg))
    sequences=((5,7,31),(5,8,31))
    grammar=lambda prefix:sorted({seq[len(prefix)] for seq in sequences if seq[:len(prefix)]==prefix and len(seq)>len(prefix)})
    result=sample_edit_tokens(roles,(1,2),max_new_tokens=3,rng=random.Random(0),token_grammar=grammar,
        top_k=32,proposal_snapshot='frozen-numeric-catalog')
    assert result.generated_ids in sequences and result.stop_reason=='eos'
    assert [d.deterministic for d in result.decisions]==[True,False,True]
    assert result.decisions[0].behavior_logprob is None
    assert result.decisions[1].support==('7','8')
    assert result.decisions[1].input_ids==(1,2,5)
