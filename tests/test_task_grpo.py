import torch
import pytest
from specrsi.learning.hf import HFRoleModel,HFModelConfig
from specrsi.learning.task_grpo import HFTaskClient,sequence_logprobs,update_group


def tiny_model():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [1,2,3]
        def decode(self,ids,**kwargs):return ','.join(map(str,ids))
    c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=8,eos_token_id=31)
    return HFRoleModel(HFModelConfig(model_path='__tiny_task_grpo__',device='cpu',dtype='float32',
        lora_rank=2,lora_alpha=4,gradient_checkpointing=False,max_length=16),base_model=Qwen3ForCausalLM(c),tokenizer=Tokenizer())


@pytest.mark.parametrize('mapping_output', [False, True])
def test_task_grpo_full_vocabulary_replay_and_actor_only_updates(mapping_output):
    torch.set_num_threads(2);torch.manual_seed(7);model=tiny_model();client=HFTaskClient(model,'test-task-policy')
    if mapping_output:
        model.tokenizer.apply_chat_template=lambda *a,**kw: {'input_ids':[[1,2,3]],'attention_mask':[[1,1,1]]}
    for seed in (1,2):
        reply=client.generate([{'role':'user','content':'test'}],max_tokens=4,temperature=1.,seed=seed,total_token_cap=7)
        seq=client.sequences[-1]
        assert seq.prompt_ids==(1,2,3)
        assert reply.prompt_tokens+reply.completion_tokens==len(seq.prompt_ids)+len(seq.generated_ids)
        assert torch.allclose(sequence_logprobs(model,seq).detach(),torch.tensor(seq.behavior_logprobs),atol=2e-5)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    parameters=[p for n,p in model.backbone.named_parameters() if 'lora_' in n and '.draft.' in n]
    optimizer=torch.optim.AdamW(parameters,lr=.01,weight_decay=0)
    report=update_group(model,optimizer,[[s] for s in client.sequences],[0.,1.],kl_weight=.01)
    assert report['updated'] and report['grad_norm']>0
    changed=[n for n,p in model.named_parameters() if not torch.equal(p.detach(),before[n])]
    assert changed and all('lora_' in n and '.draft.' in n for n in changed)
    # Equal task rewards do not remove the reference KL objective after policy drift.
    report=update_group(model,optimizer,[[s] for s in client.sequences],[1.,1.],kl_weight=.1)
    assert report['equal_native_rewards'] and report['updated'] and report['kl']>0


def test_equal_task_rewards_skip_optimizer_instead_of_fabricating_signal():
    model=tiny_model();client=HFTaskClient(model,'test-task-policy')
    for seed in (3,4):client.generate([],max_tokens=2,seed=seed,total_token_cap=5)
    parameters=[p for n,p in model.backbone.named_parameters() if 'lora_' in n and '.draft.' in n]
    optimizer=torch.optim.AdamW(parameters,lr=.01)
    before=[p.detach().clone() for p in parameters]
    result=update_group(model,optimizer,[[s] for s in client.sequences],[1.,1.],kl_weight=0)
    assert not result['updated'] and all(torch.equal(a,b) for a,b in zip(parameters,before))
