"""Backbone architecture compatibility without pretrained weights or GPU."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from specrsi.learning import hf
torch.set_num_threads(2)


def test_known_conditional_factory_and_old_causal_factory():
    classes=SimpleNamespace(AutoModelForCausalLM=object(), Qwen3_5ForConditionalGeneration=object(),
                            Qwen3_5MoeForConditionalGeneration=object())
    assert hf._backbone_factory('qwen3', classes) is classes.AutoModelForCausalLM
    assert hf._backbone_factory('qwen3_5_moe_text', classes) is classes.AutoModelForCausalLM
    assert hf._backbone_factory('qwen3_5', classes) is classes.Qwen3_5ForConditionalGeneration
    assert hf._backbone_factory('qwen3_5_moe', classes) is classes.Qwen3_5MoeForConditionalGeneration
    with pytest.raises(RuntimeError, match='validated Transformers/PEFT'):
        hf._backbone_factory('qwen3_5_moe', SimpleNamespace(AutoModelForCausalLM=object()))


def test_text_hidden_dimension_has_priority_over_vision_or_top_level():
    assert hf._text_hidden_size(SimpleNamespace(hidden_size=16)) == 16
    assert hf._text_hidden_size(SimpleNamespace(hidden_size=1152, text_config=SimpleNamespace(hidden_size=2048))) == 2048
    assert hf._text_hidden_size(SimpleNamespace(text_config={'hidden_size': 2048})) == 2048
    for config in [SimpleNamespace(), SimpleNamespace(hidden_size=True), SimpleNamespace(text_config={'hidden_size': 0})]:
        with pytest.raises(ValueError, match='positive text hidden_size'):
            hf._text_hidden_size(config)


def test_official_local_qwen35_checkpoint_uses_correct_text_width():
    raw={'architectures':['Qwen3_5MoeForConditionalGeneration'], 'text_config':{'hidden_size':2048}, 'vision_config':{'hidden_size':1152}}
    assert raw['architectures'] == ['Qwen3_5MoeForConditionalGeneration']
    assert hf._text_hidden_size(SimpleNamespace(text_config=raw['text_config'])) == 2048
    assert raw['vision_config']['hidden_size'] == 1152


def test_local_loader_preserves_dtype_revision_and_no_remote_code(tmp_path,monkeypatch):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe'}))
    observed=[]
    factory=SimpleNamespace(from_pretrained=lambda *args,**kwargs: observed.append((args,kwargs)) or 'model')
    monkeypatch.setattr(hf,'_backbone_factory',lambda model_type: factory if model_type=='qwen3_5_moe' else None)
    result=hf._load_local_backbone(hf.HFModelConfig(str(tmp_path),revision='pinned'),torch.float32)
    assert result=='model'
    assert observed==[((str(tmp_path),),{'local_files_only':True,'revision':'pinned','torch_dtype':torch.float32,'trust_remote_code':False})]


def role_config():
    return hf.HFModelConfig(model_path='__isolated_tiny_test__',device='cpu',dtype='float32',
        lora_rank=2,lora_alpha=4,option_dim=2,head_hidden_dim=8,gradient_checkpointing=False,max_length=16)


def text_config():
    from transformers import Qwen3Config
    return Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                      num_attention_heads=2,num_key_value_heads=2,head_dim=8,eos_token_id=31)


def test_old_qwen3_head_encoding_and_edit_logits_remain_functional():
    from transformers import Qwen3ForCausalLM
    from specrsi.learning.records import EditExample
    roles=hf.HFRoleModel(role_config(),base_model=Qwen3ForCausalLM(text_config()))
    roles.eval()
    assert roles.encode('', 'verify', input_ids=(1,2,3)).shape==(16,)
    assert roles.encode('', 'value', input_ids=(1,2,3)).requires_grad is False
    assert torch.isfinite(roles.edit_nll(EditExample('root',(1,2,3,4),(-100,-100,3,4),True)))


def test_qwen35_shaped_conditional_wrapper_preserves_peft_and_hidden_interfaces():
    from transformers import Qwen3ForCausalLM, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from specrsi.learning.records import EditExample

    class CompositeDecoder(torch.nn.Module):
        def __init__(self, text):
            super().__init__();self.language_model=text
            self.visual=torch.nn.Linear(7,9)
        def forward(self, **kwargs):
            return self.language_model(**kwargs)

    class CompositeGeneration(Qwen3ForCausalLM):
        def __init__(self):
            text=text_config();super().__init__(text)
            self.model=CompositeDecoder(self.model)
            self.config=PretrainedConfig(text_config=text,tie_word_embeddings=False)
            self.config.model_type='qwen3_5_moe'
            self.logit_calls=0
        def get_input_embeddings(self):
            return self.model.language_model.embed_tokens
        def set_input_embeddings(self, value):
            self.model.language_model.embed_tokens=value
        def forward(self,input_ids=None,attention_mask=None,inputs_embeds=None,use_cache=False,return_dict=True,**kwargs):
            self.logit_calls+=1
            hidden=self.model(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds=inputs_embeds,
                              use_cache=use_cache,return_dict=return_dict).last_hidden_state
            return CausalLMOutputWithPast(logits=self.lm_head(hidden))

    base=CompositeGeneration()
    roles=hf.HFRoleModel(role_config(),base_model=base)
    roles.eval()
    assert roles.encode('', 'verify', input_ids=(1,2,3)).shape==(16,)
    assert roles.encode('', 'policy', input_ids=(1,2,3)).requires_grad is False
    assert base.logit_calls==0  # Role feature extraction still bypasses vocabulary projection.
    assert roles.backbone.get_input_embeddings().weight.shape==(32,16)
    assert roles.backbone.get_output_embeddings().weight.shape==(32,16)
    assert torch.isfinite(roles.edit_nll(EditExample('root',(1,2,3,4),(-100,-100,3,4),True)))
    assert base.logit_calls==1
    assert any('lora_' in name and '.draft.' in name for name,_ in roles.backbone.named_parameters())
    assert any('lora_' in name and '.verify.' in name for name,_ in roles.backbone.named_parameters())
    assert all(not p.requires_grad for p in base.model.visual.parameters())
