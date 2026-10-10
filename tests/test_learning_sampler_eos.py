"""EOS contracts must stop real draft collection without losing the final actor."""
import math
import random
from types import SimpleNamespace

import pytest
import torch

from specrsi.learning.sampler import _eos_token_ids, sample_edit_tokens


class ScriptedBackbone:
    def __init__(self, config, generation_config=None, tokens=(2, 1, 1)):
        self.config = config
        self.generation_config = generation_config
        self.tokens = tokens
        self.calls = 0

    def __call__(self, **kwargs):
        # A nonzero alternative probability makes the terminating token a real
        # sampled actor choice, rather than deterministic grammar punctuation.
        logits = torch.zeros((1, 1, 4))
        logits[0, 0, self.tokens[self.calls]] = 4.
        self.calls += 1
        return SimpleNamespace(logits=logits, past_key_values=None)


class ScriptedRole:
    backend = "hf"
    device = "cpu"
    config = SimpleNamespace(max_length=16)
    tokenizer = None

    def __init__(self, backbone):
        self.backbone = backbone

    def eval(self):
        return self

    def _activate(self, role):
        assert role == "draft"


def sample(backbone, **kwargs):
    return sample_edit_tokens(ScriptedRole(backbone), (0, 1), max_new_tokens=3,
        rng=random.Random(0), top_k=0, exploration=0., proposal_snapshot="eos-contract", **kwargs)


@pytest.mark.parametrize("text_config", [SimpleNamespace(eos_token_id=2), {"eos_token_id": 2}])
def test_nested_conditional_text_eos_stops_before_continuing_past_end(text_config):
    backbone = ScriptedBackbone(SimpleNamespace(text_config=text_config))
    result = sample(backbone)
    assert result.generated_ids == (2,)
    assert result.stop_reason == "eos"
    assert backbone.calls == 1
    assert result.token_cost == len(result.prompt_ids) + 1
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert not decision.deterministic
    assert int(decision.support[decision.selected_index]) == 2
    assert decision.input_ids == result.prompt_ids
    expected = torch.softmax(torch.tensor([0., 0., 4., 0.]), dim=0)
    assert decision.behavior_probabilities == pytest.approx(expected.tolist())
    assert decision.behavior_logprob == pytest.approx(math.log(float(expected[2])))


@pytest.mark.parametrize("terminal", [2, 3])
def test_runtime_generation_config_keeps_every_registered_eos_and_has_priority(terminal):
    backbone = ScriptedBackbone(SimpleNamespace(eos_token_id=1,
        text_config=SimpleNamespace(eos_token_id=1)),
        SimpleNamespace(eos_token_id=[2, 3]), tokens=(terminal, 1, 1))
    assert _eos_token_ids(backbone) == {2, 3}
    result = sample(backbone)
    assert result.generated_ids == (terminal,) and result.stop_reason == "eos"
    assert backbone.calls == 1


def test_explicit_eos_overrides_generation_and_empty_override_disables_stopping():
    backbone = ScriptedBackbone(SimpleNamespace(eos_token_id=2), SimpleNamespace(eos_token_id=[2, 3]))
    assert _eos_token_ids(backbone, (1,)) == {1}
    result = sample(backbone, eos_token_ids=(1,))
    assert result.generated_ids == (2, 1) and result.stop_reason == "eos"
    backbone = ScriptedBackbone(SimpleNamespace(eos_token_id=2), SimpleNamespace(eos_token_id=[2, 3]))
    result = sample(backbone, eos_token_ids=())
    assert result.generated_ids == (2, 1, 1) and result.stop_reason == "output_cap"
    assert result.token_cost == 5 and len(result.decisions) == 3


@pytest.mark.parametrize("generation_config", [None, SimpleNamespace(eos_token_id=None)])
def test_legacy_text_model_top_level_eos_fallback_is_unchanged(generation_config):
    backbone = ScriptedBackbone(SimpleNamespace(eos_token_id=2), generation_config)
    result = sample(backbone)
    assert result.generated_ids == (2,) and result.stop_reason == "eos"
    assert result.token_cost == 3


def test_publisher_qwen35_multiple_eos_and_qwen3_terminal_are_preserved():
    # IDs are the public, local pinned checkpoints' generation config values.
    qwen35 = SimpleNamespace(config={"text_config": {"eos_token_id": 248044}},
        generation_config={"eos_token_id": [248046, 248044]})
    qwen3 = SimpleNamespace(config={"eos_token_id": 151645},
        generation_config={"eos_token_id": [151645, 151643]})
    assert _eos_token_ids(qwen35) == {248046, 248044}
    assert _eos_token_ids(qwen3) == {151645, 151643}
    assert _eos_token_ids(SimpleNamespace(config={})) == set()


@pytest.mark.parametrize("invalid", [True, -1, "2", [1, False], [1, -2], [1, "2"]])
def test_malformed_termination_contract_is_rejected_before_any_forward(invalid):
    backbone = ScriptedBackbone(SimpleNamespace(eos_token_id=2), SimpleNamespace(eos_token_id=invalid))
    with pytest.raises(ValueError, match="EOS token IDs"):
        sample(backbone)
    assert backbone.calls == 0
