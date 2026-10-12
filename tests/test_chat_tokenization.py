"""Token containers must never change exact prompt contents or their charge."""
from types import SimpleNamespace

import pytest
import torch

from specrsi.tokenization import chat_token_ids


@pytest.mark.parametrize("encoding", [
    [10, 20, 30], (10, 20, 30), [[10, 20, 30]],
    {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]},
    {"input_ids": [[10, 20, 30]], "attention_mask": [[1, 1, 1]]},
    torch.tensor([10, 20, 30]), {"input_ids": torch.tensor([[10, 20, 30]])},
])
def test_flat_ids_from_legacy_lists_dictionary_wrappers_and_one_batch(encoding):
    calls = []

    def apply(messages, **kwargs):
        calls.append((messages, kwargs))
        return encoding

    messages = [{"role": "user", "content": "Keep this complete."}]
    assert chat_token_ids(SimpleNamespace(apply_chat_template=apply), messages,
                          add_generation_prompt=True, enable_thinking=False) == [10, 20, 30]
    assert calls == [(messages, {"add_generation_prompt": True, "enable_thinking": False,
                                "tokenize": True, "return_dict": False})]


def test_closed_signature_legacy_tokenizer_does_not_require_new_keyword():
    def legacy(messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        return [4, 5, 6]

    assert chat_token_ids(SimpleNamespace(apply_chat_template=legacy), [],
                          add_generation_prompt=True, enable_thinking=False) == [4, 5, 6]


@pytest.mark.parametrize("encoding,error", [
    ({"attention_mask": [1]}, ValueError), ([[1], [2]], ValueError),
    ("input_ids", TypeError), (["input_ids"], ValueError), ([True], ValueError),
    ([-1], ValueError), ([1.5], ValueError), ([[[1]]], ValueError),
])
def test_invalid_or_multiple_chat_encodings_fail_before_generation(encoding, error):
    with pytest.raises(error):
        chat_token_ids(SimpleNamespace(apply_chat_template=lambda *a, **kw: encoding), [])


def test_internal_type_error_is_not_retried_as_legacy_signature():
    calls = []

    def apply(messages, **kwargs):
        calls.append(kwargs)
        raise TypeError("actual template problem")

    with pytest.raises(TypeError, match="actual template problem"):
        chat_token_ids(SimpleNamespace(apply_chat_template=apply), [])
    assert len(calls) == 1


def test_executor_count_is_input_length_not_dictionary_fields():
    from specrsi.experiments import QwenExecutor

    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw:
        {"input_ids": [[1, 2, 3, 4, 5]], "attention_mask": [[1] * 5]})
    assert QwenExecutor(None, tokenizer, {}).prompt_tokens([]) == 5
