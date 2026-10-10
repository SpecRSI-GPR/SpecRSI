"""Byte-token cuts must retain original Unicode rather than decoder artifacts."""
import pytest

from specrsi.learning.frozen_text import bounded_semantic_text


class ByteTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.mark.parametrize("strategy", ["prefix", "head_tail"])
@pytest.mark.parametrize("cap", [17, 31, 156])
def test_truncation_never_invents_replacement_characters(strategy, cap):
    tokenizer = ByteTokenizer()
    text = "👩🏽‍🔬" * 500
    clipped, ids = bounded_semantic_text(tokenizer, text, cap, strategy=strategy)
    assert "\ufffd" not in clipped
    assert ids == tuple(tokenizer.encode(clipped)) and 0 < len(ids) <= cap
    if strategy == "prefix":
        assert text.startswith(clipped)
    else:
        head, tail = clipped.split(" … ")
        assert text.startswith(head) and text.endswith(tail)


def test_valid_excerpts_and_exact_fit_remain_unchanged():
    tokenizer = ByteTokenizer()
    assert bounded_semantic_text(tokenizer, "你好 👩🏽‍🔬", 128) == (
        "你好 👩🏽‍🔬", tuple(tokenizer.encode("你好 👩🏽‍🔬")))
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    clipped, ids = bounded_semantic_text(tokenizer, text, 16)
    assert clipped == "ABCDEF … VWXYZ" and ids == tuple(tokenizer.encode(clipped))
