"""Protocol boundaries that can silently alter immutable runtime or spending."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from types import SimpleNamespace

import pytest

from specrsi.core import Candidate, EvaluationScope, Ledger, Outcome, Task
from specrsi.experiments import QwenDrafter, QwenExecutor, receipt_secret, score_answer
from specrsi.models import ModelReply
from specrsi.models.server import ChatRequest, enforced_output_cap
from specrsi.workflows import Workflow, execute_workflow


def test_server_cap_uses_real_tokenized_prompt_and_rejects_overfull_context():
    assert enforced_output_cap(9, 20, 12, context_cap=32) == 3
    assert enforced_output_cap(9, 20, None, context_cap=32) == 20
    with pytest.raises(ValueError, match="no output token"):
        enforced_output_cap(12, 2, 12, context_cap=32)
    with pytest.raises(ValueError, match="context"):
        enforced_output_cap(32, 1, None, context_cap=32)
    assert ChatRequest(model="fixture-model", messages=[{"role": "user", "content": "x"}],
                       max_tokens=2, temperature=0, top_p=1, seed=1,
                       max_seconds=1, total_token_cap=5).total_token_cap == 5


def test_persistent_receipt_key_is_atomic_private_and_authenticates_reload(tmp_path):
    directory = tmp_path / ".state"
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: receipt_secret(directory), range(12)))
    assert len({key for key, _ in values}) == 1
    key, path = values[0]
    assert os.stat(path).st_mode & 0o777 == 0o600
    task = Task("one", {}, 1)
    scope = EvaluationScope("one", (task,), "one", "one")
    candidate = Candidate("one")
    ledger = Ledger(receipt_secret=key)
    ledger.admit_direct(candidate, task, scope, Outcome(1., {"answer": 1}), {"trusted": "host"})
    restored = Ledger.from_json(ledger.to_json(), receipt_secret=receipt_secret(directory)[0])
    assert restored.score(candidate, scope) == 1.
    assert key.hex() not in ledger.to_json()


def test_existing_receipt_secret_symlink_is_rejected(tmp_path):
    folder = tmp_path / ".state"
    folder.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"x" * 32)
    (folder / "ledger-receipt.key").symlink_to(target)
    with pytest.raises(OSError):
        receipt_secret(folder)


class FakeTokenizer:
    def __init__(self):
        self.thinking_calls = []

    def apply_chat_template(self, messages, *, enable_thinking, **kwargs):
        self.thinking_calls.append(enable_thinking)
        return [0] * (8 if enable_thinking else 4)


def local_fixture(reply_text="1+2+3+4", thinking=False):
    calls = []

    def generate(messages, **kwargs):
        calls.append(kwargs)
        return ModelReply(reply_text, 4, 2, .1, "stop", "pinned", "local-request")

    tokenizer = FakeTokenizer()
    config = {"max_tokens": 3, "temperature": .2, "top_p": .6, "enable_thinking": thinking}
    executor = QwenExecutor(SimpleNamespace(generate=generate), tokenizer, config)
    candidate = Candidate("fixed", {"system_prompt": "Use all given numbers"},
                          {"generation": config, "seed": 4, "model_revision": "pinned"})
    task = Task("task", {"prompt": "1 2 3 4"}, [1, 2, 3, 4], {"dataset": "game24"})
    scope = EvaluationScope("fixed", (task,), "qwen-v2", "game24-v1", ("answer", "metrics", "finish_reason"))
    return executor, candidate, task, scope, calls, tokenizer


def test_executor_reads_frozen_runtime_and_passes_server_side_total_cap():
    executor, candidate, task, scope, calls, _ = local_fixture()
    executor.config["temperature"] = 1.9
    executor.config["max_tokens"] = 999
    result = executor.run(candidate, task, scope, 8)
    assert result.status == "complete" and result.actual_cost == 6
    assert calls[0]["temperature"] == .2 and calls[0]["max_tokens"] == 3
    assert calls[0]["top_p"] == .6 and calls[0]["total_token_cap"] == 8


def test_small_budget_cannot_silently_change_scored_generation_contract():
    executor, candidate, task, scope, calls, _ = local_fixture()
    result = executor.run(candidate, task, scope, 5)
    assert result.status == "interrupted" and result.actual_cost == 0 and not calls
    workflow = Workflow.from_dict({"nodes": [{"id": "a", "instruction": "solve"}], "output_node": "a"})
    answer, used, records = execute_workflow(workflow, "task", executor, cap=5, seed=0)
    assert answer is None and used == 0 and not records and not calls


def test_tokenizer_mismatch_does_not_admit_an_outcome():
    executor, candidate, task, scope, calls, _ = local_fixture()
    executor.client.generate = lambda *a, **k: ModelReply("answer", 5, 2, .1, "stop", "pinned", "request")
    result = executor.run(candidate, task, scope, 8)
    assert result.status == "error" and result.actual_cost == 7 and result.outcome is None


def test_draft_tokenization_matches_non_thinking_request_and_validates_child_cap():
    text = json.dumps({"system_prompt": "new instruction"})
    executor, candidate, task, scope, calls, tokenizer = local_fixture(text, thinking=True)
    drafter = QwenDrafter(executor, execution_cap=20)
    state = SimpleNamespace(scope=scope, candidates={candidate.uid: candidate}, ledger=Ledger())
    import random
    result = drafter.draft(candidate, state, 10, random.Random(0))
    assert result.status == "complete"
    assert tokenizer.thinking_calls[0] is False
    assert calls[0]["enable_thinking"] is False and calls[0]["total_token_cap"] == 10
    failed = QwenDrafter(executor, execution_cap=9).draft(candidate, state, 10, random.Random(0))
    assert failed.status == "error" and failed.candidate is None
    assert "unexecutable frozen draft" in failed.error


def test_hover_label_metric_requires_explicit_prospective_variant():
    with pytest.raises(ValueError, match="retrieval"):
        score_answer("hover", "SUPPORTED", {"label": "SUPPORTED"})
    score = score_answer("hover", "SUPPORTED", {"label": "SUPPORTED"},
                         metric_variant="label_accuracy_not_GEPA_multihop_retrieval")
    assert score.score == 1.


@pytest.mark.parametrize("raw", [
    {"nodes": [{"id": "a", "instruction": "x", "dependencies": "a"}], "output_node": "a"},
    {"nodes": [{"id": "a"}], "output_node": "a"},
    {"nodes": "not-a-node-list", "output_node": "a"},
])
def test_malformed_workflow_is_rejected_before_any_generation(raw):
    with pytest.raises(ValueError):
        Workflow.from_dict(raw)
