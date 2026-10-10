"""No server/model calls: sampler contracts across executor deployment paths."""
import json

import pytest

from specrsi.answer_strategies import AnswerStrategy
from specrsi.core.models import Candidate, EvaluationScope, Task
from specrsi.experiments import QwenExecutor
from specrsi.models import ModelReply
from specrsi.models.client import EXTENDED_SAMPLING_PARAMETERS


SAMPLING = {"top_k": 20, "min_p": 0., "presence_penalty": 1.5, "repetition_penalty": 1.05}
PATHS = ("ordinary", "reasoning_extract_v1", "reasoning_extract_v2", "native_completion_v3", "workflow")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] == kwargs["thinking"]
        return [0] * (6 if kwargs["enable_thinking"] else 4)


class Client:
    def __init__(self, supported=EXTENDED_SAMPLING_PARAMETERS):
        self.supported = set(supported)
        self.preflights, self.calls = [], []

    def ensure_sampling_support(self, options):
        self.preflights.append(dict(options))
        if set(options) - self.supported:
            raise ValueError("unsupported sampler controls")

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        # Force native-v3's paid completion path, so both physical stages are
        # exercised. The same partial response is sufficient as v1/v2 notes.
        first = len(self.calls) == 1
        return ModelReply("partial public calculation" if first else r"\boxed{42}",
            6 if kwargs["enable_thinking"] else 4, 2, .1,
            "length" if first else "stop", "frozen", f"request-{len(self.calls)}",
            reasoning_content="public working notes" if first else None)


def setup(path, *, options=SAMPLING, supported=EXTENDED_SAMPLING_PARAMETERS):
    generation = {"max_tokens": 8, "temperature": .6, "top_p": .95,
                  "enable_thinking": True, "max_seconds": 20, **options}
    artifacts = {"system_prompt": "REGISTERED_SYSTEM"}
    if path == "workflow":
        artifacts = {"workflow": {"nodes": [
            {"id": "derive", "instruction": "Derive the result."},
            {"id": "answer", "instruction": "Return the boxed answer.", "dependencies": ["derive"]}],
            "output_node": "answer"}}
    elif path != "ordinary":
        artifacts["answer_strategy"] = AnswerStrategy(version=path, mode="reasoning_extract",
            answer_format="boxed_integer", reasoning_tokens=8, answer_tokens=4, temperature=.6).to_dict()
    candidate = Candidate(path, artifacts, {"generation": generation, "seed": 7, "model_revision": "frozen"})
    task = Task("fixture", {"prompt": "PUBLIC_INPUT"}, 42,
                {"dataset": "aime25", "private": "PRIVATE_METADATA"})
    scope = EvaluationScope("test-sampling", (task,), "frozen", "integer-score")
    client = Client(supported)
    # Different executor defaults ensure frozen candidate runtime wins on every
    # deployment path, including each node in the constructed workflow executor.
    executor = QwenExecutor(client, Tokenizer(), {"top_k": 3, "min_p": .5,
        "presence_penalty": -.5, "repetition_penalty": .9, "enable_thinking": False})
    return executor, client, candidate, task, scope


@pytest.mark.parametrize("path", PATHS)
def test_registered_sampling_reaches_every_physical_call(path):
    executor, client, candidate, task, scope = setup(path)
    executor.validate_prompt_candidate(candidate, [task], 80)
    assert client.preflights == [SAMPLING]
    assert not client.calls
    result = executor.run(candidate, task, scope, 80)
    assert result.status == "complete"
    assert len(client.calls) == (1 if path == "ordinary" else 2)
    for messages, kwargs in client.calls:
        assert {key: kwargs[key] for key in SAMPLING} == SAMPLING
        assert kwargs["top_p"] == .95 and kwargs["max_seconds"] == 20
        assert "PRIVATE_METADATA" not in json.dumps(messages)
    assert [kw["seed"] for _, kw in client.calls] == list(range(7, 7 + len(client.calls)))
    assert result.actual_cost == sum(r["prompt_tokens"] + r["completion_tokens"] for r in executor.physical_records)
    if path in {"ordinary", "workflow"}:
        assert all(kw["enable_thinking"] is True and kw["temperature"] == .6 for _, kw in client.calls)
    else:
        assert [kw["enable_thinking"] for _, kw in client.calls] == [path != "reasoning_extract_v1", False]
        assert [kw["temperature"] for _, kw in client.calls] == [.6, 0.]


@pytest.mark.parametrize("path", PATHS)
def test_unsupported_sampling_is_rejected_by_preflight_before_generation(path):
    executor, client, candidate, task, _ = setup(path, supported=("top_k",))
    with pytest.raises(ValueError, match="unsupported sampler"):
        executor.validate_prompt_candidate(candidate, [task], 80)
    assert client.preflights == [SAMPLING]
    assert not client.calls and not executor.physical_records


@pytest.mark.parametrize("path", PATHS)
def test_invalid_sampling_is_rejected_before_capability_lookup(path):
    executor, client, candidate, task, _ = setup(path, options={"top_k": True})
    with pytest.raises(ValueError, match="top_k"):
        executor.validate_prompt_candidate(candidate, [task], 80)
    assert not client.preflights and not client.calls and not executor.physical_records


@pytest.mark.parametrize("path", PATHS)
def test_absent_optional_sampling_preserves_old_payloads_and_needs_no_capability_probe(path):
    executor, client, candidate, task, scope = setup(path, options={}, supported=())
    executor.validate_prompt_candidate(candidate, [task], 80)
    result = executor.run(candidate, task, scope, 80)
    assert result.status == "complete" and not client.preflights
    assert all(not set(EXTENDED_SAMPLING_PARAMETERS).intersection(kw) for _, kw in client.calls)


def test_native_first_call_matches_ordinary_prompt_and_decoding_when_both_are_registered_identically():
    ordinary, ordinary_client, control, task, scope = setup("ordinary")
    strategy, strategy_client, candidate, _, _ = setup("native_completion_v3")
    ordinary.run(control, task, scope, 14)
    strategy.run(candidate, task, scope, 80)
    control_messages, control_options = ordinary_client.calls[0]
    strategy_messages, strategy_options = strategy_client.calls[0]
    assert strategy_messages == control_messages
    # The native stage reserves exactly its complete contract (6 prompt + 8
    # completion); the ordinary control is explicitly given that same cap here.
    assert strategy_options == control_options


def test_native_preflight_uses_native_thinking_template_and_full_registered_output_cap():
    executor, client, candidate, task, _ = setup("native_completion_v3", options={})
    with pytest.raises(ValueError, match="requires 14 tokens"):
        executor.validate_prompt_candidate(candidate, [task], 13)
    executor.validate_prompt_candidate(candidate, [task], 14)
    assert not client.calls and not executor.physical_records
