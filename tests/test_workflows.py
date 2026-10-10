from types import SimpleNamespace
import pytest
from specrsi.workflows import Workflow, execute_workflow
from specrsi.models import ModelReply


def test_cycles_and_disconnected_nodes_rejected():
    with pytest.raises(ValueError, match="cycle"):
        Workflow.from_dict({"nodes": [{"id": "a", "instruction": "x", "dependencies": ["b"]},
                                     {"id": "b", "instruction": "x", "dependencies": ["a"]}], "output_node": "b"})
    with pytest.raises(ValueError, match="disconnected"):
        Workflow.from_dict({"nodes": [{"id": "a", "instruction": "x"},
                                     {"id": "b", "instruction": "x"}], "output_node": "b"})


def test_interrupted_prefix_has_no_complete_answer():
    calls = []
    def generate(messages, **kwargs):
        calls.append(messages)
        return ModelReply("partial", 4, 3, .1, "stop", "r", str(len(calls)))
    executor = SimpleNamespace(config={"max_tokens": 3}, prompt_tokens=lambda _: 4,
                               client=SimpleNamespace(generate=generate))
    workflow = Workflow.from_dict({"nodes": [{"id": "a", "instruction": "solve"},
        {"id": "b", "instruction": "review", "dependencies": ["a"]}], "output_node": "b"})
    answer, cost, records = execute_workflow(workflow, "task", executor, cap=10, seed=0)
    assert answer is None and cost == 7 and len(records) == len(calls) == 1
