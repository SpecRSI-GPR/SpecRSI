"""Scientific edits need text feedback even when scorer observations live on nodes."""
import json
import random
from types import SimpleNamespace

import pytest

from specrsi.core import Candidate, Outcome, Task
from specrsi.core.models import thaw
from specrsi.learning.science_bridge import HFParameterDrafter
from test_learning_draft_feedback import CharacterTokenizer


def fixture(max_length=8192):
    tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = 0
    model = SimpleNamespace(backend="hf", device="cpu", tokenizer=tokenizer,
        config=SimpleNamespace(max_length=max_length, architecture_version=3,
                               role_observation_version="observed_results_v2"))
    bridge = SimpleNamespace(model=model, cost_unit="wall-ms", direct_only=True,
        physical_inferences=[], token_decisions=[],
        context_text=lambda state: pytest.fail("Graph-role global context omits draft task/result text"))
    parent = Candidate("parent", {"parameters": {"rate": 1.}, "scenario_overrides": {}},
                       {"domain": "reactor", "contract": "science-parameter-kernel-v2"})
    tasks = (
        Task("success", {"domain": "reactor", "case_id": 17}, "PRIVATE_REFERENCE_17",
             metadata={"secret": "PRIVATE_METADATA"}),
        Task("failure", {"domain": "reactor", "case_id": 4}, "PRIVATE_REFERENCE_4"),
    )
    rows = {tasks[0].uid: SimpleNamespace(outcome=Outcome(.9, {"native_score": 9., "valid": True})),
            tasks[1].uid: SimpleNamespace(outcome=Outcome(.2, {"native_score": 2., "valid": False}, admissible=False))}
    state = SimpleNamespace(scope=SimpleNamespace(tasks=tasks), version=1, current_transition_index=3,
        ledger=SimpleNamespace(get=lambda candidate, task, scope: rows.get(task.uid), missing=lambda *args: ()))
    return bridge, parent, state


@pytest.mark.parametrize("mode", ["numeric_catalog", "freeform_json"])
def test_scientific_draft_sees_route_indices_case_ids_and_observed_parent_only(monkeypatch, mode):
    bridge, parent, state = fixture()
    drafter = HFParameterDrafter(bridge, max_new_tokens=128, proposal_mode=mode)
    monkeypatch.setattr(drafter, "catalog", lambda *args: [
        {"task_index": 1, "parameters": {"rate": 2.}, "changes": {"rate": 2.}}])
    prompts = []

    def sample(model, input_ids, **kwargs):
        assert len(input_ids) + kwargs["max_new_tokens"] <= model.config.max_length
        prompts.append(model.tokenizer.decode(input_ids))
        text = '{"edit":0}' if mode == "numeric_catalog" else '{"task_index":1,"parameters":{"rate":2.0}}'
        return SimpleNamespace(text=text, decisions=(), generated_ids=(1,), stop_reason="eos")

    monkeypatch.setattr("specrsi.learning.science_bridge.sample_edit_tokens", sample)
    result = drafter.draft(parent, state, 10000, random.Random(0))
    assert result.status == "complete"
    assert thaw(result.candidate.artifacts)["scenario_overrides"] == {"case:4": {"rate": 2.}}
    text = prompts[0]
    routes = json.loads(text.split("Registered task routes: ")[1].split("\nObserved parent feedback: ")[0])
    assert [(row["task_index"], row["visible_input"]["case_id"]) for row in routes] == [(0, 17), (1, 4)]
    feedback = json.loads(text.split("Observed parent feedback: ")[1].removesuffix("\nASSISTANT\n"))
    assert feedback["examples"][0]["task_id"] == state.scope.tasks[1].uid[:12]
    assert feedback["examples"][0]["parent_observed_outcome"]["native_score"] == 2.
    assert "Do not modify the simulator, constraints, task inputs or scoring" in text
    assert "PRIVATE_REFERENCE" not in text and "PRIVATE_METADATA" not in text


def test_scientific_draft_rejects_context_that_cannot_preserve_route_and_parameter_contract(monkeypatch):
    bridge, parent, state = fixture(max_length=128)
    monkeypatch.setattr("specrsi.learning.science_bridge.sample_edit_tokens",
        lambda *args, **kwargs: pytest.fail("Unrepresentable contract must fail before generation"))
    result = HFParameterDrafter(bridge, max_new_tokens=32, proposal_mode="freeform_json").draft(
        parent, state, 10000, random.Random(0))
    assert result.status == "interrupted" and result.candidate is None
    assert "do not fit draft context" in result.error
    assert not bridge.physical_inferences
