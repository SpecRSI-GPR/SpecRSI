"""CPU integration through real experiment/collection runners with fake model I/O."""
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from specrsi.benchmarks import BenchmarkExample
from specrsi.core import Candidate, EvaluationScope, Outcome, Task
from specrsi.core.execution import ExecutionResult
from specrsi.core.search import DraftResult
from specrsi.models import ModelReply
from specrsi.paper_profile import require_role_checkpoint, vats_config


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [0] * 4

    def encode(self, text, **kwargs):
        return [1] * 8


class FakeRoles:
    backend = "hf"
    device = "cpu"
    config = SimpleNamespace(max_length=128, feature_dim=8, option_dim=12, architecture_version=2)
    tokenizer = Tokenizer()

    def eval(self):
        return self

    def decision_values(self, decision):
        return torch.zeros(len(decision.support))

    def decision_logits(self, decision):
        # Prefer complete-check plans, then an edit; ACCEPT remains available.
        return torch.tensor([2.0 * option[2] + option[1] for option in decision.option_features])


class FakeDraft:
    def __init__(self, bridge, **kwargs):
        self.bridge = bridge

    def draft(self, parent, state, cap, rng):
        self.bridge.physical_inferences.append({"kind": "draft", "elapsed_seconds": .01,
                                               "gpu_count": 0, "prompt_tokens": 4, "completion_tokens": 2})
        child = Candidate("improved", {"system_prompt": "improved"}, parent.runtime,
                          "opaque", False, parent_id=parent.uid)
        return DraftResult(child, 6)


def fixture(tmp_path, monkeypatch, *, learned=False):
    import transformers
    import specrsi.experiments as experiments
    import specrsi.models as models
    import specrsi.benchmarks as benchmarks
    import specrsi.learning.bridge as bridge
    import specrsi.learning.hf as hf

    output = tmp_path / "run"
    lock = tmp_path / "model.lock.json"
    lock.write_text(json.dumps({"revision": "pinned", "local_path": "/fake/model"}))
    seen = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def health(self):
            return {"revision": "pinned", "server_enforces_total_token_cap": True, "protocol_version": 2}

        def generate(self, messages, **kwargs):
            seen.append(messages)
            if messages[-1]["content"] == "audit question":
                assert (output / "selected.json").is_file(), "Audit must begin after the selected artifact is saved"
                assert (output / "search.json").is_file()
            if messages[0]["content"].startswith("Improve"):
                answer = json.dumps({"system_prompt": "improved"})
            else:
                answer = "Paris" if messages[0]["content"] == "improved" else "wrong"
            return ModelReply(answer, 4, 2, .01, "stop", "pinned", f"fake-{len(seen)}")

    def load_examples(dataset, split, **kwargs):
        return [BenchmarkExample(split, dataset, split, f"{split} question", "Paris")]

    monkeypatch.setattr(experiments, "LocalModelClient", Client)
    monkeypatch.setattr(models, "LocalModelClient", Client)
    monkeypatch.setattr(experiments, "load_examples", load_examples)
    monkeypatch.setattr(benchmarks, "load_examples", load_examples)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(hf.HFRoleModel, "load", lambda *a, **k: FakeRoles())
    monkeypatch.setattr(bridge, "HFPromptDrafter", FakeDraft)
    search = {"algorithm": "vats_pdf", "controller": "learned" if learned else "heuristic_bootstrap",
              "budget_tokens": 2000, "max_actions": 3, "max_candidates": 2, "eta": 0, "cost_lambda": .001,
              "execution_cap_tokens": 16, "draft_cap_tokens": 16, "initial_cap_tokens": 16,
              "initial_prompt": "initial", "vats": {"max_queries": 8, "max_extensions": 0,
              "max_draft_depth": 1, "max_draft_parents": 1, "max_completion_targets": 1,
              "max_pair_targets": 0, "max_singletons": 1, "max_menu_options": 4, "solver_call_cost": 0}}
    if not learned:
        search["vats"].update(scorer_query_cap=0, policy_call_cost=0)
    else:
        checkpoint = tmp_path / "roles"
        checkpoint.mkdir()
        (checkpoint / "model_config.json").write_text(json.dumps({"backend": "hf", "architecture_version": 2,
                                                                 "option_dim": 12, "feature_dim": 8}))
        (checkpoint / "heads.pt").write_bytes(b"fake-test-only")
        (checkpoint / "adapters").mkdir()
        search["role_checkpoint"] = str(checkpoint)
        search["role_device"] = "cpu"
    config = {"purpose": "integration_fixture_not_benchmark", "dataset": "hotpotqa", "seed": 7,
              "state_dir": str(tmp_path / "state"), "output": str(output),
              "model": {"lock": str(lock), "base_url": "http://fake", "max_tokens": 4}, "search": search}
    return config, seen


@pytest.mark.parametrize("learned", [False, True])
def test_real_pdf_runner_uses_vats_and_freezes_successor_before_separate_audit(tmp_path, monkeypatch, learned):
    from specrsi.experiments import run_local_experiment
    config, seen = fixture(tmp_path, monkeypatch, learned=learned)
    result = run_local_experiment(config)
    summary = result["search"]
    assert summary["algorithm"] == "vats_pdf"
    assert summary["initial_f"] == 0 and summary["final_f"] == 1
    assert not summary["completion_reservation_enabled"]
    assert summary["initial_acquisition_tokens"] == 6
    assert summary["total_cost"] <= config["search"]["budget_tokens"]
    assert sum(item["cost"] for item in summary["transitions"]) == summary["total_cost"]
    assert result["audit"][0]["score"] == 1 and result["audit"][0]["tokens"] == 6
    assert seen[-1][-1]["content"] == "audit question"
    assert not summary["cost_accounting"]["physical_bill_equals_budget_spend"]
    if learned:
        assert summary["role_checkpoint_hashes"]
        assert summary["cost_accounting"]["policy_charge_per_menu"] == 128
        events = [event for transition in summary["transitions"] for event in transition["events"]]
        assert any(event["event"] == "scorer_query" for event in events)
    else:
        assert summary["total_cost"] == 12  # One paid edit and one paid check; audit/initial are separate.


def test_pdf_checkpoint_fails_before_any_task_model_call(tmp_path, monkeypatch):
    from specrsi.experiments import run_local_experiment
    config, seen = fixture(tmp_path, monkeypatch, learned=True)
    manifest = Path(config["search"]["role_checkpoint"]) / "model_config.json"
    manifest.write_text(json.dumps({"backend": "hf", "option_dim": 8, "feature_dim": 8}))
    with pytest.raises(ValueError, match="architecture_version"):
        run_local_experiment(config)
    assert not seen


def test_missing_role_weights_are_rejected_before_evaluation(tmp_path):
    (tmp_path / "model_config.json").write_text(json.dumps({"backend": "hf", "architecture_version": 2,
                                                          "option_dim": 12, "feature_dim": 8}))
    with pytest.raises(ValueError, match="incomplete"):
        require_role_checkpoint({"algorithm": "vats_pdf", "controller": "learned", "role_checkpoint": str(tmp_path)})


def test_pdf_evaluation_does_not_overwrite_previous_results(tmp_path, monkeypatch):
    from specrsi.experiments import run_local_experiment
    config, seen = fixture(tmp_path, monkeypatch)
    output = Path(config['output'])
    output.mkdir()
    result = output / 'search.json'
    result.write_text('original evidence')
    with pytest.raises(FileExistsError):
        run_local_experiment(config)
    assert result.read_text() == 'original evidence' and not seen


def test_pdf_bad_menu_fails_before_initial_task_execution(tmp_path, monkeypatch):
    from specrsi.experiments import run_local_experiment
    config, seen = fixture(tmp_path, monkeypatch)
    config['search']['vats']['max_menu_options'] = 2
    with pytest.raises(ValueError, match='menu cap'):
        run_local_experiment(config)
    assert not seen
    assert not Path(config['output']).exists()


def test_registered_max_children_zero_prevents_drafting(tmp_path, monkeypatch):
    from specrsi.experiments import run_local_experiment
    config, seen = fixture(tmp_path, monkeypatch)
    config['search']['max_children'] = 0
    config['search']['time_mu'] = .25
    result = run_local_experiment(config)
    summary = result['search']
    assert summary['final_f'] == summary['initial_f'] == 0
    assert not any(messages[0]['content'].startswith('Improve') for messages in seen)
    expected = -.25 * summary['timing']['objective_elapsed_seconds']
    assert summary['search_return'] == pytest.approx(expected)


def test_real_prompt_collector_uses_development_only_and_preserves_pdf_snapshot(tmp_path, monkeypatch):
    from specrsi.learning.bridge import collect_prompt_specrsi
    config, seen = fixture(tmp_path, monkeypatch, learned=True)
    collection = {"algorithm": "vats_pdf", "dataset": "hotpotqa", "contexts": 1, "tasks_per_context": 1,
                  "group_size": 2, "max_actions": 3, "max_candidates": 2, "budget_tokens": 2000,
                  "execution_cap": 16, "draft_cap": 16, "initial_prompt": "initial", "exploration": .1,
                  "executor": config["model"], "vats": config["search"]["vats"]}
    result = collect_prompt_specrsi(FakeRoles(), 3, {"collection": collection, "output": str(tmp_path / "collection"),
                                                  "training": {"seed": 17}})
    group = result["groups"][0]
    assert all(row.collection_snapshot == "hf-frozen-round-3" for row in group)
    assert len({row.initial_context_id for row in group}) == 1
    assert result["metadata"]["initial_acquisition_tokens"] == 6
    assert result["metadata"]["search_cost"] == sum(row.total_cost for row in group)
    assert {messages[-1]["content"] for messages in seen} == {"development question"}
    assert result["metadata"]["cost_accounting"]["policy_charge_per_menu"] == 128


@pytest.mark.parametrize("name,value", [("all_u", False), ("mode", "deploy"), ("max_sources", 1), ("cost_unit", "fake")])
def test_nested_vats_options_cannot_change_shared_contract(name, value):
    from specrsi.core import SearchConfig
    with pytest.raises(ValueError, match="cannot override"):
        vats_config(SearchConfig(), {"vats": {name: value}})


def test_hf_prompt_tokens_use_work_transition_index_for_multilayer_drafts(monkeypatch):
    import specrsi.learning.sampler as sampler
    from specrsi.learning.bridge import HFPromptDrafter
    captured = {}

    def sample(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="improved", token_cost=6, decisions=(), stop_reason="eos")

    monkeypatch.setattr(sampler, "sample_edit_tokens", sample)
    bridge = SimpleNamespace(model=FakeRoles(), cost_unit="tokens", direct_only=True,
                             context_text=lambda state: "observed", bounded_text=lambda text, limit: text,
                             physical_inferences=[], token_decisions=[])
    state = SimpleNamespace(version=99, current_transition_index=4, scope=SimpleNamespace(tasks=(Task("one"),)))
    result = HFPromptDrafter(bridge, max_new_tokens=4).draft(
        Candidate("parent", {"system_prompt": "initial"}), state, 64, None)
    assert result.status == "complete"
    assert captured["transition_index"] == 4


def test_science_runner_records_native_calls_and_reports_increment_not_absolute_utility(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/run_paper_science.py"
    spec = importlib.util.spec_from_file_location("science_pdf_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "science"

    def scope(domain, split, cases, **kwargs):
        return EvaluationScope(split, tuple(Task(f"{split}-{case}") for case in cases), "native-test", "native-test",
                               scored_fields=("native_score",), aggregation_id="science_native_weighted_mean_v1", split=split)

    class Executor:
        def run(self, candidate, task, scope, cap, resume_token=None):
            if scope.split == "audit":
                assert (output / "selected.json").is_file()
                assert (output / "search.json").is_file()
            score = 12 if candidate.name == "improved" else 10
            return ExecutionResult("complete", 3, Outcome(.5, {"native_score": score}), receipt={"fixture": True})

    class Drafter:
        def draft(self, parent, state, cap, rng):
            return DraftResult(Candidate("improved", {"value": 12}, parent_id=parent.uid), 2)

    monkeypatch.setattr(runner, "make_scope", scope)
    monkeypatch.setattr(runner, "make_candidate", lambda domain: Candidate("initial", {"value": 10}))
    monkeypatch.setattr(runner, "ScienceExecutor", Executor)
    monkeypatch.setattr(runner, "ParameterDrafter", Drafter)
    report = runner.run({"domain": "reactor", "selection_cases": [0], "audit_cases": [0], "seed": 3,
                         "output": str(output), "search": {"algorithm": "vats_pdf", "controller": "heuristic_bootstrap",
                         "max_actions": 3, "max_candidates": 2, "budget_ms": 100, "execution_cap_ms": 20,
                         "draft_cap_ms": 10, "vats": {"max_queries": 8, "max_extensions": 0,
                         "max_draft_depth": 1, "max_draft_parents": 1, "max_completion_targets": 1,
                         "max_pair_targets": 0, "max_singletons": 1, "max_menu_options": 4,
                         "solver_call_cost": 0, "scorer_query_cap": 0, "policy_call_cost": 0}}})
    assert report["selection_native_gain"] == 2 and report["selection_native_utility"] == 12
    assert report["audit_complete"]
    calls = [json.loads(line) for line in (output / "physical_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 4 and sum(call["cost_ms"] for call in calls) == 12
    assert {call["split"] for call in calls} == {"selection", "audit"}
