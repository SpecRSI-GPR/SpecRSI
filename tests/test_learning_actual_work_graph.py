"""Opt-in actual-job graphs preserve old checkpoints and physical admission."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from specrsi.core.vats import VATSEngine, VATSPlan
from specrsi.learning.frozen_text import node_text_cache, semantic_input_cap
from specrsi.learning.heads import TinyModelConfig, TinyRoleModel
from specrsi.learning.paper_bridge import VATSSearchBridge
from specrsi.learning.records import _decision
from specrsi.learning.work_graph import ACTUAL_GRAPH_SCHEMA, LEGACY_GRAPH_SCHEMA
from specrsi.planning.completion import CompletionProblem, VerificationJob
from test_learning_semantic_graph import state_and_plans, hf_roles


def fixture():
    state, original = state_and_plans()
    keys = tuple(plan.jobs[0].keys[0] for plan in original)
    left = VerificationJob("independent-read", (keys[0],), 1, group="shared")
    right = VerificationJob("independent-write", (keys[1],), 1, group="shared")
    prerequisite = VerificationJob("prepare", (), 1, group="joint", kind="prerequisite")
    joint = VerificationJob("joint-result", keys, 1, group="joint")
    problem = CompletionProblem((left, right, prerequisite, joint), {"shared": 2, "joint": 2}, 1000, 10,
        dependencies=((joint.uid, prerequisite.uid),), conflicts=((left.uid, joint.uid),))
    state.job_catalog = problem
    plans = (VATSPlan("verify", jobs=(left, right), work_cap=4, attempt_cap=2),
             VATSPlan("verify", jobs=(prerequisite, joint), work_cap=4, attempt_cap=2))
    assert all(problem.feasible(plan.job_ids) for plan in plans)
    return state, plans


def tiny(schema=ACTUAL_GRAPH_SCHEMA):
    return TinyRoleModel(TinyModelConfig(architecture_version=3, option_dim=12, hidden_dim=16,
                                       graph_schema=schema)).eval()


def actual_hf(tmp_path, **overrides):
    from specrsi.learning.hf import HFRoleModel
    old = hf_roles(tmp_path)
    config = replace(old.config, graph_schema=ACTUAL_GRAPH_SCHEMA, semantic_budget_mode="redistribute",
        semantic_global_tokens=96, semantic_node_min_tokens=16, semantic_node_max_tokens=128,
        semantic_total_tokens=1024, **overrides)
    return HFRoleModel(config, tokenizer=old.tokenizer).eval()


def runner():
    p = Path(__file__).resolve().parents[1] / "scripts/run_registered_comparison.py"
    spec = importlib.util.spec_from_file_location("actual_work_graph_registered_runner", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_coverage_and_cost_plans_have_distinct_actual_work_inputs_and_gradients():
    state, plans = fixture()
    old = VATSSearchBridge(tiny(LEGACY_GRAPH_SCHEMA))._record(state, plans)
    assert old.option_features[0] == old.option_features[1]
    assert old.option_node_masks[0] == old.option_node_masks[1]
    model = tiny()
    decision = VATSSearchBridge(model)._record(state, plans)
    assert decision.option_features[0] == decision.option_features[1]
    assert decision.option_node_masks[0] != decision.option_node_masks[1]
    assert len(decision.graph_nodes) == len(old.graph_nodes) + 4 + 2
    assert {edge[2] for edge in decision.graph_edges} == set(range(11))
    assert any(node[2] == 2. and node[:2] == (0., 0.) for node in decision.graph_nodes)
    payload = json.dumps(asdict(decision))
    assert "918273645123" not in payload and "PRIVATE_METADATA" not in payload
    value = model.decision_values(decision)
    assert not torch.isclose(value[0], value[1], atol=1e-6)
    (value[0] - value[1] - 1.).square().backward()
    assert model.value_head.messages[6].weight.grad.abs().max() > 0
    assert all(p.grad is None for p in model.policy_head.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())


def test_dependency_conflict_and_setup_semantics_survive_replay_and_permutation(tmp_path):
    state, plans = fixture()
    model = tiny()
    decision = VATSSearchBridge(model)._record(state, plans)
    replay = _decision(json.loads(json.dumps(asdict(decision))))
    assert replay == decision
    size = len(decision.graph_nodes)
    reversed_record = replace(decision, graph_nodes=decision.graph_nodes[::-1],
        graph_node_texts=decision.graph_node_texts[::-1],
        graph_node_text_features=decision.graph_node_text_features[::-1],
        graph_edges=tuple((size-1-a, size-1-b, kind) for a, b, kind in decision.graph_edges),
        option_node_masks=tuple(mask[::-1] for mask in decision.option_node_masks))
    model.save(tmp_path)
    restored = TinyRoleModel.load(tmp_path).eval()
    for method in ("decision_logits", "decision_values"):
        expected = getattr(model, method)(decision)
        assert torch.equal(expected, getattr(restored, method)(replay))
        assert torch.allclose(expected, getattr(model, method)(reversed_record), atol=1e-6)
    changed = replace(state, job_catalog=replace(state.job_catalog, dependencies=(), conflicts=()))
    new = VATSSearchBridge(model)._record(changed, plans)
    assert new.graph_edges != decision.graph_edges
    assert not {6, 7, 8}.intersection(edge[2] for edge in new.graph_edges)


def test_missing_or_relabelled_graph_contract_is_rejected(tmp_path):
    state, plans = fixture()
    model = tiny()
    bridge = VATSSearchBridge(model)
    with pytest.raises(ValueError, match="immutable verification job catalog"):
        bridge._record(replace(state, job_catalog=None), plans)
    decision = bridge._record(state, plans)
    with pytest.raises(ValueError, match="graph_schema differs"):
        tiny(LEGACY_GRAPH_SCHEMA).decision_values(decision)
    relabelled = tiny(LEGACY_GRAPH_SCHEMA)
    relabelled.config.graph_schema = ACTUAL_GRAPH_SCHEMA
    with pytest.raises(ValueError, match="separately trained matching role heads"):
        relabelled.decision_values(decision)
    with pytest.raises(ValueError, match="registered directed relation"):
        replace(decision, graph_schema=LEGACY_GRAPH_SCHEMA)
    model.save(tmp_path)
    p = tmp_path / "model_config.json"
    raw = json.loads(p.read_text()); raw.pop("graph_schema"); p.write_text(json.dumps(raw))
    with pytest.raises((ValueError, RuntimeError)):
        TinyRoleModel.load(tmp_path)
    with pytest.raises(ValueError, match="architecture_version=3"):
        TinyModelConfig(graph_schema=ACTUAL_GRAPH_SCHEMA)


def test_default_schema_preserves_exact_pre_schema_encoder_and_checkpoint(tmp_path):
    model = tiny(LEGACY_GRAPH_SCHEMA)
    identity = node_text_cache(model).identity
    old_config = asdict(model.config); old_config.pop("graph_schema")
    model.config = SimpleNamespace(**old_config)
    assert node_text_cache(model).identity == identity
    model.config = TinyModelConfig(**old_config)
    state, plans = fixture(); d = VATSSearchBridge(model)._record(state, plans)
    raw = asdict(d); raw.pop("graph_schema")
    assert _decision(raw) == d
    model.save(tmp_path)
    manifest = tmp_path / "model_config.json"
    raw = json.loads(manifest.read_text()); raw.pop("graph_schema"); manifest.write_text(json.dumps(raw))
    loaded = TinyRoleModel.load(tmp_path)
    assert len(loaded.value_head.messages) == 4
    assert node_text_cache(loaded).identity == identity
    assert torch.equal(model.decision_values(d), loaded.decision_values(d))
    assert node_text_cache(tiny()).identity != identity


def test_catalog_reaches_both_conservative_queries_and_paid_policy_then_snapshot():
    from test_core_vats import make_vats
    seen = []
    def observe(state, plans):
        assert isinstance(state.job_catalog, CompletionProblem)
        seen.append(state.job_catalog)
        return [5. if plan.kind == "draft" and len(state.candidates) == 1 else -5. for plan in plans]
    def score(state, plan):
        assert state.job_catalog is not None
        assert all(job == state.job_catalog.by_id[job.uid] for job in plan.jobs)
        return 0.
    engine = make_vats(policy_menu=observe, scorer=score, capture_paid_states=True)
    engine.run()
    assert len(seen) == 2
    assert all(s.state.job_catalog is s.problem for s in engine.paid_snapshots)
    saved = engine.paid_snapshots[-1]
    fork = VATSEngine.from_paid_state(saved, "ACCEPT", executor=engine.executor, drafter=engine.drafter, seed=3)
    assert fork.state.job_catalog is saved.problem


def test_hf_graph_forward_admission_fixed_token_cap_actual_receipts_and_replay(tmp_path, monkeypatch):
    model = actual_hf(tmp_path)
    state, plans = fixture(); bridge = VATSSearchBridge(model, cost_unit="tokens")
    api = runner(); cap, call_cap = api.role_admission(bridge, state)
    assert cap == semantic_input_cap(model.config) == 1024
    assert call_cap == bridge.semantic_forward_cap(state)
    assert not bridge.physical_inferences  # Preflight cannot launch encodings.
    meter = api.PhysicalMeter({"initial": {"tokens": 2048, "calls": 64}}, tmp_path / "calls.jsonl")
    api.measured_role(bridge, meter, lambda: bridge.scorer(state, plans[0]), cap, "scorer", calls=call_cap)
    cold = list(bridge.physical_inferences)
    assert len(cold) <= call_cap and sum(row["prompt_tokens"] for row in cold) <= cap
    assert meter.spent["initial"] == {"tokens": sum(row["prompt_tokens"] for row in cold), "calls": len(cold)}
    assert not meter.unknown_reservations and meter.pending is None
    logits = bridge.policy_menu(state, plans)
    captured = bridge.captured[(state.version, "plan", plans[0].uid)]
    assert captured.graph_schema == ACTUAL_GRAPH_SCHEMA
    assert len(captured.graph_node_input_ids) == len(captured.graph_nodes)
    count = len(bridge.physical_inferences)
    bridge.scorer(state, plans[0])
    assert len(bridge.physical_inferences) == count + 1
    assert api.role_admission(bridge, state) == (cap, call_cap)
    replay = bridge._record(state, plans)
    assert torch.equal(torch.tensor(logits), model.decision_logits(replay).detach())
    model.save(tmp_path / "actual")
    from specrsi.learning.hf import HFRoleModel
    loaded = HFRoleModel.load(tmp_path / "actual", device="cpu").eval()
    assert torch.equal(model.decision_values(replay), loaded.decision_values(replay))
    incompatible = replace(replay, graph_schema=LEGACY_GRAPH_SCHEMA, graph_edges=())
    monkeypatch.setattr(model, "encode", lambda *a, **kw: pytest.fail("no paid forward before schema rejection"))
    for method in (model.decision_values, model.decision_logits):
        with pytest.raises(ValueError, match="graph_schema differs"):
            method(incompatible)


def test_unfittable_graph_minimum_rejected_before_paid_admission_or_forward(tmp_path):
    model = actual_hf(tmp_path)
    model.config.semantic_node_min_tokens = 128
    model.config.semantic_node_max_tokens = 128
    model.config.semantic_total_tokens = 129
    state, plans = fixture(); bridge = VATSSearchBridge(model, cost_unit="tokens")
    with pytest.raises(ValueError, match="every unique node"):
        runner().role_admission(bridge, state)
    assert not bridge.physical_inferences
    with pytest.raises(ValueError, match="every unique node"):
        bridge._record(state, plans)
    assert not bridge.physical_inferences


def test_explicit_resume_graph_schema_mismatch_fails_before_model_load(tmp_path, monkeypatch):
    from test_semantic_profile_entrypoints import training_script
    from specrsi.learning.hf import HFRoleModel
    (tmp_path / "model_config.json").write_text(json.dumps({"backend":"hf", "architecture_version":3}))
    monkeypatch.setattr(HFRoleModel, "load", lambda *a, **k: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="graph_schema differs"):
        training_script().create_model({"model":{"graph_schema":ACTUAL_GRAPH_SCHEMA}}, tmp_path)


def test_semantic_excerpt_step_below_zero_terminates_with_budget_error():
    from specrsi.learning.frozen_text import bounded_semantic_text
    class Characters:
        def encode(self, text, **kwargs):
            return list(map(ord, text))
        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))
    # The omission marker itself exceeds cap=1. The first decrement jumps from
    # count=1 to a negative number, which must not keep the loop running.
    with pytest.raises(ValueError, match="cannot fit"):
        bounded_semantic_text(Characters(), "a long visible observation", 1)
