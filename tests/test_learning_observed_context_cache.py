"""Explicit observed-result semantics and metered frozen-global cache opt-ins."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from specrsi.core import BudgetLedger, Candidate, EvaluationScope, Ledger, Task
from specrsi.core.models import Outcome
from specrsi.core.vats import VATSPlan, VATSState
from specrsi.learning.frozen_text import node_text_cache, semantic_input_cap
from specrsi.learning.heads import TinyModelConfig, TinyRoleModel
from specrsi.learning.hf import HFModelConfig, HFRoleModel
from specrsi.learning.paper_bridge import VATSSearchBridge
from specrsi.planning.completion import CompletionProblem, VerificationJob
from test_learning_semantic_graph import hf_roles, state_and_plans


def observed_state(answer):
    task = Task("same-task", {"question": "Explain the failing assertion"},
                "PRIVATE_REFERENCE_SENTINEL", {"private": "PRIVATE_METADATA_SENTINEL"})
    scope = EvaluationScope("observed-feedback", (task,), "fixture-executor", "fixture-scorer", ("answer",))
    root = Candidate("root", {"system_prompt": "same"}, {"seed": 37})
    child = Candidate("child", root.artifacts, root.runtime, parent_id=root.uid)
    ledger = Ledger()
    ledger.admit_direct(root, task, scope, Outcome(0., {"answer": answer, "valid": False,
        "private": "PRIVATE_OUTPUT_FIELD_SENTINEL"}), {"executor": "synthetic-fixture"})
    ledger.register(child, scope)
    job = VerificationJob("check-child", ((child.uid, task.uid),), 1)
    state = VATSState(scope, root, {root.uid: root, child.uid: child}, ledger, BudgetLedger(100),
        remaining_attempts=10, initial_attempts=10,
        job_catalog=CompletionProblem((job,), {"execution": 0}, 100, 10))
    return state, (VATSPlan("verify", jobs=(job,), work_cap=1, attempt_cap=1),)


def roles(tmp_path, **overrides):
    model = hf_roles(tmp_path)
    model.config = replace(model.config, semantic_budget_mode="redistribute", semantic_global_tokens=128,
        semantic_node_min_tokens=16, semantic_node_max_tokens=256, semantic_total_tokens=2048,
        role_observation_version="observed_results_v2", frozen_global_cache=True, **overrides)
    return model


def test_result_tail_enters_real_node_inputs_without_reference_or_unobserved_fields():
    left, plans = observed_state("X" * 160 + " FAIL: buffer off by one 👩🏽‍🔬")
    right, _ = observed_state("X" * 160 + " FAIL: wrong path dependency 中文")
    legacy = VATSSearchBridge(TinyRoleModel(TinyModelConfig(architecture_version=3, option_dim=12)))
    assert asdict(legacy._record(left, plans)) == asdict(legacy._record(right, plans))
    model = TinyRoleModel(TinyModelConfig(architecture_version=3, option_dim=12,
        role_observation_version="observed_results_v2"))
    bridge = VATSSearchBridge(model)
    a, b = bridge._record(left, plans), bridge._record(right, plans)
    assert a.graph_nodes == b.graph_nodes
    assert a.graph_node_texts != b.graph_node_texts
    assert a.graph_node_text_features != b.graph_node_text_features
    assert not torch.equal(model.decision_values(a), model.decision_values(b))
    assert not torch.equal(model.decision_logits(a), model.decision_logits(b))
    text = str(asdict(a))
    assert "buffer off by one" in text and "👩🏽‍🔬" in " ".join(a.graph_node_texts)
    assert "PRIVATE_" not in text
    assert any('"seed":37' in t for t in a.graph_node_texts)
    assert any('"observed":null' in t for t in a.graph_node_texts)
    assert len(bridge._graph_node_texts(left)) == len(a.graph_nodes)


def test_new_result_nodes_obey_unicode_exact_ids_and_shared_budget(tmp_path):
    model = roles(tmp_path)
    # Byte BPE can split individual Unicode codepoints; use actual tokenizer
    # encode/decode and the normal bounded observation path, with no model call.
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    raw = Tokenizer(models.BPE({s: i for i, s in enumerate(alphabet)}, []))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    model.tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    state, _ = observed_state("👩🏽‍🔬" * 500 + " 已观察的最后错误")
    context = bridge.context_text(state)
    texts, encoded = bridge._bounded_node_observations(state, context)
    assert all("\ufffd" not in text for text in [context, *texts])
    assert all(tuple(model.tokenizer.encode(text)) == encoded[text] for text in set(texts))
    assert all(len(ids) <= model.config.semantic_node_max_tokens for ids in encoded.values())
    assert len(model.tokenizer.encode(context)) <= model.config.semantic_global_tokens
    assert len(model.tokenizer.encode(context)) + sum(len(ids) for ids in set(encoded.values())) <= semantic_input_cap(model.config)
    assert bridge.semantic_forward_cap(state) == len(set(encoded.values())) + 1
    assert "PRIVATE_" not in " ".join(texts)


def test_cached_context_reuses_only_frozen_forward_but_charges_same_quota(tmp_path):
    model = roles(tmp_path)
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    state, plans = state_and_plans()
    backbone_calls, head_calls = [], []
    h1 = model.backbone.get_base_model().base_model.register_forward_hook(lambda *a: backbone_calls.append(1))
    h2 = model.value_head.register_forward_hook(lambda *a: head_calls.append(1))
    try:
        first = bridge.scorer(state, plans[0])
        n = len(backbone_calls)
        assert n > 1
        assert bridge.physical_inferences[-1]["encoder_forward_count"] == 1
        count = len(bridge.physical_inferences)
        second = bridge.scorer(state, plans[0])
        assert first == second
        assert len(backbone_calls) == n and len(head_calls) == 2
        assert len(bridge.physical_inferences) == count + 1
        row = bridge.physical_inferences[-1]
        assert row["encoder_cache_hit"] and row["encoder_forward_count"] == 0 and row["head_forward_count"] == 1
        assert row["prompt_tokens"] == row["completion_tokens"] == 0
        assert row["context_input_token_ids"] == model.tokenizer.encode(bridge.context_text(state))
        assert row["search_budget_charge"] == first.actual_cost > 0
        assert "input_token_ids" not in row
        assert sum(r["encoder_forward_count"] for r in bridge.physical_inferences) == n
        # Separate policy parameters see the same frozen observation; their
        # forward is never replaced by a cached scorer/policy output.
        bridge.policy_menu(state, plans)
        assert bridge.physical_inferences[-1]["encoder_cache_hit"]
        assert bridge.physical_inferences[-1]["head_forward_count"] == 1
        assert len(backbone_calls) == n
        with torch.no_grad():
            model.value_head.output[-1].bias.add_(1.)
        third = bridge.scorer(state, plans[0])
        assert third.value == pytest.approx(first.value + 1., abs=1e-6)
        assert len(backbone_calls) == n and len(head_calls) == 3
    finally:
        h1.remove(); h2.remove()


def test_cache_preserves_exact_outputs_gradients_and_invalidates_frozen_weights(tmp_path):
    model = roles(tmp_path)
    state, plans = state_and_plans()
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    decision = bridge._record(state, plans)
    expected = model._encode_frozen_text(decision.context_text, decision.input_ids)
    assert torch.equal(expected, model.encode(decision.context_text, "value", decision.input_ids))
    assert torch.equal(expected, model.encode(decision.context_text, "policy", decision.input_ids))
    warm = model.decision_values(decision)
    warm.sum().backward()
    gradient = model.value_head.output[-1].weight.grad.clone()
    model.zero_grad(set_to_none=True)
    node_text_cache(model)._cache.clear()
    cold = model.decision_values(decision)
    cold.sum().backward()
    assert torch.equal(warm, cold)
    assert torch.equal(gradient, model.value_head.output[-1].weight.grad)
    assert all(p.grad is None for p in model.policy_head.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    identity = node_text_cache(model).identity
    with torch.no_grad():
        next(p for name, p in model.backbone.named_parameters() if "lora_" not in name).add_(.001)
    assert node_text_cache(model).identity != identity
    assert not node_text_cache(model)._cache
    with pytest.raises(ValueError, match="encoder identity"):
        model.decision_values(decision)


def test_cache_failure_receipts_do_not_invent_a_free_forward_or_successful_head(tmp_path, monkeypatch):
    model = roles(tmp_path)
    state, plans = state_and_plans()
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    bridge._record(state, plans)  # node cache only; the global context is cold
    original = model._encode_frozen_text
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic frozen encoder failure")
    monkeypatch.setattr(model, "_encode_frozen_text", fail)
    with pytest.raises(RuntimeError, match="synthetic frozen encoder"):
        bridge.scorer(state, plans[0])
    row = bridge.physical_inferences[-1]
    assert row["status"] == "error" and row["encoder_forward_count"] == 1
    assert row["prompt_tokens"] > 0 and not row["encoder_cache_hit"] and row["head_forward_count"] == 0
    monkeypatch.setattr(model, "_encode_frozen_text", original)
    bridge.scorer(state, plans[0])
    monkeypatch.setattr(model.value_head, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic frozen encoder"):
        bridge.scorer(state, plans[0])
    row = bridge.physical_inferences[-1]
    assert row["status"] == "error" and row["head_forward_count"] == 1
    assert row["encoder_cache_hit"] and row["encoder_forward_count"] == row["prompt_tokens"] == 0


def test_exact_ids_and_tokenizer_identity_bound_global_cache_and_preserve_actor_modes(tmp_path):
    model = roles(tmp_path)
    model.train()
    modes = tuple(module.training for module in model.backbone.modules())
    seen = []
    handle = model.backbone.get_base_model().base_model.register_forward_pre_hook(
        lambda *args: seen.append(tuple(module.training for module in model.backbone.modules())))
    try:
        model.encode("read write", "value")
        assert model._last_global_encoding["encoder_forward_count"] == 1
        model.encode("read write", "policy")
        assert model._last_global_encoding["encoder_cache_hit"]
        model.encode("write read", "value")
        assert model._last_global_encoding["encoder_forward_count"] == 1
        identity = node_text_cache(model).identity
        model.tokenizer.add_tokens(["newly-registered-token"])
        assert node_text_cache(model).identity != identity
        model.encode("read write", "value")
        assert model._last_global_encoding["encoder_forward_count"] == 1
        assert len(seen) == 3 and not any(flag for forward in seen for flag in forward)
        assert tuple(module.training for module in model.backbone.modules()) == modes
    finally:
        handle.remove()


def test_observed_node_budget_failure_precedes_every_forward(tmp_path, monkeypatch):
    model = roles(tmp_path)
    model.config = replace(model.config, semantic_total_tokens=model.config.semantic_global_tokens + 1)
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    state, plans = observed_state("X" * 300)
    monkeypatch.setattr(model, "_encode_frozen_text", lambda *args, **kw: pytest.fail("must reject before physical inference"))
    with pytest.raises(ValueError, match="minimum"):
        bridge.semantic_forward_cap(state)
    with pytest.raises(ValueError, match="minimum"):
        bridge._record(state, plans)
    assert bridge.physical_inferences == []


def test_defaults_preserve_identity_new_contract_roundtrips_and_mismatch_is_rejected(tmp_path):
    old = TinyRoleModel(TinyModelConfig(architecture_version=3, option_dim=12))
    original = old.config
    identity = node_text_cache(old).identity
    legacy_config = asdict(original)
    legacy_config.pop("role_observation_version"); legacy_config.pop("frozen_global_cache")
    old.config = SimpleNamespace(**legacy_config)
    assert node_text_cache(old).identity == identity
    old.config = original
    model = roles(tmp_path)
    state, plans = state_and_plans()
    decision = VATSSearchBridge(model)._record(state, plans)
    model.save(tmp_path / "checkpoint")
    restored = HFRoleModel.load(tmp_path / "checkpoint", device="cpu").eval()
    assert restored.config.role_observation_version == "observed_results_v2" and restored.config.frozen_global_cache
    assert torch.equal(model.decision_values(decision), restored.decision_values(decision))
    restored.config = replace(restored.config, role_observation_version="legacy_v1")
    with pytest.raises(ValueError, match="encoder identity"):
        restored.decision_values(decision)
    with pytest.raises(ValueError, match="architecture_version=3"):
        HFModelConfig("unused-local", frozen_global_cache=True)
    with pytest.raises(ValueError, match="redistribution"):
        HFModelConfig("unused-local", architecture_version=3, role_observation_version="observed_results_v2")
