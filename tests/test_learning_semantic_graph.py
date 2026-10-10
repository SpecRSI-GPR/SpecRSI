"""v3 must learn different work values without labels entering node semantics."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from specrsi.core import AcquisitionCaps, BudgetLedger, Candidate, EvaluationScope, Ledger, PureIntegerExecutor, ResultAcquirer, Task
from specrsi.core.vats import VATSPlan, VATSState
from specrsi.learning import SpecRSITrainer, TrainingConfig
from specrsi.learning.frozen_text import node_text_cache
from specrsi.learning.heads import TinyModelConfig, TinyRoleModel
from specrsi.learning.paper_bridge import VATSSearchBridge, collect_reference_vats
from specrsi.learning.records import PairwiseValueExample, _decision
from specrsi.planning.completion import VerificationJob


def state_and_plans():
    tasks = (Task("read", {"route": "read", "x": 2}, 918273645123,
                  {"private": "PRIVATE_METADATA"}),
             Task("write", {"route": "write", "x": 2}, 918273645124))
    scope = EvaluationScope("semantic-fixture", tasks, "pure-int-v1", "integer-equality-v1",
                            ("answer", "primitives"), split="dev")
    root = Candidate("base", {"routes": {"read": ("x",), "write": ("x",)}},
                     {"contract": "pure-int-v1"}, "pure_integer", True)
    ledger = Ledger()
    acquirer = ResultAcquirer(ledger, BudgetLedger(100), PureIntegerExecutor())
    for task in tasks:
        acquirer.acquire(root, task, scope, (), AcquisitionCaps(10))
    child = Candidate("child", root.artifacts, root.runtime, "pure_integer", True, parent_id=root.uid)
    ledger.register(child, scope)
    state = VATSState(scope, root, {root.uid: root, child.uid: child}, ledger, BudgetLedger(1000),
                      remaining_attempts=10, initial_attempts=10, workers=1)
    plans = tuple(VATSPlan("verify", jobs=(VerificationJob("check-" + str(i), ((child.uid, task.uid),), 10),),
                          work_cap=10, attempt_cap=1) for i, task in enumerate(tasks))
    return state, plans


def tiny(version=3, seed=0):
    return TinyRoleModel(TinyModelConfig(architecture_version=version, option_dim=12,
                                       hidden_dim=16, seed=seed)).eval()


def record(model):
    state, plans = state_and_plans()
    bridge = VATSSearchBridge(model)
    return bridge, state, plans, bridge._record(state, plans)


def pair(decision):
    left = replace(decision, selected_index=0, deterministic=True, behavior_logprob=None)
    right = replace(left, selected_index=1)
    return PairwiseValueExample(left, right, (1.,), (0.,), "same-paid-state", "frozen-policy",
                                (10,), (11,), 1, 1)


def test_v2_symmetry_is_preserved_but_v3_has_trainable_same_state_job_contrast():
    old = tiny(2)
    _, _, _, before = record(old)
    assert before.option_features[0] == before.option_features[1]
    assert old.decision_values(before)[0] == old.decision_values(before)[1]
    assert old.decision_logits(before)[0] == old.decision_logits(before)[1]
    model = tiny()
    _, _, _, decision = record(model)
    assert decision.option_features[0] == decision.option_features[1]
    value = model.decision_values(decision)
    (value[0] - value[1] - 1.).square().backward()
    assert model.value_head.node_text_projection.weight.grad.abs().max() > 1e-5
    assert all(parameter.grad is None for parameter in model.policy_head.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    model.zero_grad(set_to_none=True)
    trainer = SpecRSITrainer(model, TrainingConfig(learning_rate=.01, pair_weight=1., weight_decay=0.))
    example = pair(decision)
    initial = float(trainer._pair_loss(example))
    for _ in range(20):
        trainer.warmup_epoch(pairs=[example])
    assert float(trainer._pair_loss(example)) < initial * .25


def test_semantic_graph_permutation_and_record_roundtrip_preserve_predictions():
    model = tiny()
    _, _, _, decision = record(model)
    count = len(decision.graph_nodes)
    permuted = replace(decision, graph_nodes=tuple(reversed(decision.graph_nodes)),
        graph_node_texts=tuple(reversed(decision.graph_node_texts)),
        graph_node_text_features=tuple(reversed(decision.graph_node_text_features)),
        graph_edges=tuple((count - 1 - a, count - 1 - b, rel) for a, b, rel in decision.graph_edges),
        option_node_masks=tuple(tuple(reversed(mask)) for mask in decision.option_node_masks))
    restored = _decision(json.loads(json.dumps(asdict(decision))))
    assert restored == decision
    for method in (model.decision_values, model.decision_logits):
        assert torch.allclose(method(decision), method(permuted), atol=1e-6)
        assert torch.equal(method(decision), method(restored))


def test_semantics_use_only_visible_artifacts_inputs_and_cache_misses_are_physical():
    model = tiny()
    bridge, state, plans, decision = record(model)
    payload = json.dumps(asdict(decision))
    assert "918273645123" not in payload and "918273645124" not in payload and "PRIVATE_METADATA" not in payload
    assert any("read" in text for text in decision.graph_node_texts)
    assert any("write" in text for text in decision.graph_node_texts)
    assert len(bridge.physical_inferences) == len(set(decision.graph_node_texts))
    count = len(bridge.physical_inferences)
    again = bridge._record(state, plans[::-1])
    assert again.graph_node_text_features == decision.graph_node_text_features
    assert len(bridge.physical_inferences) == count
    # Restored paid-state bridges share only this identical frozen model cache.
    restored_bridge = VATSSearchBridge(model)
    restored = restored_bridge._record(state, plans)
    assert restored.graph_node_text_features == decision.graph_node_text_features
    assert restored_bridge.physical_inferences == []


def test_frozen_weight_change_invalidates_cache_and_old_replay_is_rejected():
    model = tiny()
    bridge, state, plans, decision = record(model)
    identity = node_text_cache(model).identity
    # Actor updates must not invalidate a frozen text encoder.
    with torch.no_grad():
        model.policy_head.output[-1].bias.add_(1.)
    assert node_text_cache(model).identity == identity
    with torch.no_grad():
        model.backbone.weight.add_(.1)
    assert node_text_cache(model).identity != identity
    with pytest.raises(ValueError, match="encoder identity"):
        model.decision_values(decision)
    fresh = bridge._record(state, plans)
    assert fresh.graph_node_text_features != decision.graph_node_text_features
    with pytest.raises(ValueError, match="encoder identity"):
        tiny(seed=5).decision_logits(fresh)


def test_checkpoint_roundtrip_requires_explicit_version_and_encoder_identity(tmp_path):
    model = tiny()
    _, _, _, decision = record(model)
    model.save(tmp_path / "v3")
    restored = TinyRoleModel.load(tmp_path / "v3")
    assert torch.equal(model.decision_values(decision), restored.decision_values(decision))
    assert torch.equal(model.decision_logits(decision), restored.decision_logits(decision))
    old = tiny(2)
    _, _, _, old_decision = record(old)
    old.save(tmp_path / "v2")
    old_restored = TinyRoleModel.load(tmp_path / "v2")
    assert torch.equal(old.decision_values(old_decision), old_restored.decision_values(old_decision))
    with pytest.raises(ValueError, match="v2 roles"):
        old_restored.decision_values(decision)
    with pytest.raises(ValueError, match="v3 replay"):
        restored.decision_values(old_decision)
    manifest = tmp_path / "v3" / "model_config.json"
    raw = json.loads(manifest.read_text())
    raw["frozen_text_snapshot"] = "wrong-encoder"
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="encoder identity"):
        TinyRoleModel.load(tmp_path / "v3")


def test_same_state_pair_rejects_changed_semantics_and_incomplete_vectors():
    _, _, _, decision = record(tiny())
    example = pair(decision)
    with pytest.raises(ValueError, match="same recorded paid state"):
        replace(example, right=replace(example.right, graph_text_snapshot="other-encoder"))
    with pytest.raises(ValueError, match="aligned texts"):
        replace(decision, graph_node_text_features=decision.graph_node_text_features[:-1])


def test_actual_reference_vats_forced_collection_replays_v3_records():
    model = tiny()
    collected = collect_reference_vats(model, 0, {"collection": {"contexts": 1, "group_size": 2,
        "budget": 200, "max_actions": 6, "forced_repetitions": 2, "vats": {"max_queries": 4}}})
    assert collected["pairs"] and len(collected["forced_trajectories"]) == 4
    assert all(d.graph_text_snapshot for group in collected["groups"] for t in group
               for d in t.decisions if d.kind == "plan")
    trainer = SpecRSITrainer(model, TrainingConfig(pair_weight=.1))
    metrics = trainer.gvo_epoch(collected["groups"], forced_trajectories=collected["forced_trajectories"],
                                pairs=collected["pairs"])
    assert metrics["forced_searches"] == 4 and metrics["pair_examples"] == 1


def hf_roles(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from specrsi.learning.hf import HFModelConfig, HFRoleModel
    tokens = ["[UNK]", "[PAD]", "[EOS]", "read", "write", "x", "2", "task", "inputs", "candidate", "artifacts"]
    raw_tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, unk_token="[UNK]",
                                        pad_token="[PAD]", eos_token="[EOS]")
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, head_dim=8, eos_token_id=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        base = Qwen3ForCausalLM(config)
    location = tmp_path / "base"
    base.save_pretrained(location)
    tokenizer.save_pretrained(location)
    return HFRoleModel(HFModelConfig(model_path=str(location), architecture_version=3, option_dim=12,
        device="cpu", dtype="float32", lora_rank=2, lora_alpha=4, max_length=512, head_hidden_dim=16,
        gradient_checkpointing=False), base_model=base, tokenizer=tokenizer).eval()


def test_hf_v3_real_frozen_features_gradient_metering_and_save_load(tmp_path):
    from specrsi.learning.hf import HFRoleModel
    model = hf_roles(tmp_path)
    state, plans = state_and_plans()
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    first = bridge.scorer(state, plans[0])
    calls = list(bridge.physical_inferences)
    assert any(row["role"] == "frozen_node_text" for row in calls)
    assert first.actual_cost == sum(row["prompt_tokens"] for row in calls)
    assert first.actual_cost <= model.config.max_length
    count = len(calls)
    warm = bridge.scorer(state, plans[1])
    assert [row["role"] for row in bridge.physical_inferences[count:]] == ["scorer"]
    assert warm.actual_cost == first.actual_cost
    assert sum(row["prompt_tokens"] for row in bridge.physical_inferences[count:]) < warm.actual_cost
    assert bridge.physical_inferences[-1]["search_budget_contract"] == "fixed_record_context_plus_unique_node_input_quota"
    decision = bridge._record(state, plans)
    values = model.decision_values(decision)
    (values[0] - values[1] - 1.).square().backward()
    assert model.value_head.node_text_projection.weight.grad.abs().max() > 1e-6
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.policy_head.parameters())
    model.save(tmp_path / "checkpoint")
    restored = HFRoleModel.load(tmp_path / "checkpoint", device="cpu").eval()
    assert torch.equal(model.decision_values(decision), restored.decision_values(decision))
    assert torch.equal(model.decision_logits(decision), restored.decision_logits(decision))
    # A v2 head file cannot masquerade as a v3 checkpoint.
    heads_path = tmp_path / "checkpoint" / "heads.pt"
    heads = torch.load(heads_path, weights_only=True)
    heads.pop("value_head.node_text_projection.weight")
    torch.save(heads, heads_path)
    with pytest.raises(ValueError, match="exact semantic-head"):
        HFRoleModel.load(tmp_path / "checkpoint", device="cpu")
    manifest = tmp_path / "checkpoint" / "model_config.json"
    raw = json.loads(manifest.read_text())
    raw["architecture_version"] = 2
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="v2 checkpoint cannot"):
        HFRoleModel.load(tmp_path / "checkpoint", device="cpu")


def test_hf_semantic_forward_failure_retains_attempted_input_cost(tmp_path, monkeypatch):
    model = hf_roles(tmp_path)
    state, plans = state_and_plans()
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic encoder failure")
    monkeypatch.setattr(model, "encode", fail)
    with pytest.raises(RuntimeError, match="synthetic encoder"):
        bridge.scorer(state, plans[0])
    assert len(bridge.physical_inferences) == 1
    receipt = bridge.physical_inferences[0]
    assert receipt["status"] == "error" and receipt["prompt_tokens"] > 0
    assert receipt["role"] == "frozen_node_text"


def test_hf_duplicate_job_text_uses_unique_text_budget_and_is_billed_once(tmp_path):
    model = hf_roles(tmp_path)
    state, plans = state_and_plans()
    root = state.incumbent
    for index in range(6):
        child = Candidate("sibling-" + str(index), root.artifacts, root.runtime,
                          root.semantic_kind, root.deterministic, parent_id=root.uid)
        state.candidates[child.uid] = child
        state.ledger.register(child, state.scope)
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    assert (model.config.max_length - len(model.tokenizer.encode(bridge.context_text(state)))) // 24 < 16
    estimate = bridge.scorer(state, plans[0])
    decision = bridge._record(state, plans)
    assert len(decision.graph_nodes) == 24
    assert len(set(decision.graph_node_texts)) == 3
    node_calls = [row for row in bridge.physical_inferences if row["role"] == "frozen_node_text"]
    assert len(node_calls) == 3
    assert estimate.actual_cost == sum(row["prompt_tokens"] for row in bridge.physical_inferences)
    assert estimate.actual_cost <= model.config.max_length
    assert all(row["prompt_tokens"] == len(row["input_token_ids"]) for row in node_calls)


def test_hf_tokenizer_normalizer_mutation_rejects_old_replay(tmp_path):
    from tokenizers.normalizers import Lowercase
    model = hf_roles(tmp_path)
    _, _, _, decision = record(model)
    model.tokenizer.backend_tokenizer.normalizer = Lowercase()
    with pytest.raises(ValueError, match="encoder identity"):
        model.decision_values(decision)


def test_hf_cached_identity_never_merges_full_vocabulary(tmp_path, monkeypatch):
    from transformers import PreTrainedTokenizerFast
    model = hf_roles(tmp_path)
    cache = node_text_cache(model)
    identity = cache.identity

    def forbidden_length(_):
        raise AssertionError("Cached identity must not merge the full vocabulary")

    monkeypatch.setattr(PreTrainedTokenizerFast, "__len__", forbidden_length)
    assert cache.identity == identity
    assert cache.identity == identity
    # Explicit full validation still hashes the complete serialized tokenizer;
    # its fingerprint is identical without the expensive merged-vocab length.
    assert cache.validate_identity() == identity


def test_hf_added_token_change_invalidates_cached_observation(tmp_path):
    model = hf_roles(tmp_path)
    _, _, _, decision = record(model)
    cache = node_text_cache(model)
    previous = cache.identity
    assert cache._cache
    model.tokenizer.add_tokens(["NEW_ADDED_TOKEN"])
    assert cache.identity != previous
    assert not cache._cache
    with pytest.raises(ValueError, match="encoder identity"):
        model.decision_values(decision)


def test_hf_same_size_added_token_metadata_invalidates_cached_identity(tmp_path):
    from tokenizers import AddedToken
    model = hf_roles(tmp_path)
    model.tokenizer.add_tokens([AddedToken("EXTRA_TOKEN", lstrip=False, normalized=False)])
    cache = node_text_cache(model)
    previous, size = cache.identity, len(model.tokenizer)
    model.tokenizer.add_tokens([AddedToken("EXTRA_TOKEN", lstrip=True, normalized=False)])
    assert len(model.tokenizer) == size
    changed = cache.identity
    assert changed != previous
    assert cache.validate_identity() == changed


def test_hf_base_vocabulary_growth_invalidates_without_full_boundary(tmp_path):
    from tokenizers.models import WordLevel
    model = hf_roles(tmp_path)
    cache = node_text_cache(model)
    previous = cache.identity
    vocabulary = model.tokenizer.backend_tokenizer.get_vocab(with_added_tokens=False)
    vocabulary["NEW_INTERNAL_TOKEN"] = len(vocabulary)
    model.tokenizer.backend_tokenizer.model = WordLevel(vocabulary, unk_token="[UNK]")
    assert cache.identity != previous


@pytest.mark.parametrize("boundary", ["search", "warmup"])
def test_hf_same_size_internal_vocab_mutation_is_detected_at_search_boundary(tmp_path, boundary):
    from tokenizers.models import WordLevel
    model = hf_roles(tmp_path)
    _, _, _, decision = record(model)
    vocab = model.tokenizer.get_vocab()
    vocab["read"], vocab["write"] = vocab["write"], vocab["read"]
    model.tokenizer.backend_tokenizer.model = WordLevel(vocab, unk_token="[UNK]")
    # Tokenizers exposes no vocabulary mutation counter. Full validation is
    # explicitly performed at search/checkpoint/training boundaries.
    if boundary == "search":
        VATSSearchBridge(model)
    else:
        trainer = SpecRSITrainer(model, TrainingConfig(pair_weight=1.))
        with pytest.raises(ValueError, match="encoder identity"):
            trainer.warmup_epoch(pairs=[pair(decision)])
    with pytest.raises(ValueError, match="encoder identity"):
        model.decision_logits(decision)


def test_hf_saved_tokenizer_restores_independently_of_base_directory(tmp_path):
    from specrsi.learning.hf import HFRoleModel
    model = hf_roles(tmp_path)
    _, _, _, decision = record(model)
    checkpoint = tmp_path / "checkpoint"
    model.save(checkpoint)
    model.tokenizer.add_tokens(["NEW_BASE_DIRECTORY_TOKEN"])
    model.tokenizer.save_pretrained(tmp_path / "base")
    restored = HFRoleModel.load(checkpoint, device="cpu").eval()
    assert "NEW_BASE_DIRECTORY_TOKEN" not in restored.tokenizer.get_vocab()
    assert restored.config.architecture_version == 3
    assert torch.isfinite(restored.decision_values(decision)).all()
    # Altering the saved tokenizer itself fails its recorded identity.
    model.tokenizer.save_pretrained(checkpoint / "tokenizer")
    with pytest.raises(ValueError, match="encoder identity"):
        HFRoleModel.load(checkpoint, device="cpu")


def redistributed(model, **overrides):
    model.config = replace(model.config, **{
        "semantic_budget_mode": "redistribute", "semantic_global_tokens": 96,
        "semantic_node_min_tokens": 16, "semantic_node_max_tokens": 256,
        "semantic_total_tokens": 640, **overrides})
    return model


def long_visible_state():
    state, plans = state_and_plans()
    tasks = (Task("long-read", {"prompt": "read " * 600 + "write"}, "PRIVATE_REFERENCE_READ"),
             Task("long-write", {"prompt": "write " * 600 + "read"}, "PRIVATE_REFERENCE_WRITE"))
    state.scope = EvaluationScope("development-semantics", tasks, "pure-int-v1", "equality", split="dev")
    return state, plans


def test_water_filling_reuses_short_node_allowances_and_respects_minimum_and_maximum():
    from specrsi.learning.frozen_text import allocate_node_tokens
    assert allocate_node_tokens((20, 300, 300), available=300, minimum=16, maximum=200) == (20, 140, 140)
    assert allocate_node_tokens((2, 300, 300), available=203, minimum=16, maximum=100) == (2, 100, 100)
    assert allocate_node_tokens((300, 300, 300), available=50, minimum=16, maximum=100) == (17, 17, 16)
    with pytest.raises(ValueError, match="every unique node"):
        allocate_node_tokens((100, 100), available=31, minimum=16, maximum=100)


def test_semantic_budget_requires_explicit_profile_and_registered_resource_fields():
    from specrsi.learning.frozen_text import semantic_input_cap
    from specrsi.learning.hf import HFModelConfig
    assert semantic_input_cap({"architecture_version": 3, "max_length": 512}) == 512
    with pytest.raises(ValueError, match="explicit"):
        HFModelConfig("unloaded", semantic_total_tokens=4096)
    with pytest.raises(ValueError, match="positive integer"):
        HFModelConfig("unloaded", architecture_version=3, semantic_budget_mode="redistribute")
    config = HFModelConfig("unloaded", architecture_version=3, max_length=512,
        semantic_budget_mode="redistribute", semantic_global_tokens=96, semantic_node_min_tokens=16,
        semantic_node_max_tokens=256, semantic_total_tokens=640)
    assert semantic_input_cap(config) == semantic_input_cap(asdict(config)) == 640
    with pytest.raises(ValueError, match="max_length"):
        replace(config, semantic_node_max_tokens=513)


def test_semantic_excerpt_counts_special_tokens_and_preserves_head_and_tail():
    from specrsi.learning.frozen_text import bounded_semantic_text

    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=True):
            return ([1] if add_special_tokens else []) + [ord(char) + 2 for char in text]

        def decode(self, ids, **kwargs):
            return "".join(chr(token - 2) for token in ids)

    tokenizer = CharacterTokenizer()
    clipped, ids = bounded_semantic_text(tokenizer, "HEAD" + "x" * 100 + "TAIL", 16)
    assert clipped.startswith("HEAD") and clipped.endswith("TAIL") and " … " in clipped
    assert ids == tuple(tokenizer.encode(clipped)) and len(ids) <= 16
    prefix, ids = bounded_semantic_text(tokenizer, "HEAD" + "x" * 100 + "TAIL", 16, strategy="prefix")
    assert prefix.startswith("HEAD") and "TAIL" not in prefix and len(ids) <= 16


def test_redistributed_hf_context_records_exact_ids_caps_cache_fees_and_replay(tmp_path):
    from specrsi.learning.frozen_text import semantic_input_cap
    from specrsi.learning.hf import HFRoleModel
    model = redistributed(hf_roles(tmp_path))
    state, plans = long_visible_state()
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    first = bridge.scorer(state, plans[0])
    calls = list(bridge.physical_inferences)
    decision = bridge._record(state, plans)
    assert len(decision.input_ids) <= 96
    assert len(decision.input_ids) == len(model.tokenizer.encode(decision.context_text))
    assert all(len(ids) <= 256 for ids in decision.graph_node_input_ids)
    expected = len(decision.input_ids) + sum(len(ids) for ids in set(decision.graph_node_input_ids))
    assert first.actual_cost == expected == sum(row["prompt_tokens"] for row in calls)
    assert model.config.max_length < expected <= semantic_input_cap(model.config)
    assert all(tuple(model.tokenizer.encode(text)) == ids for text, ids in
               zip(decision.graph_node_texts, decision.graph_node_input_ids))
    assert "PRIVATE_REFERENCE" not in decision.context_text + "".join(decision.graph_node_texts)
    warm = bridge.scorer(state, plans[1])
    assert warm.actual_cost == first.actual_cost
    assert [row["role"] for row in bridge.physical_inferences[len(calls):]] == ["scorer"]
    assert bridge.physical_inferences[-1]["prompt_tokens"] == len(decision.input_ids)
    restored_record = _decision(json.loads(json.dumps(asdict(decision))))
    assert restored_record == decision
    model.save(tmp_path / "redistributed")
    restored = HFRoleModel.load(tmp_path / "redistributed", device="cpu").eval()
    assert restored.config.semantic_total_tokens == 640
    assert torch.equal(model.decision_logits(decision), restored.decision_logits(restored_record))
    restored.config = replace(restored.config, semantic_global_tokens=95)
    with pytest.raises(ValueError, match="encoder identity"):
        restored.decision_values(decision)


def test_legacy_checkpoint_without_new_budget_fields_keeps_exact_encoder_identity(tmp_path):
    from specrsi.learning.frozen_text import SEMANTIC_BUDGET_FIELDS
    from specrsi.learning.hf import HFRoleModel
    model = hf_roles(tmp_path)
    _, _, _, decision = record(model)
    checkpoint = tmp_path / "legacy"
    model.save(checkpoint)
    path = checkpoint / "model_config.json"
    raw = json.loads(path.read_text())
    for field in SEMANTIC_BUDGET_FIELDS:
        raw.pop(field)
    path.write_text(json.dumps(raw))
    restored = HFRoleModel.load(checkpoint, device="cpu").eval()
    assert node_text_cache(restored).identity == decision.graph_text_snapshot
    assert torch.equal(model.decision_values(decision), restored.decision_values(decision))


@pytest.mark.parametrize("role", ["scorer", "policy"])
def test_redistributed_final_forward_failure_keeps_node_and_attempted_global_receipts(tmp_path, monkeypatch, role):
    model = redistributed(hf_roles(tmp_path))
    state, plans = long_visible_state()
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    original = model.decision_values if role == "scorer" else model.decision_logits

    def fail_after_forward(*args):
        original(*args)
        raise RuntimeError("failed after actual global forward")

    monkeypatch.setattr(model, "decision_values" if role == "scorer" else "decision_logits", fail_after_forward)
    with pytest.raises(RuntimeError, match="actual global forward"):
        bridge.scorer(state, plans[0]) if role == "scorer" else bridge.policy_menu(state, plans)
    rows = bridge.physical_inferences
    assert len(rows) == 4 and rows[-1]["role"] == role and rows[-1]["status"] == "error"
    assert rows[-1]["prompt_tokens"] == len(rows[-1]["input_token_ids"])
    assert sum(row["prompt_tokens"] for row in rows) <= 640


def test_infeasible_node_minima_fail_before_any_forward(tmp_path):
    model = redistributed(hf_roles(tmp_path), semantic_global_tokens=64,
                          semantic_node_min_tokens=32, semantic_total_tokens=100)
    state, plans = long_visible_state()
    bridge = VATSSearchBridge(model, cost_unit="tokens")
    with pytest.raises(ValueError, match="every unique node"):
        bridge.scorer(state, plans[0])
    assert not bridge.physical_inferences
