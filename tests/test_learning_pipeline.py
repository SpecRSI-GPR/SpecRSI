from dataclasses import replace
import math

import pytest
import torch

torch.set_num_threads(2)

from specrsi.learning.pipeline import stage_configs
from specrsi.learning.records import DecisionRecord, EditExample, validate_group


def test_two_stage_freezes_stage_sources_and_requires_refreshed_gvo(tmp_path):
    config = {"stage": "two_stage", "collector": "actual:collect", "model": {"backend": "hf"},
        "collection": {"algorithm": "vats_pdf", "forced_repetitions": 2, "exploration": .1},
        "training": {"seed": 4}, "warmup": {"epochs": 2}, "gvo": {"epochs": 3}}
    warmup, gvo = stage_configs(config, tmp_path)
    assert warmup["collection"]["demonstration_policy"] == "complete_then_draft"
    assert warmup["collection"]["forced_repetitions"] == warmup["collection"]["exploration"] == 0
    assert gvo["collection"]["forced_repetitions"] == 2
    assert gvo["collection"]["exploration"] == .1
    assert "demonstration_policy" not in gvo["collection"]
    assert gvo["resume"] == str(tmp_path / "warmup/checkpoint")
    assert config["collection"]["exploration"] == .1
    assert [warmup["epochs"], gvo["epochs"]] == [2, 3]
    with pytest.raises(ValueError, match="freshly collect"):
        stage_configs({**config, "collector": None, "data": "actual.jsonl"}, tmp_path)
    with pytest.raises(ValueError, match="menu cap"):
        stage_configs({**config, "gvo": {"collection": {"vats": {"max_menu_options": 2}}}}, tmp_path)
    with pytest.raises(ValueError, match="quality rule"):
        stage_configs({**config, "warmup": {"collection": {
            "demonstration_quality": {"edit_rule": "forecast_gain"}}}}, tmp_path)


def test_edit_demonstration_rejects_fabricated_prefix_or_missing_token():
    decision = DecisionRecord("edit_token", ("3", "4"), 0, math.log(.5), input_ids=(1, 2))
    edit = EditExample("root", (1, 2, 3), (-100, -100, 3), True, token_decisions=(decision,))
    assert edit.token_decisions == (decision,)
    with pytest.raises(ValueError, match="actual prefix"):
        replace(edit, token_decisions=(replace(decision, input_ids=(1, 5)),))
    with pytest.raises(ValueError, match="every supervised token"):
        replace(edit, input_ids=(1, 2, 3, 4), labels=(-100, -100, 3, 4))
    with pytest.raises(ValueError, match="actual prefix"):
        replace(edit, input_ids=(1, 2, 9))
    with pytest.raises(ValueError, match="singleton"):
        replace(edit, token_decisions=(replace(decision, deterministic=True, behavior_logprob=None),))


def test_lineage_partition_persists_across_epochs_stages_and_protects_audit(tmp_path):
    from specrsi.learning.splits import LineageSplitRegistry
    path = tmp_path / "splits.json"
    first = LineageSplitRegistry(path, validation_fraction=.5, seed=4)
    records = [{"lineage_id": str(i)} for i in range(20)] + [{"lineage_id": "protected", "audit_only": True}]
    train, validation, audit = first.split_records(records)
    assert train and validation and len(audit) == 1
    second = LineageSplitRegistry(path, validation_fraction=.5, seed=4)
    repeated = second.split_records(list(reversed(records[::2])))
    for subset, target in zip(repeated, ("train", "validation", "audit")):
        assert all(first.assignments[row["lineage_id"]] == target for row in subset)
    assert second.assign("protected") == "audit"
    with pytest.raises(ValueError, match="previously trained"):
        second.assign(train[0]["lineage_id"], audit_only=True)
    with pytest.raises(ValueError, match="cannot change"):
        LineageSplitRegistry(path, validation_fraction=.2, seed=4)


def test_shared_registry_excludes_protected_roots_from_gvo_forced_and_pairs(tmp_path):
    from types import SimpleNamespace
    from specrsi.learning.splits import LineageSplitRegistry
    registry = LineageSplitRegistry(tmp_path / "splits.json", validation_fraction=0.)
    registry.split_records([{"lineage_id": "protected", "audit_only": True}])
    safe = SimpleNamespace(lineage_id="allowed", audit_only=False)
    protected = SimpleNamespace(lineage_id="protected", audit_only=False)
    collection = {"groups": [[safe, safe], [protected, protected]],
        "forced_trajectories": [safe, protected], "pairs": [safe, protected]}
    filtered, heldout = registry.split_collection(collection)
    assert filtered["groups"] == [[safe, safe]]
    assert filtered["forced_trajectories"] == filtered["pairs"] == [safe]
    assert not heldout


def test_collected_demonstrations_cover_only_actual_actions_and_are_not_gvo():
    from specrsi.learning.heads import TinyRoleModel, TinyModelConfig
    from specrsi.learning.paper_bridge import collect_reference_vats
    from specrsi.learning.demonstrations import warmup_records
    from specrsi.learning.trainer import SpecRSITrainer, TrainingConfig
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).resolve().parents[1] / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = TinyRoleModel(TinyModelConfig(option_dim=12, feature_dim=8, hidden_dim=16))
    collection = collect_reference_vats(model, 0, {"stage": "warmup", "collection": {
        "algorithm": "vats_pdf", "contexts": 1, "group_size": 2, "budget": 500,
        "max_candidates": 2, "max_actions": 5, "exploration": 0,
        "demonstration_policy": "complete_then_draft", "vats": {"max_queries": 4}}})
    assert all(t.demonstration for t in collection["groups"][0])
    with pytest.raises(ValueError, match="Demonstration"):
        validate_group(collection["groups"][0])
    records = warmup_records(collection)
    parsed = module.parse_records(records)
    assert parsed["plans"] and parsed["trajectories"]
    assert all(d.deterministic for d in parsed["plans"] if d.kind == "plan")
    metrics = SpecRSITrainer(model, TrainingConfig(batch_size=8)).warmup_epoch(**parsed)
    assert metrics["training_examples"]["plan"] > 0
    assert metrics["training_examples"]["value"] > 0
    assert math.isfinite(metrics["plan_loss"])


def test_hf_sft_uses_logged_supports_and_sums_actual_token_likelihoods():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from specrsi.learning.hf import HFModelConfig, HFRoleModel
    from specrsi.learning.objectives import mixture_probabilities
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
    model = HFRoleModel(HFModelConfig(model_path="__test__", device="cpu", dtype="float32",
        lora_rank=2, lora_alpha=4, option_dim=12, head_hidden_dim=8, gradient_checkpointing=False), base_model=base)
    model.eval()
    decisions = (DecisionRecord("edit_token", ("3", "4"), 0, math.log(.5), input_ids=(1, 2),
        temperature=.7, exploration=.2),
        DecisionRecord("edit_token", ("4",), 0, None, input_ids=(1, 2, 3), deterministic=True))
    edit = EditExample("root", (1, 2, 3, 4), (-100, -100, 3, 4), True, token_decisions=decisions)
    expected = -mixture_probabilities(model.decision_logits(decisions[0]), .7, .2)[0].log()
    actual = model.edit_nll(edit)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert any(p.grad is not None for n, p in model.backbone.named_parameters() if ".draft." in n)
    assert all(p.grad is None for p in model.value_head.parameters())


def test_two_stage_cannot_publish_gvo_with_only_inherited_sft_steps(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import yaml
    import specrsi.learning.pipeline as pipeline
    config = {'stage': 'two_stage', 'collector': 'actual:collect', 'model': {'backend': 'tiny'},
              'collection': {'algorithm': 'vats_pdf'}, 'training': {'seed': 1}}

    def run(command, **kwargs):
        cfg = yaml.safe_load(Path(command[-1]).read_text())
        stage, directory = cfg['stage'], Path(cfg['output'])
        (directory / 'checkpoint').mkdir(parents=True)
        metrics = {'source_changed_during_run': False, 'source_hashes': {'src': 'frozen'},
                   'initial_optimizer_steps': 0 if stage == 'warmup' else 3,
                   'optimizer_updates': 3 if stage == 'warmup' else 0,
                   'history': [{'steps': 3, 'training_examples': {'plan': 2, 'value': 2}}],
                   'checkpoint': str(directory / 'checkpoint')}
        (directory / 'metrics.json').write_text(json.dumps(metrics))

    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    output = tmp_path / 'trial'
    with pytest.raises(RuntimeError, match='new optimizer updates'):
        pipeline.run_two_stage(config, output, root=tmp_path)
    assert not (output / 'checkpoint').exists()
    assert json.loads((output / 'failure.json').read_text())['completed_stages'][0]['stage'] == 'warmup'
