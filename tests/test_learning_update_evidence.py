"""Real CPU optimizer evidence: actor learning is not an optimizer step count."""
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from specrsi.learning import DecisionRecord, SearchTrajectory, TransitionRecord, TinyModelConfig, TinyRoleModel
from specrsi.learning.trainer import SpecRSITrainer, TrainingConfig, merge_learning_signals

torch.set_num_threads(2)


def model():
    return TinyRoleModel(TinyModelConfig(feature_dim=3, option_dim=2, hidden_dim=8, lora_rank=2))


def decision(net, kind="plan", selected=0):
    item = DecisionRecord(kind, ("a", "b"), selected, math.log(.5),
        state_features=(1., 0., 0.), option_features=((1., 0.), (0., 1.)))
    with torch.no_grad():
        from specrsi.learning.objectives import mixture_probabilities
        probs = mixture_probabilities(net.decision_logits(item))
    return replace(item, behavior_logprob=float(probs[selected].log()), behavior_probabilities=tuple(probs.tolist()))


def trajectory(decisions, seed, final=1.):
    return SearchTrajectory("g", "context", f"search-{seed}", seed,
        (TransitionRecord(0., final, 2.),), tuple(decisions), final_f=final, lineage_id="development")


@pytest.mark.parametrize("decay, expected", [(0., "no_parameter_change"), (.1, "optimizer_only_parameter_change")])
def test_zero_advantages_remain_legal_and_decay_is_not_actor_learning(decay, expected):
    net = model()
    d = decision(net)
    before = {name: p.detach().clone() for name, p in net.policy_head.named_parameters()}
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01, weight_decay=decay, value_weight=0.))
    result = trainer.gvo_epoch([[trajectory([d], 1), trajectory([d], 2)]])
    evidence = result["learning_signal"]
    assert trainer.steps == 1 and evidence["optimizer_steps"] == 1
    assert evidence["status"] == expected
    assert evidence["groups_with_zero_loo_advantage"] == 1
    assert not evidence["actor_roles_updated"] and not evidence["all_actor_roles_updated"]
    policy = evidence["roles"]["policy"]
    assert policy["gradient_steps"] == 1 and policy["nonzero_gradient_steps"] == 0
    assert policy["signal_parameter_change_steps"] == 0
    changed = any(not torch.equal(p, before[name]) for name, p in net.policy_head.named_parameters())
    assert changed == bool(decay) == bool(policy["parameter_change_steps"])
    assert result["first_hf_cuda_optimizer_step"] is None


def test_equal_return_groups_can_learn_only_critic_without_actor_claim():
    net = model()
    d = decision(net)
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01, weight_decay=0.))
    metrics = trainer.gvo_epoch([[trajectory([d], 1), trajectory([d], 2)]])
    signal = metrics["learning_signal"]
    assert signal["status"] == "critic_only_updated"
    assert signal["roles"]["value"]["signal_parameter_change_steps"] == 1
    assert not signal["roles"]["policy"]["nonzero_gradient_steps"]
    assert signal["group_dispersion"][0]["return_variance"] == 0.
    assert signal["joint_collection_updates"]["status"] == "critic_only_updated"


def test_real_policy_draft_and_critic_updates_are_separately_observed_and_saved(tmp_path):
    net = model()
    choices = [[decision(net, kind, selected) for kind in ("plan", "edit")] for selected in (0, 1)]
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01, weight_decay=0.))
    result = trainer.gvo_epoch([[trajectory(choices[0], 1, 2.), trajectory(choices[1], 2, 1.)]])
    signal = result["learning_signal"]
    assert signal["status"] == "actor_and_critic_updated"
    assert signal["actor_roles_updated"] == ["draft", "policy"]
    assert signal["groups_with_zero_loo_advantage"] == 0
    assert signal["group_dispersion"][0]["loo_nonzero_advantages"] == 2
    assert all(signal["roles"][role]["gradient_l2_max"] > 0 for role in ("draft", "policy", "value"))
    assert all(signal["roles"][role]["changed_elements"] > 0 for role in ("draft", "policy", "value"))
    assert all(not signal["roles"][role]["gradient_steps"] for role in ("verify", "surface", "other"))
    trainer.save(tmp_path)
    saved = json.loads((tmp_path / "training_evidence.json").read_text())
    assert saved["gvo_learning_signal"]["all_actor_roles_updated"]
    restored = SpecRSITrainer(TinyRoleModel.load(tmp_path), trainer.config)
    restored.restore_optimizer(tmp_path)
    assert restored.learning_history == trainer.learning_history
    assert restored.steps == trainer.steps
    (tmp_path / "training_evidence.json").unlink()
    legacy = SpecRSITrainer(TinyRoleModel.load(tmp_path), trainer.config)
    legacy.restore_optimizer(tmp_path)
    assert legacy.inherited_evidence_status == "legacy_checkpoint_without_role_update_evidence"
    assert not legacy.learning_history


def test_forced_critic_updates_do_not_count_as_ordinary_actor_learning():
    net = model()
    forced_decision = decision(net)
    # Ordinary complete searches can terminate without a stochastic decision.
    group = [trajectory([], 1), trajectory([], 2)]
    forced = replace(trajectory([forced_decision], 3, 2.), forced=True)
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01, weight_decay=0.))
    result = trainer.gvo_epoch([group], forced_trajectories=[forced])
    signal = result["learning_signal"]
    assert signal["status"] == "critic_only_updated"
    assert signal["joint_collection_updates"]["optimizer_steps"] == 1
    assert "ordinary_search_updates" not in signal and "forced_critic_updates" not in signal
    assert result["update_schedule"]["q_tuple_counts"] == {"ordinary": 0, "forced": 1, "total": 1}
    assert result["update_schedule"]["forced_actor_choices"] == 0
    assert not signal["all_actor_roles_updated"]
    assert len(trainer.learning_history) == 1  # Nested critic warm-up is not counted twice.
    assert merge_learning_signals([None])["status"] == "legacy_or_missing_evidence"


@pytest.mark.parametrize("gvo_kind", ["critic", "zero"])
def test_v3_pipeline_publishes_accurate_critic_only_or_zero_signal_manifest(tmp_path, monkeypatch, gvo_kind):
    import specrsi.learning.pipeline as pipeline
    net = model()
    item = decision(net)
    warmup = SpecRSITrainer(net, TrainingConfig(weight_decay=0.))
    warmup_metrics = warmup.warmup_epoch(plans=[item], trajectories=[trajectory([item], 1)])
    empty = merge_learning_signals([{"schema": "role_update_evidence_v1", "optimizer_steps": 0,
        "roles": {role: {**row, **{key: 0 for key in row}} for role, row in warmup_metrics["learning_signal"]["roles"].items()}}])
    gvo_trainer = SpecRSITrainer(model(), TrainingConfig(weight_decay=0.))
    critic = gvo_trainer.gvo_epoch([[trajectory([], 1), trajectory([], 2)]],
        forced_trajectories=[replace(trajectory([item], 3), forced=True)])
    config = {"stage": "two_stage", "collector": "actual:collect",
        "model": {"backend": "tiny", "architecture_version": 3},
        "collection": {"algorithm": "vats_pdf"}, "training": {"seed": 1}}

    def run(command, **kwargs):
        cfg = yaml.safe_load(Path(command[-1]).read_text())
        directory = Path(cfg["output"])
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir(parents=True)
        initial = 0 if cfg["stage"] == "warmup" else warmup.steps
        row = dict(warmup_metrics if cfg["stage"] == "warmup" else critic)
        updates = warmup.steps if cfg["stage"] == "warmup" else (1 if gvo_kind == "critic" else 0)
        if cfg["stage"] == "gvo" and gvo_kind == "zero":
            row["learning_signal"] = empty
        row["steps"] = initial + updates
        (checkpoint / "training_evidence.json").write_text(json.dumps({"schema": "role_training_evidence_v1"}))
        (directory / "metrics.json").write_text(json.dumps({"source_changed_during_run": False,
            "source_hashes": {"src": "frozen"}, "initial_optimizer_steps": initial, "optimizer_updates": updates,
            "history": [row], "checkpoint": str(checkpoint)}))

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    output = tmp_path / "trial"
    pipeline.run_two_stage(config, output, root=tmp_path)
    summary = json.loads((output / "metrics.json").read_text())
    assert summary["complete"] and not summary["all_roles_updated_in_gvo"]
    assert summary["gvo_learning_signal"]["status"] == ("critic_only_updated" if gvo_kind == "critic" else "no_parameter_change")
    assert summary["stages"][-1]["training_evidence_sha256"]
    assert (output / "checkpoint").is_dir()
