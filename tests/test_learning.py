from dataclasses import replace
import math

import pytest
import torch

torch.set_num_threads(2)

from specrsi.learning import (CheckerAudit, DecisionRecord, SearchTrajectory, TransitionRecord, TinyModelConfig,
    TinyRoleModel, SpecRSITrainer, TrainingConfig, collect_independent_group, split_by_lineage)
from specrsi.learning.objectives import clipped_surrogate, group_advantages, mixture_probabilities
from specrsi.learning.records import trajectory_from_dict, validate_group


def decision(kind="outer", **kwargs):
    return DecisionRecord(kind, ("a", "b"), 0, math.log(.5), state_features=(1., 0., 0.),
                          option_features=((1., 0.), (0., 1.)), **kwargs)


def model():
    return TinyRoleModel(TinyModelConfig(feature_dim=3, option_dim=2, hidden_dim=8, lora_rank=2))


def trajectory(dec=None, seed=1, final=2.):
    return SearchTrajectory("g", "context", f"search-{seed}", seed,
        (TransitionRecord(0., 1., 2.), TransitionRecord(1., final, 3.)),
        (dec or decision(),), final_f=final, lineage_id="root")


def test_completed_labels_are_not_success_or_interruption():
    with pytest.raises(ValueError, match="Interrupted"):
        CheckerAudit("root", "checker-v1", False, 0, (1.,))
    with pytest.raises(ValueError, match="binary"):
        CheckerAudit("root", "checker-v1", True, None, (1.,))
    assert CheckerAudit("root", "checker-v1", False, None, (1.,)).label is None


def test_lineage_split_protects_entire_audit_lineage():
    records = [{"lineage_id": f"l{i}", "audit_only": False} for i in range(10)]
    records += [{"lineage_id": "l0", "audit_only": True}, {"lineage_id": "l4", "audit_only": False}]
    train, val, audit = split_by_lineage(records, .3, 1)
    sets = [{r["lineage_id"] for r in part} for part in (train, val, audit)]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sets[2] == {"l0"}
    assert sum(len(x) for x in (train, val, audit)) == len(records)


def test_return_to_go_cost_origin_and_terminal_completion():
    t = trajectory()
    assert t.full_return(.1) == pytest.approx(1.5)
    assert t.target(t.decisions[0], .1) == pytest.approx(1.5)
    late = replace(t.decisions[0], transition_index=1, prior_cost=.4)
    assert t.target(late, .1) == pytest.approx(.66)
    assert sum(x.f_after - x.f_before - .1*x.cost for x in t.transitions) == pytest.approx(t.full_return(.1))


def test_fixed_support_and_no_artificial_solver_likelihood():
    with pytest.raises(ValueError, match="canonicalized"):
        DecisionRecord("batch", ("duplicate", "duplicate"), 0, -.5)
    with pytest.raises(ValueError, match="Forced"):
        replace(decision(), deterministic=True)
    forced = replace(decision(), deterministic=True, behavior_logprob=None)
    assert forced.behavior_logprob is None
    with pytest.raises(ValueError, match="disagrees"):
        replace(decision(), behavior_probabilities=(.9, .1))


def test_loo_group_normalization_and_clipping_exact_equations():
    returns = torch.tensor([1., 2., 6.])
    assert torch.allclose(group_advantages(returns), torch.tensor([-3., -1.5, 4.5]))
    assert torch.isclose(group_advantages(returns, "grpo").mean(), torch.tensor(0.), atol=1e-7)
    clipped = clipped_surrogate(torch.tensor([math.log(2.), math.log(.5)]), torch.zeros(2), torch.tensor([1., -1.]), .2)
    assert torch.allclose(clipped, torch.tensor([1.2, -.8]))
    assert torch.equal(group_advantages(torch.ones(3)), torch.zeros(3))


def test_collection_mixture_positive_on_all_legal_actions():
    p = mixture_probabilities(torch.tensor([1000., -1000., 0.]), exploration=.1)
    assert (p > 0).all() and p.sum() == pytest.approx(1.)
    assert p[1] == pytest.approx(.1/3)


def test_isolated_full_search_group_no_shared_discoveries():
    original = {"ledger": ["initial"]}
    seen = []
    def rollout(state, rng, meta):
        seen.append(list(state["ledger"]))
        state["ledger"].append(meta["search_id"])
        return SearchTrajectory(**meta, transitions=(TransitionRecord(0., 0., 1.),), decisions=())
    group = collect_independent_group(original, rollout, group_size=4, seed=0, group_id="g", initial_context_id="ctx")
    assert seen == [["initial"]] * 4 and original == {"ledger": ["initial"]}
    assert len({x.seed for x in group}) == 4
    with pytest.raises(ValueError, match="distinct"):
        validate_group([group[0], group[0]])


def test_only_selected_option_gets_realized_value_target():
    net = model()
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01, value_weight=1.))
    before = {n: p.clone() for n,p in net.backbone.named_parameters()}
    t = trajectory()
    metrics = trainer.warmup_epoch(trajectories=[t])
    assert math.isfinite(metrics["value_loss"])
    assert all(torch.equal(p, before[n]) and p.grad is None for n,p in net.backbone.named_parameters())


def test_supervision_rejects_mixed_protocols_and_audit_data():
    trainer = SpecRSITrainer(model(), TrainingConfig())
    a = CheckerAudit("l1", "v1", True, 1, (1., 0., 0.))
    with pytest.raises(ValueError, match="protocol"):
        trainer.warmup_epoch(audits=[a, replace(a, protocol_id="v2")])
    with pytest.raises(ValueError, match="audit"):
        trainer.warmup_epoch(audits=[replace(a, audit_only=True)])


def test_roundtrip_training_checkpoint_and_resumed_optimizer(tmp_path):
    net = model()
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01))
    audits = [CheckerAudit(f"l{i}", "v1", True, i%2, (float(i%2), 0., 1.)) for i in range(8)]
    before = trainer.evaluate(audits=audits)["surface_completed_population"]["log_loss"]
    for _ in range(15):
        trainer.warmup_epoch(audits=audits)
    after = trainer.evaluate(audits=audits)["surface_completed_population"]["log_loss"]
    assert after < before
    trainer.save(tmp_path)
    loaded = TinyRoleModel.load(tmp_path)
    assert torch.equal(net.decision_values(decision()), loaded.decision_values(decision()))
    resumed = SpecRSITrainer(loaded, trainer.config)
    resumed.restore_optimizer(tmp_path)
    assert resumed.steps == trainer.steps


def test_gvo_sums_decisions_and_uses_held_behavior_support():
    net = model()
    d = decision()
    with torch.no_grad():
        probs = mixture_probabilities(net.decision_logits(d))
    d = replace(d, behavior_logprob=float(probs[0].log()), behavior_probabilities=tuple(probs.tolist()))
    group = [trajectory(d, seed=1, final=2.), trajectory(replace(d, selected_index=1,
        behavior_logprob=float(probs[1].log())), seed=2, final=1.)]
    trainer = SpecRSITrainer(net, TrainingConfig(kl_weight=.1))
    metrics = trainer.gvo_epoch([group])
    assert metrics["searches"] == 2 and metrics["stochastic_decisions"] == 2
    assert math.isfinite(metrics["value_loss"])
    assert metrics["kl"] == pytest.approx(0., abs=1e-6)


def test_incomplete_search_and_inconsistent_f_rejected():
    with pytest.raises(ValueError, match="terminated"):
        replace(trajectory(), terminated=False)
    with pytest.raises(ValueError, match="continuous"):
        replace(trajectory(), transitions=(TransitionRecord(0., 1., 2.), TransitionRecord(0., 2., 3.)))


def test_serialized_trajectory_roundtrip():
    from dataclasses import asdict
    t = trajectory()
    assert trajectory_from_dict(asdict(t)) == t
