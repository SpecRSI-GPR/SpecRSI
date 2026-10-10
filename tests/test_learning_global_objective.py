"""Closed-form, CPU-only checks of the PDF's collection-level objectives."""
from dataclasses import replace
import json
import math

import pytest
import torch

from specrsi.learning.records import DecisionRecord, PairwiseValueExample, SearchTrajectory, TransitionRecord
from specrsi.learning.trainer import SpecRSITrainer, TrainingConfig

torch.set_num_threads(2)


class ScalarRoles(torch.nn.Module):
    """Three unrelated scalars, not a pretrained or served language model."""
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.policy_head = torch.nn.Linear(1, 1, bias=False)
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        self.draft_adapter = torch.nn.Linear(1, 1, bias=False)
        for parameter in self.parameters():
            torch.nn.init.zeros_(parameter)
        self.calls = []

    def _record(self, role):
        self.calls.append((role, tuple(float(p.detach().flatten()[0]) for p in self.parameters())))

    def decision_logits(self, decision):
        role = "draft" if decision.kind == "edit_token" else "policy"
        self._record(role)
        value = (self.draft_adapter if role == "draft" else self.policy_head).weight.flatten()[0]
        return torch.stack((value, -value))

    def decision_values(self, decision):
        self._record("value")
        value = self.value_head.weight.flatten()[0]
        return value * torch.tensor([option[0] for option in decision.option_features])

    def save(self, directory):
        from pathlib import Path
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "scalars.pt")


def plan(*, deterministic=True, selected=0):
    return DecisionRecord("plan", ("left", "right"), selected,
        None if deterministic else math.log(.5), option_features=((1.,), (-1.,)),
        deterministic=deterministic, exploration=0.)


def trajectory(target, seed, *, decisions=None, forced=False, group="group"):
    gain, cost = max(target, 0.), max(-target, 0.)
    return SearchTrajectory(group, "same-root", f"search-{seed}", seed,
        (TransitionRecord(0., gain, cost),), (plan(),) if decisions is None else tuple(decisions),
        final_f=gain, forced=forced, collection_snapshot="snapshot", lineage_id="development")


def trainer(**kwargs):
    model = ScalarRoles()
    result = SpecRSITrainer(model, TrainingConfig(learning_rate=.1, weight_decay=0.,
        cost_coefficient=1., max_grad_norm=1e6, **kwargs))
    # A transparent optimizer makes the actual parameter change equal the
    # independently derived gradient, without Adam/clip obscuring the oracle.
    result.optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    return result


def pair():
    left = plan()
    return PairwiseValueExample(left, replace(left, selected_index=1), (2.,), (0.,),
        "same-paid-state", "snapshot", (10,), (11,), 1, 1, lineage_id="development")


def test_q_union_and_pair_are_separate_empirical_means_at_one_snapshot():
    training = trainer(value_weight=2., pair_weight=.25)
    group = [trajectory(2., 1), trajectory(4., 2)]
    forced = [trajectory(-1., 3, forced=True)]
    result = training.gvo_epoch([group], forced_trajectories=forced, pairs=[pair()])
    # At q=0: grad(2 * mean((q-[2,4,-1])**2)) = -20/3.
    # grad(.25 * ((q-(-q))-2)**2) = -2.  Total -26/3.
    assert float(training.model.value_head.weight) == pytest.approx(.1 * 26 / 3)
    assert result["value_loss"] == pytest.approx(7.)
    assert result["pair_loss"] == pytest.approx(4.)
    assert training.steps == result["learning_signal"]["optimizer_steps"] == 1
    assert {state for _, state in training.model.calls} == {(0., 0., 0.)}
    schedule = result["update_schedule"]
    assert schedule["q_tuple_counts"] == {"ordinary": 2, "forced": 1, "total": 3}
    assert schedule["q_source_mass"] == {"ordinary": 2 / 3, "forced": 1 / 3}
    assert schedule["pair_count"] == 1
    assert schedule["evaluation_point"] == "one_preupdate_parameter_snapshot"


def test_q_weight_is_invariant_to_source_and_actor_group_partition():
    trajectories = [trajectory(value, index) for index, value in enumerate((1., 2., 3., -2., 0., 4.))]
    arrangements = [
        ([trajectories], []),
        ([trajectories[:2], [replace(t, group_id="second") for t in trajectories[2:]]], []),
        ([trajectories[:2]], [replace(t, forced=True) for t in trajectories[2:]]),
    ]
    observed = []
    for groups, forced in arrangements:
        training = trainer(batch_size=1)
        result = training.gvo_epoch(groups, forced_trajectories=forced)
        observed.append(float(training.model.value_head.weight))
        assert result["update_schedule"]["q_tuple_counts"]["total"] == 6
        assert training.steps == 1
    assert observed == pytest.approx([.1 * 2 * (8 / 6)] * 3)


def test_actor_sums_tokens_and_means_all_searches_with_group_local_loo():
    left = DecisionRecord("edit_token", ("2", "3"), 0, math.log(.5), input_ids=(1,), exploration=0.)
    next_left = replace(left, input_ids=(1, 2))
    right = replace(left, selected_index=1)
    groups = [[trajectory(1., 0, decisions=[left, next_left]), trajectory(0., 1, decisions=[right])],
              [trajectory(0., i, decisions=[], group="second") for i in range(2, 6)]]
    training = trainer(value_weight=0.)
    result = training.gvo_epoch(groups)
    # First root group advantages +1/-1. Derivative of log p(left/right)
    # at zero is +1/-1: gradient -(2 + 1) / 6, not a mean over 3 tokens.
    assert float(training.model.draft_adapter.weight) == pytest.approx(.05)
    assert result["actor_loss"] == pytest.approx(-1 / 6)
    assert result["stochastic_decisions"] == 3
    assert result["update_schedule"]["actor_group_sizes"] == [2, 4]
    assert result["update_schedule"]["actor_search_count"] == 6
    assert training.steps == 1


def test_forced_stochastic_records_never_enter_actor_objective():
    ordinary = [trajectory(0., 1, decisions=[]), trajectory(0., 2, decisions=[])]
    forced = trajectory(2., 3, forced=True, decisions=[plan(deterministic=False)])
    training = trainer()
    result = training.gvo_epoch([ordinary], forced_trajectories=[forced])
    assert all(role == "value" for role, _ in training.model.calls)
    assert result["stochastic_decisions"] == result["update_schedule"]["forced_actor_choices"] == 0
    assert training.model.policy_head.weight.grad is None
    assert result["learning_signal"]["status"] == "critic_only_updated"
    assert "forced_critic_updates" not in result["learning_signal"]


def test_empty_searches_and_zero_advantages_have_honest_update_evidence():
    training = trainer()
    empty = [trajectory(0., 1, decisions=[]), trajectory(0., 2, decisions=[])]
    result = training.gvo_epoch([empty])
    assert training.steps == 0
    assert result["update_schedule"]["q_source_mass"] == {"ordinary": 0., "forced": 0.}
    assert result["update_schedule"]["actual_optimizer_steps"] == 0
    assert result["learning_signal"]["status"] == "no_parameter_change"


def test_zero_cost_coefficient_preserves_charges_without_cost_only_actor_signal():
    training = trainer(value_weight=0.)
    training.config.cost_coefficient = 0.
    group = [trajectory(-cost, seed, decisions=[plan(deterministic=False)])
             for seed, cost in enumerate((1., 7., 23.))]
    result = training.gvo_epoch([group])
    assert [t.total_cost for t in group] == [1., 7., 23.]
    assert result["mean_search_return"] == 0.
    assert result["learning_signal"]["group_dispersion"][0]["loo_advantage_abs_max"] == 0.
    assert result["learning_signal"]["roles"]["policy"]["nonzero_gradient_steps"] == 0
    assert not result["learning_signal"]["all_actor_roles_updated"]
    assert float(training.model.policy_head.weight) == 0.


@pytest.mark.parametrize("problem", ["group", "forced_snapshot", "pair_snapshot", "missing_forced_snapshot",
    "missing_ordinary_snapshot", "legacy_unknown_with_known_pair"])
def test_whole_collection_validation_precedes_any_optimizer_update(problem):
    training = trainer(pair_weight=1.)
    groups = [[trajectory(1., 1), trajectory(2., 2)]]
    forced, pairs = [], []
    if problem == "group":
        groups.append([trajectory(1., 3, group="second")])
    elif problem == "forced_snapshot":
        forced = [replace(trajectory(2., 3, forced=True), collection_snapshot="different")]
    elif problem == "missing_forced_snapshot":
        forced = [replace(trajectory(2., 3, forced=True), collection_snapshot="")]
    elif problem == "missing_ordinary_snapshot":
        groups.append([replace(trajectory(1., seed, group="unknown"), collection_snapshot="") for seed in (3, 4)])
    elif problem == "legacy_unknown_with_known_pair":
        groups = [[replace(t, collection_snapshot="") for t in group] for group in groups]
        pairs = [pair()]
    else:
        pairs = [replace(pair(), snapshot_id="different")]
    with pytest.raises(ValueError):
        training.gvo_epoch(groups, forced_trajectories=forced, pairs=pairs)
    assert not training.model.calls and training.steps == 0
    assert all(float(parameter.flatten()[0]) == 0. for parameter in training.model.parameters())


def test_wholly_unspecified_legacy_snapshot_remains_supported():
    training = trainer()
    group = [replace(trajectory(1., seed), collection_snapshot="") for seed in (1, 2)]
    forced = [replace(trajectory(-1., 3, forced=True), collection_snapshot="")]
    result = training.gvo_epoch([group], forced_trajectories=forced)
    assert result["update_schedule"]["collection_snapshots"] == [""]
    assert training.steps == 1


def test_explicit_legacy_schedule_retains_group_then_forced_steps():
    training = trainer(gvo_update_schedule="legacy_group_then_forced_v1", batch_size=1)
    result = training.gvo_epoch([[trajectory(1., 1), trajectory(1., 2)]],
                                forced_trajectories=[trajectory(-1., 3, forced=True)])
    assert training.steps == 2
    assert float(training.model.value_head.weight) == pytest.approx(-.04)
    assert result["update_schedule"]["name"] == "legacy_group_then_forced_v1"
    assert result["learning_signal"]["ordinary_search_updates"]["optimizer_steps"] == 1
    assert result["learning_signal"]["forced_critic_updates"]["optimizer_steps"] == 1
    assert len({state for _, state in training.model.calls}) > 1


@pytest.mark.parametrize("override, expected", [(None, "legacy_group_then_forced_v1"),
    ("frozen_collection_v1", "frozen_collection_v1")])
def test_old_checkpoint_schedule_is_explicitly_preserved_or_overridden(tmp_path, override, expected):
    original = trainer()
    original.save(tmp_path)
    path = tmp_path / "training_config.json"
    manifest = json.loads(path.read_text())
    manifest.pop("gvo_update_schedule")
    path.write_text(json.dumps(manifest))
    resumed = trainer()
    resumed.restore_optimizer(tmp_path, gvo_update_schedule_override=override)
    assert resumed.config.gvo_update_schedule == expected
    evidence = resumed.schedule_restore_history[-1]
    assert evidence["checkpoint_schedule"] == "legacy_group_then_forced_v1"
    assert evidence["missing_checkpoint_schedule_uses_legacy"]
    assert evidence["explicit_override"] == override and evidence["resolved_schedule"] == expected
    resumed.save(tmp_path / "resaved")
    saved = json.loads((tmp_path / "resaved/training_evidence.json").read_text())
    assert saved["schedule_restore_history"] == resumed.schedule_restore_history
    assert saved["gvo_update_schedule"] == expected


def test_new_checkpoint_and_unknown_schedule_validation(tmp_path):
    original = trainer()
    original.save(tmp_path)
    resumed = trainer(gvo_update_schedule="legacy_group_then_forced_v1")
    resumed.restore_optimizer(tmp_path)
    assert resumed.config.gvo_update_schedule == "frozen_collection_v1"
    assert not resumed.schedule_restore_history[-1]["missing_checkpoint_schedule_uses_legacy"]
    with pytest.raises(ValueError, match="schedule"):
        TrainingConfig(gvo_update_schedule="unregistered")
    with pytest.raises(ValueError, match="schedule"):
        resumed.restore_optimizer(tmp_path, gvo_update_schedule_override="unregistered")
