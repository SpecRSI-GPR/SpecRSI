"""Analytic regression oracles for the GVO equations in Sec. 3.4 / F.5."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from specrsi.core import Candidate, EvaluationScope, Ledger, Task
from specrsi.core.models import Outcome
from specrsi.core.search import SearchConfig
from specrsi.learning.objectives import clipped_surrogate, current_reference_kl, group_advantages, regression_loss
from specrsi.learning.observations import RuleController
from specrsi.learning.records import TransitionRecord
from specrsi.learning.trainer import TrainingConfig
from test_learning_global_objective import pair, plan, trainer, trajectory


def test_eq13_keeps_one_point_gain_after_large_shared_search_cost():
    training = trainer(value_weight=0.)
    group = [trajectory(1., 1, decisions=[plan(deterministic=False)]),
             trajectory(0., 2, decisions=[plan(deterministic=False, selected=1)])]
    group = [replace(t, transitions=(TransitionRecord(0., t.final_f, 1e8),)) for t in group]
    result = training.gvo_epoch([group])
    # R=[-99999999,-100000000], so A=[+1,-1]. At policy scalar 0,
    # d log p(left)/dw=+1, d log p(right)/dw=-1; Eq. (20) gradient=-1.
    # Float32 return construction incorrectly collapsed both labels to -1e8.
    assert training.model.policy_head.weight.item() == pytest.approx(.1)
    assert result["learning_signal"]["group_dispersion"][0]["loo_nonzero_advantages"] == 2


def test_eq13_is_translation_invariant_and_returns_have_no_autograd_path():
    returns = torch.tensor([1e12, 1e12 + 1., 1e12 + 4.], dtype=torch.float64, requires_grad=True)
    observed = group_advantages(returns)
    assert not observed.requires_grad
    assert torch.equal(observed, group_advantages(returns.detach() - 1e12))
    assert torch.allclose(observed, torch.tensor([-2.5, -1., 3.5], dtype=torch.float64))


def test_eq20_and_eq23_detach_behavior_advantages_and_observed_targets():
    current = torch.tensor(0., requires_grad=True)
    behavior = torch.tensor(0., requires_grad=True)
    advantage = torch.tensor(2., requires_grad=True)
    clipped_surrogate(current, behavior, advantage).backward()
    assert current.grad.item() == 2.
    assert behavior.grad is advantage.grad is None
    prediction = torch.tensor(1., requires_grad=True)
    target = torch.tensor(3., requires_grad=True)
    regression_loss(prediction, target).backward()
    assert prediction.grad.item() == -4.
    assert target.grad is None


@pytest.mark.parametrize("current", [[-.1, 1.1], [0., 0.], [float("nan"), 1.], [float("inf"), 0.]])
def test_eq21_rejects_malformed_current_distributions(current):
    with pytest.raises(ValueError, match="current probabilities"):
        current_reference_kl(torch.tensor(current), torch.tensor([.5, .5]))


def test_eq21_handles_zero_mass_and_keeps_reference_fixed():
    current = torch.tensor([1., 0.], requires_grad=True)
    reference = torch.tensor([.5, .5], requires_grad=True)
    loss = current_reference_kl(current, reference)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.)).item())
    loss.backward()
    assert reference.grad is None
    assert torch.isfinite(current.grad).all()


@pytest.mark.parametrize("changed", ["support", "option_features", "proposal_snapshot", "same_arm"])
def test_eq22_requires_distinct_arms_from_the_same_saved_menu(changed):
    example = pair()
    changes = {"support": {"support": ("different-work", "right")},
               "option_features": {"option_features": ((100.,), (-1.,))},
               "proposal_snapshot": {"proposal_snapshot": "different-generator"},
               "same_arm": {"selected_index": example.left.selected_index}}
    with pytest.raises(ValueError, match="same recorded menu|distinct legal arms"):
        replace(example, right=replace(example.right, **changes[changed]))


def test_pair_regression_uses_one_full_menu_forward_with_exact_difference_gradient():
    training = trainer()
    loss = training._pair_loss(pair())
    # Q_left=w, Q_right=-w, delta=2, so L=(2w-2)^2, dL/dw=-8 at w=0.
    assert loss.item() == 4.
    loss.backward()
    assert training.model.value_head.weight.grad.item() == -8.
    assert [role for role, _ in training.model.calls] == ["value"]


@pytest.mark.parametrize("setting", ["learning_rate", "cost_coefficient", "pair_weight", "kl_weight"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_invalid_objective_coefficients_fail_before_optimizer_or_collection(setting, bad_value):
    with pytest.raises(ValueError, match="finite"):
        TrainingConfig(**{setting: bad_value})


def test_quality_baseline_counts_contract_operating_penalty_exactly_once():
    task = Task("quality")
    scope = EvaluationScope("quality", (task,), "fixture", "fixture")
    costly = Candidate("costly", {"value": 1.}, operating_penalty=.2)
    cheap = Candidate("cheap", {"value": .7})
    ledger = Ledger()
    ledger.admit_direct(costly, task, scope, Outcome(1., {"answer": "fixture"}), {"fixture": True})
    ledger.admit_direct(cheap, task, scope, Outcome(.7, {"answer": "fixture"}), {"fixture": True})
    state = SimpleNamespace(ledger=ledger, scope=scope)
    rule = RuleController(SearchConfig())
    # J_Gamma(costly)=1-.2=.8 > .7. Double subtraction would reverse this
    # ranking, accidentally weakening a baseline on a different utility.
    assert rule._quality(state, costly) == pytest.approx(.8)
    assert rule._quality(state, costly) > rule._quality(state, cheap)
