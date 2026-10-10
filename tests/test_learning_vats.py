from dataclasses import replace
import math

import pytest
import torch

from specrsi.learning import SpecRSITrainer, TrainingConfig
from specrsi.learning.heads import TinyModelConfig, TinyRoleModel
from specrsi.learning.objectives import mixture_probabilities
from specrsi.learning.paper_bridge import collect_reference_vats
from specrsi.learning.records import DecisionRecord, PairwiseValueExample, SearchTrajectory, TransitionRecord, validate_group

torch.set_num_threads(2)


def model():
    return TinyRoleModel(TinyModelConfig(option_dim=12, hidden_dim=16))


def decision(net):
    d = DecisionRecord("plan", ("draft", "ACCEPT"), 0, math.log(.5),
        state_features=(0.,) * 8, option_features=((0.,) * 12, (1.,) + (0.,) * 11),
        graph_nodes=((1.,) + (0.,) * 11, (0., 1.) + (0.,) * 10),
        graph_edges=((0, 1, 2), (1, 0, 3)), option_node_masks=((1., 0.), (0., 0.)))
    with torch.no_grad():
        p = mixture_probabilities(net.decision_logits(d))
    return replace(d, behavior_logprob=float(p[0].log()), behavior_probabilities=tuple(p.tolist()))


def trajectory(d, seed=0, final=1., forced=False):
    accept = d.support[d.selected_index] == "ACCEPT"
    return SearchTrajectory("group", "same-context", f"search-{seed}", seed,
        (TransitionRecord(0., final, 0. if accept else 3., elapsed_seconds=0. if accept else 2.),), (d,), final_f=final,
        forced=forced, collection_snapshot="frozen-v2", initial_budget=100., workers=1)


def test_actor_and_scorer_have_disjoint_parameters_and_updates():
    net = model()
    d = decision(net)
    assert net.action_head is net.batch_head is net.value_head
    assert not ({id(p) for p in net.value_head.parameters()} & {id(p) for p in net.policy_head.parameters()})
    assert net.decision_logits(d)[1].item() == 0.
    assert net.decision_values(d)[1].item() == 0.
    before_value = {name: p.clone() for name, p in net.value_head.named_parameters()}
    before_policy = {name: p.clone() for name, p in net.policy_head.named_parameters()}
    second = replace(d, selected_index=1, behavior_logprob=math.log(d.behavior_probabilities[1]))
    group = [trajectory(d, 0, 1.), trajectory(second, 1, 0.)]
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01, value_weight=0., weight_decay=.1))
    trainer.gvo_epoch([group])
    assert all(torch.equal(p, before_value[name]) for name, p in net.value_head.named_parameters())
    assert any(not torch.equal(p, before_policy[name]) for name, p in net.policy_head.named_parameters())
    policy_after_actor = {name: p.clone() for name, p in net.policy_head.named_parameters()}
    trainer.config.value_weight = 1.
    trainer.warmup_epoch(trajectories=group)
    assert all(torch.equal(p, policy_after_actor[name]) for name, p in net.policy_head.named_parameters())


def test_elapsed_return_excludes_only_current_paid_preparation():
    net = model()
    d = replace(decision(net), transition_index=1)
    t = SearchTrajectory("g", "s", "id", 1,
        (TransitionRecord(0., 0., 2., elapsed_seconds=1.),
         TransitionRecord(0., 1., 3., elapsed_seconds=2.),
         TransitionRecord(1., 1., 4., elapsed_seconds=3.)), (d,), final_f=1.)
    assert t.full_return(.1, .2) == pytest.approx(1 - .9 - 1.2)
    assert t.target(d, .1, .2) == pytest.approx(1 - .7 - 1.)


def test_forced_pairs_train_only_scorer_and_require_all_registered_repetitions():
    net = model()
    left = decision(net)
    right = replace(left, selected_index=1, behavior_logprob=math.log(left.behavior_probabilities[1]))
    pair = PairwiseValueExample(left, right, (1., -1.), (0.,), "saved-paid-state", "snapshot",
        (10, 11), (12,), 2, 1)
    assert pair.difference == 0.  # The unsuccessful repetition is retained.
    with pytest.raises(ValueError, match="registered"):
        replace(pair, left_returns=(1.,))
    with pytest.raises(ValueError, match="independent"):
        replace(pair, right_seeds=(10,))
    forced = trajectory(left, forced=True)
    with pytest.raises(ValueError, match="forced"):
        validate_group([forced, trajectory(left, seed=1)])
    before = {name: p.clone() for name, p in net.policy_head.named_parameters()}
    trainer = SpecRSITrainer(net, TrainingConfig(learning_rate=.01, pair_weight=1.))
    trainer.warmup_epoch(trajectories=[forced], pairs=[pair])
    assert all(torch.equal(p, before[name]) for name, p in net.policy_head.named_parameters())


def test_vats_collector_replays_paid_state_graph_and_exact_behavior():
    net = model()
    result = collect_reference_vats(net, 0, {"collection": {"contexts": 1, "group_size": 2,
        "budget": 200, "max_actions": 8, "vats": {"max_queries": 8, "workers": 2}},
        "training": {"seed": 0}})
    group = result["groups"][0]
    assert all(t.workers == 2 for t in group)
    assert sum(t.total_cost for t in group) == result["metadata"]["search_cost"]
    verify_seen = False
    for t in group:
        for d in t.decisions:
            with torch.no_grad():
                p = mixture_probabilities(net.decision_logits(d), d.temperature, d.exploration)
            assert float(p[d.selected_index].log()) == pytest.approx(d.behavior_logprob, abs=2e-6)
            if d.kind == "plan":
                assert d.graph_nodes and d.graph_edges
                assert d.prior_cost == d.excluded_prefix_cost == 0.
                for uid, mask in zip(d.support, d.option_node_masks):
                    if uid.startswith("verify:"):
                        verify_seen = True
                        assert sum(mask) >= 2  # Requested job and its candidate.
    assert verify_seen
    metrics = SpecRSITrainer(net, TrainingConfig(kl_weight=.1)).gvo_epoch([group])
    assert metrics["kl"] == pytest.approx(0., abs=1e-6)
    assert math.isfinite(metrics["value_loss"])


def test_group_rejects_mismatched_collection_snapshot_and_workers():
    net = model()
    d = decision(net)
    a, b = trajectory(d, 0), trajectory(d, 1)
    with pytest.raises(ValueError, match="collection_snapshot"):
        validate_group([a, replace(b, collection_snapshot="new-policy")])
    with pytest.raises(ValueError, match="workers"):
        validate_group([a, replace(b, workers=2)])


def test_relational_branch_is_invariant_to_graph_node_order():
    net = model()
    d = decision(net)
    permuted = replace(d, graph_nodes=tuple(reversed(d.graph_nodes)),
        graph_edges=tuple((1 - source, 1 - target, kind) for source, target, kind in d.graph_edges),
        option_node_masks=tuple(tuple(reversed(mask)) for mask in d.option_node_masks))
    assert torch.allclose(net.decision_values(d), net.decision_values(permuted), atol=1e-6)
    assert torch.allclose(net.decision_logits(d), net.decision_logits(permuted), atol=1e-6)


def test_paid_alternative_collection_keeps_forced_rollouts_out_of_actor_groups():
    net = model()
    config = {"collection": {"contexts": 1, "group_size": 2, "budget": 200,
        "max_actions": 6, "forced_repetitions": 2, "vats": {"max_queries": 8}}}
    result = collect_reference_vats(net, 0, config)
    assert len(result["pairs"]) == 1 and len(result["forced_trajectories"]) == 4
    pair = result["pairs"][0]
    assert pair.left_count == pair.right_count == 2
    assert pair.left.support[pair.left.selected_index] == "ACCEPT"
    assert pair.left_returns == (0., 0.)
    forced = result["forced_trajectories"]
    assert all(t.forced for t in forced)
    assert all(not t.forced for group in result["groups"] for t in group)
    assert all(t.decisions[0].deterministic and t.decisions[0].behavior_logprob is None for t in forced)
    assert sum(t.total_cost for t in forced) == result["metadata"]["forced_search_cost"]
    metrics = SpecRSITrainer(net, TrainingConfig(pair_weight=.1)).gvo_epoch(result["groups"],
        forced_trajectories=forced, pairs=result["pairs"])
    assert metrics["forced_searches"] == 4 and metrics["pair_examples"] == 1
    assert math.isfinite(metrics["pair_loss"])
