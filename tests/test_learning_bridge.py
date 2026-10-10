import math

import torch

from specrsi.learning import SpecRSITrainer, TinyModelConfig, TinyRoleModel, TrainingConfig
from specrsi.learning.bridge import collect_reference_specrsi
from specrsi.learning.objectives import mixture_probabilities
from specrsi.learning.records import load_jsonl, save_jsonl, trajectory_from_dict

torch.set_num_threads(2)


def test_real_specrsi_collection_replays_exact_inputs_and_costs(tmp_path):
    model = TinyRoleModel(TinyModelConfig(feature_dim=8,option_dim=8,hidden_dim=16,lora_rank=2))
    result = collect_reference_specrsi(model,0,{"collection":{"contexts":1,"group_size":2,
        "max_actions":8,"budget":120,"max_candidates":5},"training":{"seed":4}})
    group = result["groups"][0]
    assert len({t.search_id for t in group}) == 2
    assert len({t.initial_context_id for t in group}) == 1
    assert result["metadata"]["search_cost"] == sum(t.total_cost for t in group)
    assert result["metadata"]["initial_record_execution_cost"] == 8
    assert {d.kind for t in group for d in t.decisions} == {"outer","batch","edit"}
    for trajectory in group:
        for decision in trajectory.decisions:
            with torch.no_grad():
                probs = mixture_probabilities(model.decision_logits(decision),decision.temperature,decision.exploration)
            assert abs(float(probs[decision.selected_index].log())-decision.behavior_logprob) < 2e-6
            assert decision.prior_cost == 0  # Planning already occurs in actual transition costs.
    path = tmp_path / "collected.jsonl"
    save_jsonl(path,group)
    records = load_jsonl(path)
    assert records[0]["record_type"] == "trajectory"
    assert trajectory_from_dict(records[0]) == group[0]
    trainer = SpecRSITrainer(model,TrainingConfig(learning_rate=1e-4,kl_weight=.01))
    metrics = trainer.gvo_epoch(result["groups"],audits=result["audits"])
    assert metrics["stochastic_decisions"] > 5 and math.isfinite(metrics["value_loss"])
    assert abs(metrics["kl"]) < 1e-6


def test_reference_audits_come_from_actual_completed_checker_calls():
    model = TinyRoleModel(TinyModelConfig())
    result = collect_reference_specrsi(model,0,{"collection":{"contexts":1,"group_size":2,
        "max_actions":12,"budget":150,"max_candidates":6},"training":{"seed":0}})
    expected = [check for raw in result["metadata"]["raw_search_results"] for transition in raw["transitions"]
                for event in transition["events"] for check in event.get("checks",[])]
    assert len(result["audits"]) == len(expected)
    assert all(a.completed and a.label in (0,1) for a in result["audits"])
    assert [a.label for a in result["audits"]] == [int(c["status"] == "P") for c in expected]
