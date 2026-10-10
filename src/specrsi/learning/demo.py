"""Explicit synthetic smoke data, never an author-data or benchmark substitute."""
from __future__ import annotations

import math
import random

import torch

from .objectives import mixture_probabilities
from .records import CheckerAudit, DecisionRecord, SearchTrajectory, TransitionRecord, collect_independent_group


def demo_audits(feature_dim: int = 8, count: int = 32):
    audits = []
    for i in range(count):
        # A bounded equality check on one completely specified integer outcome.
        source_input, child_offset = i % 4, i % 2
        source_result, child_result = source_input, source_input + child_offset
        features = [0.] * feature_dim
        features[0] = float(child_offset)
        features[1] = float(source_input) / 4
        audits.append(CheckerAudit(f"synthetic-lineage-{i}", "demo-integer-equality-v1", True,
                                   int(source_result == child_result), tuple(features)))
    return audits


def collect_demo_groups(model, *, groups=4, group_size=4, seed=0, round_index=0):
    """A complete two-candidate integer-expression search with explicit toy costs."""
    if model.backend != "tiny":
        raise ValueError("Synthetic demo uses the tiny feature model only")
    result = []
    model.eval()
    for context in range(groups):
        def rollout(state, rng, metadata):
            features = [0.] * model.config.feature_dim
            features[0] = state["initial_offset"]
            options = []
            for option in range(2):
                values = [0.] * model.config.option_dim
                values[0] = float(option)
                values[1] = 1. - float(option)
                options.append(tuple(values))
            draft = DecisionRecord("edit", ("keep-offset", "repair-offset"), 0, math.log(.5),
                                    tuple(features), tuple(options), proposal_snapshot=f"demo-round-{round_index}")
            with torch.no_grad():
                probs = mixture_probabilities(model.decision_values(draft))
            selected = rng.choices(range(2), weights=probs.cpu().tolist())[0]
            from dataclasses import replace
            decision = replace(draft, selected_index=selected, behavior_logprob=float(probs[selected].log()),
                               behavior_probabilities=tuple(probs.cpu().tolist()))
            # Both required outcomes are directly executed, then candidate score
            # is computed. There is no reward for an incomplete candidate.
            offset = 0 if selected == 1 else state["initial_offset"]
            correct = [int(x + offset == x) for x in (2, 3)]
            initial_score = 0.
            complete_gain = sum(correct) / len(correct)
            return SearchTrajectory(**metadata, transitions=(TransitionRecord(0., 0., .1),
                TransitionRecord(0., complete_gain, .2), TransitionRecord(complete_gain, complete_gain, .05)),
                decisions=(decision,), initial_f=initial_score, final_f=complete_gain,
                lineage_id=f"synthetic-search-lineage-{context}")
        result.append(collect_independent_group({"initial_offset": 1}, rollout, group_size=group_size,
            seed=seed + context + round_index * 10000, group_id=f"synthetic-round-{round_index}-group-{context}",
            initial_context_id=f"synthetic-context-{context}"))
    return result
