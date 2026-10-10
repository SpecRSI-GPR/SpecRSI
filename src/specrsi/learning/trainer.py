"""Warm-up and complete-search GVO with fixed supports and realized value targets."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Sequence

import torch
from torch.nn import functional as F

from .objectives import binary_calibration, clipped_surrogate, current_reference_kl, group_advantages, mixture_probabilities, regression_loss
from .records import CheckerAudit, DecisionRecord, EditExample, PairwiseValueExample, SearchTrajectory, validate_group


_ROLES = ("draft", "policy", "value", "verify", "surface", "other")
_ROLE_COUNTS = ("gradient_steps", "nonzero_gradient_steps", "parameter_change_steps",
                "signal_parameter_change_steps", "zero_gradient_change_steps", "changed_elements")
_GVO_SCHEDULES = {"frozen_collection_v1", "legacy_group_then_forced_v1"}
_OPTIMIZER_RULES = {"adamw_legacy_v1", "pdf_gradient_v1"}
_KL_REFERENCES = {"behavior_snapshot", "frozen_sft_actor"}


def _empty_role_evidence():
    return {**dict.fromkeys(_ROLE_COUNTS, 0), "gradient_l2_max": 0.}


def merge_learning_signals(signals):
    """Aggregate observed updates, never infer actor learning from optimizer steps.

    A signal change means a coordinate changed in a step in which its clipped
    gradient was nonzero. This is observational evidence, not an attribution of
    the change solely to that gradient (Adam moments/decay also affect updates).
    Missing old checkpoint evidence remains explicitly unknown.
    """
    signals = list(signals)
    known = [s for s in signals if s and s.get("schema") == "role_update_evidence_v1"]
    roles = {role: _empty_role_evidence() for role in _ROLES}
    for signal in known:
        for role in _ROLES:
            source = signal["roles"][role]
            for key in _ROLE_COUNTS:
                roles[role][key] += source[key]
            roles[role]["gradient_l2_max"] = max(roles[role]["gradient_l2_max"], source["gradient_l2_max"])
    actors = [role for role in ("draft", "policy") if roles[role]["signal_parameter_change_steps"]]
    critic = bool(roles["value"]["signal_parameter_change_steps"])
    if not known or len(known) != len(signals):
        status = "legacy_or_missing_evidence"
    elif len(actors) == 2:
        status = "actor_and_critic_updated" if critic else "all_actor_roles_updated"
    elif actors:
        status = "partial_actor_update"
    elif critic:
        status = "critic_only_updated"
    elif any(roles[r]["signal_parameter_change_steps"] for r in ("verify", "surface", "other")):
        status = "auxiliary_only_updated"
    elif any(row["parameter_change_steps"] for row in roles.values()):
        status = "optimizer_only_parameter_change"
    else:
        status = "no_parameter_change"
    return {"schema": "role_update_evidence_v1", "status": status,
            "diagnostics_complete": bool(known) and len(known) == len(signals),
            "roles": roles, "actor_roles_updated": actors,
            "all_actor_roles_updated": len(actors) == 2 and len(known) == len(signals),
            "critic_updated": critic, "optimizer_steps": sum(s["optimizer_steps"] for s in known),
            "scope_note": "Counts are observed per-step events; zero advantages are legal. Parameter changes alone may reflect Adam moments or weight decay."}


def _parameter_role(name):
    if name.startswith(("draft_adapter.", "edit_head.")) or ("lora_" in name and ".draft." in name):
        return "draft"
    if name.startswith(("verify_adapter.",)) or ("lora_" in name and ".verify." in name):
        return "verify"
    for role in ("policy", "value", "surface"):
        if name.startswith(role + "_head."):
            return role
    return "other"


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    cost_coefficient: float = .01
    time_coefficient: float = 0.
    surface_weight: float = 1.
    plan_weight: float = 1.
    edit_weight: float = 1.
    value_weight: float = 1.
    pair_weight: float = 0.
    kl_weight: float = 0.
    clip_epsilon: float = .2
    estimator: str = "gvo"
    regression: str = "mse"
    max_grad_norm: float = 1.
    batch_size: int = 8
    seed: int = 0
    checker_protocol_id: str = ""
    reward_mode: str = 'complete_search'
    gvo_update_schedule: str = "frozen_collection_v1"
    optimizer_rule: str = "adamw_legacy_v1"
    actor_learning_rate: float | None = None
    value_learning_rate: float | None = None
    actor_max_grad_norm: float = 0.
    value_max_grad_norm: float = 0.
    kl_reference: str = "behavior_snapshot"

    def __post_init__(self):
        finite_settings = ("learning_rate", "weight_decay", "cost_coefficient", "time_coefficient",
            "surface_weight", "plan_weight", "edit_weight", "value_weight", "pair_weight", "kl_weight",
            "clip_epsilon", "max_grad_norm")
        if any(not math.isfinite(getattr(self, name)) for name in finite_settings):
            raise ValueError("Training rates, coefficients, loss weights and clipping settings must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or min(self.cost_coefficient, self.time_coefficient) < 0 or type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Invalid training optimizer, cost coefficient or batch size")
        if not 0 < self.clip_epsilon < 1 or self.max_grad_norm < 0:
            raise ValueError("Clipping epsilon must lie in (0, 1) and the legacy norm limit must be nonnegative")
        if self.estimator not in {"gvo", "grpo", "rloo"} or self.regression not in {"mse", "huber"}:
            raise ValueError("Unknown estimator or regression objective")
        if self.reward_mode not in {'complete_search','completed_items'}:
            raise ValueError('Unknown reward mode')
        if self.gvo_update_schedule not in _GVO_SCHEDULES:
            raise ValueError("Unknown GVO update schedule")
        if self.optimizer_rule not in _OPTIMIZER_RULES or self.kl_reference not in _KL_REFERENCES:
            raise ValueError("Unknown optimizer rule or KL reference")
        if self.optimizer_rule == "pdf_gradient_v1" and self.weight_decay != 0:
            raise ValueError("Eq23 plain-gradient updates require weight_decay=0")
        if any(rate is not None and (not math.isfinite(rate) or rate <= 0)
               for rate in (self.actor_learning_rate, self.value_learning_rate)):
            raise ValueError("Actor/value learning rates must be positive finite values")
        if any(not math.isfinite(norm) or norm < 0 for norm in (self.actor_max_grad_norm, self.value_max_grad_norm)):
            raise ValueError("Role clipping norms must be finite and nonnegative; zero disables clipping")
        if self.optimizer_rule == "pdf_gradient_v1" and self.gvo_update_schedule != "frozen_collection_v1":
            raise ValueError("Eq23 plain-gradient updates require the frozen-collection objective")
        if any(getattr(self, x) < 0 for x in ("surface_weight", "plan_weight", "edit_weight", "value_weight", "pair_weight", "kl_weight")):
            raise ValueError("Loss weights cannot be negative")


class SpecRSITrainer:
    def __init__(self, model, config: TrainingConfig):
        self.model, self.config = model, config
        # In particular, the pretrained backbone is NOT in the optimizer.
        self.optimizer = self._make_optimizer()
        self.actor_reference = None
        self.optimizer_restore_history = []
        self.steps = 0
        self._update_events = []
        self.learning_history = []
        self.inherited_evidence_status = "new_training"
        self.schedule_restore_history = []
        self.cuda_memory_evidence = None
        named = model.named_parameters() if hasattr(model, "named_parameters") else (
            (f"unclassified.{i}", p) for i, p in enumerate(model.parameters()))
        self._parameter_roles = {id(p): _parameter_role(name) for name, p in named}

    def _make_optimizer(self):
        if self.config.optimizer_rule == "adamw_legacy_v1":
            return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if self.config.weight_decay != 0:
            raise ValueError("Eq23 plain-gradient updates require weight_decay=0")
        groups = defaultdict(list)
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                role = _parameter_role(name)
                groups["actor" if role in {"draft", "policy"} else "value" if role == "value" else "auxiliary"].append(parameter)
        rates = {"actor": self.config.actor_learning_rate or self.config.learning_rate,
                 "value": self.config.value_learning_rate or self.config.learning_rate,
                 "auxiliary": self.config.learning_rate}
        # Eq. (23), with vartheta=(theta,psi) and L_value=L_Q+beta_pair*L_pair:
        #   phi_{k+1} = phi_k - eta_phi * [grad_phi L_value(phi)]_{phi=phi_k},
        #   vartheta_{k+1} = vartheta_k
        #       - eta_pi * [grad_vartheta (L_GVO(vartheta)
        #                        + beta_KL*L_KL(vartheta))]_{vartheta=vartheta_k},
        #   Theta_{k+1} = (theta_{k+1},psi_{k+1},phi_{k+1},nu),
        #   bar_phi_{k+1} = phi_{k+1}.
        # actor_learning_rate maps to eta_pi and value_learning_rate to eta_phi;
        # each falls back to learning_rate when its separate rate is unspecified.
        # Actor and value parameters are disjoint, even when their text encoder
        # is shared and frozen. Accumulating their separate gradients and then
        # calling one SGD step implements both assignments at the same initial
        # parameter point. Momentum and weight decay are absent from this rule.
        # Returns, advantages, contexts, supports, old probabilities and reference
        # distributions stay constant in these derivatives. The next collector
        # takes its immutable bar_phi from the updated live phi; nu and the real
        # verification contract remain fixed. Parameter refresh does not turn a
        # return collected under pi_k into a new observation under pi_{k+1}.
        return torch.optim.SGD([{"params": parameters, "lr": rates[role], "update_role": role}
                                for role, parameters in groups.items()], lr=self.config.learning_rate,
                               momentum=0., weight_decay=0.)

    def bind_actor_reference(self, *, provenance):
        if self.config.kl_reference != "frozen_sft_actor":
            raise ValueError("A frozen actor reference must be explicitly registered")
        if self.actor_reference is not None:
            raise ValueError("The fixed SFT reference cannot be refreshed each collection round")
        from .reference_policy import FrozenActorReference
        self.actor_reference = FrozenActorReference(self.model, provenance=provenance)
        return self.actor_reference.identity

    def _reference_rows(self, decisions):
        if not self.config.kl_weight or self.config.kl_reference == "behavior_snapshot":
            return (None,) * len(decisions)
        if self.actor_reference is None:
            raise ValueError("Frozen SFT actor KL requires an identified reference before collection/update")
        return self.actor_reference.probabilities(self.model, decisions)

    def _update_contract(self):
        plain = self.config.optimizer_rule == "pdf_gradient_v1"
        return {"optimizer_rule": self.config.optimizer_rule,
                "actor_learning_rate": (self.config.actor_learning_rate if plain else None) or self.config.learning_rate,
                "value_learning_rate": (self.config.value_learning_rate if plain else None) or self.config.learning_rate,
                "gradient_clipping": ("independent_actor_critic_registered_extension" if
                    self.config.actor_max_grad_norm or self.config.value_max_grad_norm else "disabled")
                    if self.config.optimizer_rule == "pdf_gradient_v1" else "legacy_joint_trainable_gradient_norm",
                "actor_max_grad_norm": self.config.actor_max_grad_norm if plain else None,
                "value_max_grad_norm": self.config.value_max_grad_norm if plain else None,
                "kl_reference": self.config.kl_reference,
                "kl_reference_identity": self.actor_reference.identity if self.actor_reference is not None else None,
                "kl_direction": "current_to_reference", "kl_estimator": "full_recorded_support_sum",
                "behavior_reference_local_gradient": "zero_at_exact_collection_snapshot"
                    if self.config.kl_reference == "behavior_snapshot" else None}

    def _probabilities(self, decision: DecisionRecord, values=None):
        before = decision.support_digest
        if values is None:
            values = self.model.decision_logits(decision)
        u = torch.tensor(decision.exploration_probabilities, device=values.device) if decision.exploration_probabilities else None
        probs = mixture_probabilities(values, decision.temperature, decision.exploration, u)
        if before != decision.support_digest:
            raise RuntimeError("Recorded feasible support changed during actor recomputation")
        return values, probs

    def _actor_terms(self, decision, advantage, values=None, reference=None):
        # Eq. (19), where vartheta=(theta,psi), defines the typed probability:
        #   p_vartheta(y_{g,j}|H_{g,j},Y_{g,j})
        #     = pi_psi(omega_t|S_t,R_t,Omega_t), for a sampled plan,
        #     = pi_theta(y_{g,j}|H_{g,j},Y_{g,j}), for a sampled edit token.
        # _probabilities applies the stored temperature, mask/support and
        # exploration law to current logits. For a plan, Y=Omega_t; for a
        # token, H contains its real parent, observed history and exact prefix.
        # Eq. (20)'s rho=p_current(y|H,Y)/p_behavior(y|H,Y) uses the stored
        # behavior_logprob, with the same H and Y in numerator and denominator.
        # _replay_units may batch token forwards, but this function still gives
        # each actually sampled token exactly one likelihood and one A_g factor.
        # Deterministic records and every forced suffix are excluded by callers.
        if self.config.kl_weight and self.config.kl_reference == "frozen_sft_actor" and reference is None:
            if values is not None:
                raise ValueError("Reference replay must precede the live token forward")
            reference = self._reference_rows((decision,))[0]
        _, probs = self._probabilities(decision, values)
        logprob = probs[decision.selected_index].log()
        old_logprob = logprob.new_tensor(decision.behavior_logprob)
        if self.config.estimator == "rloo":
            actor = -(logprob - old_logprob).exp() * advantage.detach()
        else:
            actor = -clipped_surrogate(logprob, old_logprob, advantage, self.config.clip_epsilon)
        kl = actor.new_zeros(())
        if self.config.kl_weight:
            if self.config.kl_reference == "behavior_snapshot":
                if not decision.behavior_probabilities:
                    raise ValueError("Exact behavior-reference KL requires full stored behavior probabilities")
                reference = probs.new_tensor(decision.behavior_probabilities)
            kl = current_reference_kl(probs, reference)
        return actor, kl

    def _replay_units(self, decisions):
        """Group only exact contiguous token prefixes; plans stay separate."""
        chain = []
        for decision in decisions:
            can_batch = decision.kind == "edit_token" and hasattr(self.model, "edit_token_logits")
            if chain:
                previous = chain[-1]
                follows = (can_batch and decision.transition_index == previous.transition_index and
                    decision.input_ids == previous.input_ids + (int(previous.support[previous.selected_index]),))
                if not follows:
                    yield tuple(chain)
                    chain = []
            if can_batch:
                chain.append(decision)
            else:
                yield (decision,)
        if chain:
            yield tuple(chain)

    def _audits(self, audits):
        if any(a.audit_only for a in audits):
            raise ValueError("Untouched audit examples cannot be training data")
        completed = [a for a in audits if a.completed]
        protocols = {a.protocol_id for a in completed}
        if len(protocols) > 1 or (self.config.checker_protocol_id and protocols - {self.config.checker_protocol_id}):
            raise ValueError("Surface labels must match one declared checker version/source policy/budget protocol")
        if protocols and not self.config.checker_protocol_id:
            self.config.checker_protocol_id = next(iter(protocols))
        return completed

    def _step(self):
        parameters = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        roles = {role: _empty_role_evidence() for role in _ROLES}
        if parameters:
            # Clone only parameters the optimizer will touch, never the frozen
            # backbone. Observe actual storage changes, including BF16 rounding.
            before = [p.detach().clone() for p in parameters]
            norms = defaultdict(list)
            for p in parameters:
                norms[self._parameter_roles[id(p)]].append(torch.linalg.vector_norm(p.grad.detach().float()))
            for role, values in norms.items():
                magnitude = float(torch.linalg.vector_norm(torch.stack(values)))
                roles[role].update(gradient_steps=1, nonzero_gradient_steps=int(magnitude > 0), gradient_l2_max=magnitude)
            if self.config.optimizer_rule == "pdf_gradient_v1":
                if any(not torch.isfinite(p.grad).all() for p in parameters):
                    raise ValueError("Nonfinite gradient in Eq23 update")
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters]))
                # Eq23 uses disjoint parameter assignments. A critic's scale
                # never rescales actor gradients, or vice versa. Zero means no
                # clipping; positive thresholds are explicit extensions.
                for update_role, maximum in (("actor", self.config.actor_max_grad_norm),
                                             ("value", self.config.value_max_grad_norm)):
                    selected = [p for p in parameters if
                        (self._parameter_roles[id(p)] in {"draft", "policy"} if update_role == "actor"
                         else self._parameter_roles[id(p)] == "value")]
                    if selected and maximum:
                        torch.nn.utils.clip_grad_norm_(selected, maximum, error_if_nonfinite=True)
            else:
                norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self.steps += 1
            if (self.cuda_memory_evidence is None and getattr(self.model, "backend", None) == "hf"
                    and parameters[0].device.type == "cuda"):
                device = parameters[0].device
                self.cuda_memory_evidence = {"process_id": os.getpid(), "device": str(device),
                    "optimizer_step": self.steps, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "memory_allocated_bytes": torch.cuda.memory_allocated(device),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(device),
                    "scope": "This process's CUDA allocator through its first actual HF optimizer step, including model loading and diagnostic parameter copies; counters were not reset. Other service processes are excluded."}
            changed, signal_changed, zero_changed = defaultdict(list), defaultdict(list), defaultdict(list)
            for p, original in zip(parameters, before):
                role = self._parameter_roles[id(p)]
                different = p.detach() != original
                nonzero = p.grad.detach() != 0
                changed[role].append(different.sum())
                signal_changed[role].append((different & nonzero).sum())
                zero_changed[role].append((different & ~nonzero).sum())
            for role, values in changed.items():
                count = int(torch.stack(values).sum())
                roles[role].update(changed_elements=count, parameter_change_steps=int(count > 0),
                    signal_parameter_change_steps=int(int(torch.stack(signal_changed[role]).sum()) > 0),
                    zero_gradient_change_steps=int(int(torch.stack(zero_changed[role]).sum()) > 0))
        else:
            norm = torch.tensor(0.)
        self._update_events.append({"schema": "role_update_evidence_v1", "roles": roles,
                                    "optimizer_steps": int(bool(parameters))})
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def _validate_frozen_text(self):
        if getattr(getattr(self.model, "config", None), "architecture_version", 2) == 3:
            from .frozen_text import node_text_cache
            node_text_cache(self.model).validate_identity()

    def warmup_epoch(self, *, audits: Sequence[CheckerAudit] = (), plans: Sequence[DecisionRecord] = (),
                     trajectories: Sequence[SearchTrajectory] = (), edits: Sequence[EditExample] = (),
                     pairs: Sequence[PairwiseValueExample] = (), _record_evidence=True):
        """Eq. (18): initialize edit generation and legal work-plan selection.

            L_SFT(theta,psi)
              = -E_{(H,e)~D_edit^sup}[log pi_theta(e|H)]
                -E_{(S,R,Omega,omega)~D_plan^sup}
                   [log pi_psi(omega|S,R,Omega)].
        Each expectation is its own empirical dataset mean. If N_edit and
        N_plan denote example counts, this is -sum_e log pi_theta(e|H)/N_edit
        -sum_plan log pi_psi(omega|S,R,Omega)/N_plan. An edit likelihood is the
        product of its autoregressive token probabilities, including sampled
        EOS; edit_nll returns the negative SUM of those log probabilities.
        Therefore tokens inside one edit are not averaged as separate examples.

        ``edits`` contains actually executed legal demonstrations with their
        real parent contexts. ``plans`` contains demonstrated choices in their
        legal paid menus. Unexecuted options have no supervised label. The
        scorer is separately initialized by Eq. (11)'s observed suffix targets.
        The mixed minibatch loop uses len(units)/counts[kind], then divides by
        batch size, to weight uniform mixed draws as these component means.
        Explicit component weights multiply the corresponding objectives;
        checker-surface supervision is a separate optional auxiliary term.
        """
        update_start = len(self._update_events)
        completed = self._audits(audits)
        if any(t.audit_only for t in trajectories) or any(e.audit_only for e in edits) or any(p.audit_only for p in pairs):
            raise ValueError("Final audit data must remain untouched")
        units = [("surface", a) for a in completed] + [("plan", p) for p in plans] + [("edit", e) for e in edits]
        units.extend(("pair", p) for p in pairs)
        for trajectory in trajectories:
            units.extend(("value", (d, trajectory.target(d, self.config.cost_coefficient, self.config.time_coefficient)))
                         for d in trajectory.decisions if d.kind in {"outer", "batch", "plan"})
        if not units:
            raise ValueError("Warm-up needs verified edits, completed audits, legal traces, or observed returns")
        # Use deterministic neural evaluations even while obtaining gradients;
        # dropout would otherwise alter recomputed policy likelihoods.
        # HFRoleModel enables block checkpointing locally without enabling dropout.
        self._validate_frozen_text()
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        random.Random(self.config.seed + self.steps).shuffle(units)
        counts = {kind: sum(k == kind for k, _ in units) for kind in ("surface", "plan", "edit", "value", "pair")}
        totals = defaultdict(float)
        for start in range(0, len(units), self.config.batch_size):
            batch = units[start:start + self.config.batch_size]
            for kind, item in batch:
                if not getattr(self.config, f"{kind}_weight"):
                    continue
                if kind == "surface":
                    logits = self.model.surface_logits(item.features, item.context_text)
                    loss = F.binary_cross_entropy_with_logits(logits, logits.new_tensor(float(item.label)))
                elif kind == "plan":
                    _, probs = self._probabilities(item)
                    loss = -probs[item.selected_index].log()
                elif kind == "edit":
                    if not hasattr(self.model, "edit_nll"):
                        raise ValueError("Token edit demonstrations require HFRoleModel")
                    loss = self.model.edit_nll(item)
                elif kind == "pair":
                    loss = self._pair_loss(item)
                else:
                    decision, target = item
                    values = self.model.decision_values(decision)
                    prediction = values[decision.selected_index]
                    loss = regression_loss(prediction, prediction.new_tensor(target), self.config.regression)
                totals[kind] += float(loss.detach())
                # Uniformly sampled mixed examples get inverse-frequency weight,
                # making each declared component an empirical mean over its set.
                weight = getattr(self.config, f"{kind}_weight") * len(units) / counts[kind]
                (weight * loss / len(batch)).backward()
            self._step()
        signal = merge_learning_signals(self._update_events[update_start:])
        schedule = {"name": "shuffled_component_minibatch_v1", "batch_size": self.config.batch_size,
            "component_counts": counts, "normalization": "inverse_frequency_empirical_component_means",
            "actual_optimizer_steps": signal["optimizer_steps"], **self._update_contract()}
        if _record_evidence:
            self.learning_history.append({"stage": "warmup", "update_schedule": schedule, "learning_signal": signal})
        return {"stage": "warmup", "steps": self.steps, "learning_signal": signal,
                "update_schedule": schedule,
                "first_hf_cuda_optimizer_step": self.cuda_memory_evidence, "completed_checker_labels": len(completed),
                "interrupted_checks_excluded": len(audits) - len(completed),
                "training_examples": counts,
                "trainable_edit_tokens": sum(sum(not d.deterministic for d in edit.token_decisions)
                    if edit.token_decisions else sum(label != -100 for label in edit.labels[1:]) for edit in edits),
                **{f"{kind}_loss": totals[kind] / n for kind, n in counts.items() if n}}

    def _pair_loss(self, pair):
        # Eq. (12): L_pair = E[(Q_phi(S,a)-Q_phi(S,b)-delta_hat_ab)^2],
        # delta_hat_ab = mean_s G_{a,s} - mean_s G_{b,s} (Eq. 22).
        # More explicitly, G_bar_a=(1/n_a)*sum_{s=1}^{n_a}G_{a,s} and
        # G_bar_b=(1/n_b)*sum_{s=1}^{n_b}G_{b,s}; delta_hat_ab=G_bar_a-G_bar_b.
        # This function returns one squared pair residual. The caller divides
        # by |D_pair|, independently of the |D_Q| normalization for value tuples.
        # Both arm terms use the SAME value_head, so pairs constrain relative action
        # values. L_Q is still needed: adding a common constant to every Q would
        # cancel here, leaving the absolute continuation-return scale unidentified.
        # PairwiseValueExample verifies the same paid observation, full menu,
        # plan features and graph masks on both arms. The value model predicts
        # all menu entries independently of selected_index, so one forward gives
        # both terms with exactly the same gradient as two identical forwards.
        # This halves scorer forwards for pairs and avoids a repeated frozen
        # text encode when the registered global encoding cache is disabled.
        predictions = self.model.decision_values(pair.left)
        left = predictions[pair.left.selected_index]
        right = predictions[pair.right.selected_index]
        return regression_loss(left - right, left.new_tensor(pair.difference), self.config.regression)

    def gvo_epoch(self, groups: Sequence[Sequence[SearchTrajectory]], *, audits: Sequence[CheckerAudit] = (),
                  forced_trajectories: Sequence[SearchTrajectory] = (), pairs: Sequence[PairwiseValueExample] = ()):
        """Eq. (4): apply the outer update after complete inner-search collection.

            (c_{k+1}^{(g)},D_k^{(g)}) = Search_{Theta_k}(c_k), g=1,...,G,
            D_k = union_{g=1}^{G} D_k^{(g)},
            Theta_{k+1} = GVO(Theta_k,D_k),
            Theta_k = (theta_k,psi_k,phi_k,nu),  bar_phi_k = phi_k.
        The caller has finished every inner search with all roles frozen before
        invoking this update. A group contains independent searches from one
        matching root, contract, budget and worker count, with output/cost logs
        recorded even when intermediate candidates remain incomplete. Multiple
        root groups retain separate leave-one-out baselines during one update.
        Their ordinary logs form the actor dataset and observed plan-start D_Q.

        Paid forced continuations add only scorer tuples and same-state pairs;
        none of their sampled suffix actions belongs to the actor union. Eq.
        (23) updates only Drafter theta, Policy psi and Scorer phi. Fixed menu
        settings nu, immutable task artifacts and evaluator outcomes are not
        optimizer parameters. Carrying or resetting the task root for the next
        round is distinct from ACCEPT's choice within a search. The next round
        freezes the refreshed roles and collects new trajectories under them.
        """
        if self.config.kl_weight and self.config.kl_reference == "frozen_sft_actor" and self.actor_reference is None:
            raise ValueError("Frozen SFT actor KL requires an identified reference before GVO")
        if self.actor_reference is not None:
            self.actor_reference.validate_model(self.model)
        if self.config.gvo_update_schedule == "legacy_group_then_forced_v1":
            return self._gvo_epoch_legacy(groups, audits=audits, forced_trajectories=forced_trajectories, pairs=pairs)
        return self._gvo_epoch_frozen(groups, audits=audits, forced_trajectories=forced_trajectories, pairs=pairs)

    def _gvo_epoch_frozen(self, groups, *, audits=(), forced_trajectories=(), pairs=()):
        """Evaluate every component at one snapshot, then take one update.

        Q tuples from ordinary and forced searches belong to the same empirical
        dataset. Their source and actor-group partition never change a tuple's
        regression weight. Pair examples form their own empirical mean. Across
        root groups the actor is a mean over *all ordinary searches*, preserving
        each group's leave-one-out baseline and each search's sum of choices.
        Eq. (20):
            rho_{g,j} = p_vartheta(y_{g,j}|H_{g,j},Y_{g,j})
                          / p_vartheta_k(y_{g,j}|H_{g,j},Y_{g,j}),
            L_GVO = -(1/G) sum_g sum_{j=1}^{J_g}
                    min(rho_{g,j}*A_g,
                        clip(rho_{g,j},1-epsilon_c,1+epsilon_c)*A_g).
        There is deliberately no ``1/J_g``: long and short searches share the
        same per-search coefficient. Ordinary searches with J_g=0 retain their
        place in G and their contribution to other members' Eq. (13) baselines.
        Eq. (11) averages all observed Q tuples, including forced suffixes:
            L_Q = (1/|D_Q|) sum_{(S,omega,G_t) in D_Q}
                    (Q_hat_phi(S,omega)-G_t)^2.
        Eq. (12) separately averages paid pairs:
            L_pair = (1/|D_pair|) sum_{(S,a,b,G_bar_a,G_bar_b) in D_pair}
                (Q_hat_phi(S,a)-Q_hat_phi(S,b)-(G_bar_a-G_bar_b))^2.
        With value_weight=1, pair_weight is beta_pair and the critic objective
        is L_value=L_Q+beta_pair*L_pair. An explicitly changed value_weight
        multiplies L_Q, while surface_weight controls a separate auxiliary
        checker loss. Neither coefficient rescales the actor's 1/G average.
        The forced records never receive actor likelihood. One backward per
        independent forward releases its graph, while all gradients accumulate
        at the same pre-update parameters until the final Eq. (23) optimizer
        step. No critic update changes the scorer partway through actor replay.

        ``batch_size`` applies only to supervised/explicit legacy minibatches;
        independent forward graphs here are released after each backward.
        """
        if not groups:
            raise ValueError("GVO needs collected groups of complete searches")
        completed = self._audits(audits)
        for group in groups:
            validate_group(group)
        if any(not t.forced or t.audit_only or t.demonstration for t in forced_trajectories):
            raise ValueError("Critic-only continuations must be explicitly forced and outside demonstrations/final audit")
        if any(p.audit_only for p in pairs):
            raise ValueError("Final audit pairs must remain untouched")
        ordinary = [trajectory for group in groups for trajectory in group]
        # Empty IDs are supported only for wholly unspecified legacy data.
        # Dropping them would silently mix an identified continuation policy
        # with labels whose continuation snapshot is unknown.
        snapshots = {t.collection_snapshot for t in ordinary + list(forced_trajectories)}
        snapshots.update(pair.snapshot_id for pair in pairs)
        if len(snapshots) > 1:
            raise ValueError("One frozen collection update requires a matched continuation snapshot for every Q/pair label")
        q_tuples = []
        q_counts = {"ordinary": 0, "forced": 0}
        for source, trajectories in (("ordinary", ordinary), ("forced", forced_trajectories)):
            for trajectory in trajectories:
                for decision in trajectory.decisions:
                    if decision.kind not in {"outer", "batch", "plan"}:
                        continue
                    target = (trajectory.item_return(self.config.cost_coefficient, decision)
                              if self.config.reward_mode == "completed_items" else
                              trajectory.target(decision, self.config.cost_coefficient, self.config.time_coefficient))
                    q_tuples.append((source, decision, target))
                    q_counts[source] += 1
        self._validate_frozen_text()
        self.model.eval()
        update_start = len(self._update_events)
        self.optimizer.zero_grad(set_to_none=True)
        actor_total = value_total = pair_total = kl_total = surface_total = 0.
        value_by_source = dict.fromkeys(q_counts, 0.)
        all_returns, dispersion = [], []
        decisions_count = 0
        for group in groups:
            returns = torch.tensor([(t.item_return(self.config.cost_coefficient) if self.config.reward_mode == "completed_items"
                else t.full_return(self.config.cost_coefficient, self.config.time_coefficient)) for t in group],
                # Eq. (13) subtracts whole-search costs before comparing small
                # gains. Keep observed labels in float64: float32 turns returns
                # [-1e8, -1e8+1] into equal values and falsely disables learning.
                # Model tensors remain in their registered training dtype.
                dtype=torch.float64, device=self.model.device)
            advantages = group_advantages(returns, self.config.estimator)
            loo = group_advantages(returns, "gvo")
            dispersion.append({"searches": len(group), "return_min": float(returns.min()),
                "return_max": float(returns.max()), "return_variance": float(returns.var(unbiased=False)),
                "loo_advantage_abs_max": float(loo.abs().max()), "loo_nonzero_advantages": int(torch.count_nonzero(loo)),
                "objective_nonzero_advantages": int(torch.count_nonzero(advantages))})
            all_returns.extend(returns.tolist())
            for trajectory, advantage in zip(group, advantages):
                for unit in self._replay_units(trajectory.decisions):
                    if all(decision.deterministic for decision in unit):
                        continue
                    references = self._reference_rows(unit)
                    if unit[0].kind == "edit_token" and hasattr(self.model, "edit_token_logits"):
                        rows = self.model.edit_token_logits(unit)
                    else:
                        rows = (None,) * len(unit)
                    losses = []
                    for decision, row, reference in zip(unit, rows, references):
                        if decision.deterministic:
                            continue
                        actor, kl = self._actor_terms(decision, advantage, row, reference)
                        losses.append((actor + self.config.kl_weight * kl) / len(ordinary))
                        actor_total += float(actor.detach()) / len(ordinary)
                        kl_total += float(kl.detach()) / len(ordinary)
                        decisions_count += 1
                    loss = torch.stack(losses).sum()
                    if loss.requires_grad:
                        loss.backward()
        if self.config.value_weight:
            for source, decision, target in q_tuples:
                prediction = self.model.decision_values(decision)[decision.selected_index]
                value = regression_loss(prediction, prediction.new_tensor(target), self.config.regression)
                value_total += float(value.detach()) / len(q_tuples)
                value_by_source[source] += float(value.detach()) / q_counts[source]
                loss = self.config.value_weight * value / len(q_tuples)
                if loss.requires_grad:
                    loss.backward()
        if self.config.pair_weight:
            for pair in pairs:
                value = self._pair_loss(pair)
                pair_total += float(value.detach()) / len(pairs)
                loss = self.config.pair_weight * value / len(pairs)
                if loss.requires_grad:
                    loss.backward()
        if self.config.surface_weight:
            for audit in completed:
                logits = self.model.surface_logits(audit.features, audit.context_text)
                surface = F.binary_cross_entropy_with_logits(logits, logits.new_tensor(float(audit.label)))
                surface_total += float(surface.detach()) / len(completed)
                loss = self.config.surface_weight * surface / len(completed)
                if loss.requires_grad:
                    loss.backward()
        # Eq. (23) applies the accumulated derivatives only after every observed
        # actor/Q/pair component has been evaluated at (vartheta_k,phi_k).
        # The two parameter sets receive their own learning rates. The next
        # collection, rather than any remaining sample in this one, receives
        # the refreshed (theta_{k+1},psi_{k+1},bar_phi_{k+1}=phi_{k+1},nu).
        self._step()
        signal = merge_learning_signals(self._update_events[update_start:])
        # A shared optimizer step cannot honestly be attributed as separate
        # ordinary/forced parameter updates. Source counts describe objectives;
        # the existing role evidence describes actual joint storage changes.
        signal["joint_collection_updates"] = dict(signal)
        signal["group_dispersion"] = dispersion
        signal["groups_with_zero_loo_advantage"] = sum(row["loo_nonzero_advantages"] == 0 for row in dispersion)
        schedule = {"name": "frozen_collection_v1", "evaluation_point": "one_preupdate_parameter_snapshot",
            "actual_optimizer_steps": signal["optimizer_steps"], "maximum_optimizer_steps": 1,
            "actor_normalization": "sum_sampled_choices_mean_all_ordinary_searches",
            "actor_search_count": len(ordinary), "actor_group_sizes": [len(group) for group in groups],
            "actor_sampled_choices": decisions_count, "forced_actor_choices": 0,
            "q_normalization": "empirical_mean_all_ordinary_and_forced_plan_starts",
            "q_tuple_counts": {**q_counts, "total": len(q_tuples)},
            "q_source_mass": {source: count / len(q_tuples) if q_tuples else 0. for source, count in q_counts.items()},
            "pair_normalization": "empirical_mean_pair_examples", "pair_count": len(pairs),
            "loss_weights": {key: getattr(self.config, key + "_weight") for key in ("value", "pair", "surface", "kl")},
            "batch_size_applies_to": "warmup_and_explicit_legacy_schedule_only",
            "collection_snapshots": sorted(snapshots), **self._update_contract()}
        self.learning_history.append({"stage": "gvo", "update_schedule": schedule, "learning_signal": signal})
        return {"stage": "gvo", "estimator": self.config.estimator, "reward_mode": self.config.reward_mode,
            "steps": self.steps, "update_schedule": schedule, "learning_signal": signal,
            "first_hf_cuda_optimizer_step": self.cuda_memory_evidence,
            "actor_loss": actor_total, "value_loss": value_total, "value_loss_by_source": value_by_source,
            "pair_loss": pair_total, "surface_loss": surface_total, "kl": kl_total,
            "mean_search_return": sum(all_returns) / len(all_returns), "groups": len(groups),
            "searches": len(ordinary), "stochastic_decisions": decisions_count,
            "forced_searches": len(forced_trajectories), "pair_examples": len(pairs)}

    def _gvo_epoch_legacy(self, groups, *, audits=(), forced_trajectories=(), pairs=()):
        if not groups:
            raise ValueError("GVO needs collected groups of complete searches")
        completed = self._audits(audits)
        if any(not t.forced or t.audit_only for t in forced_trajectories):
            raise ValueError("Critic-only continuations must be explicitly forced and outside the final audit")
        if any(p.audit_only for p in pairs):
            raise ValueError("Final audit pairs must remain untouched")
        self._validate_frozen_text()
        self.model.eval()
        update_start = len(self._update_events)
        actor_total = value_total = kl_total = 0.
        all_returns = []
        dispersion = []
        decisions_count = 0
        for group in groups:
            validate_group(group)
            self.optimizer.zero_grad(set_to_none=True)
            returns = torch.tensor([(t.item_return(self.config.cost_coefficient) if self.config.reward_mode=='completed_items'
                                     else t.full_return(self.config.cost_coefficient, self.config.time_coefficient)) for t in group], dtype=torch.float64, device=self.model.device)
            advantages = group_advantages(returns, self.config.estimator)
            loo = group_advantages(returns, "gvo")
            dispersion.append({"searches": len(group), "return_min": float(returns.min()),
                "return_max": float(returns.max()), "return_variance": float(returns.var(unbiased=False)),
                "loo_advantage_abs_max": float(loo.abs().max()),
                "loo_nonzero_advantages": int(torch.count_nonzero(loo)),
                "objective_nonzero_advantages": int(torch.count_nonzero(advantages))})
            all_returns.extend(returns.tolist())
            value_count = sum(d.kind in {"outer", "batch", "plan"} for t in group for d in t.decisions)
            for trajectory, advantage in zip(group, advantages):
                for unit in self._replay_units(trajectory.decisions):
                    references = self._reference_rows(unit) if any(not d.deterministic for d in unit) else (None,) * len(unit)
                    if unit[0].kind == "edit_token" and hasattr(self.model, "edit_token_logits"):
                        if all(decision.deterministic for decision in unit):
                            continue
                        rows = self.model.edit_token_logits(unit)
                        token_losses = []
                        for decision, row, reference in zip(unit, rows, references):
                            if decision.deterministic:
                                continue
                            actor, kl = self._actor_terms(decision, advantage, row, reference)
                            token_losses.append((actor + self.config.kl_weight * kl) / len(group))
                            actor_total += float(actor.detach()) / len(group)
                            kl_total += float(kl.detach()) / len(group)
                            decisions_count += 1
                        # All rows share one causal graph, hence one backward.
                        torch.stack(token_losses).sum().backward()
                        continue
                    decision = unit[0]
                    if decision.kind=='edit_token' and decision.deterministic:
                        # Fixed grammar punctuation has probability one on its
                        # saved singleton support and no actor/value objective.
                        continue
                    loss = None
                    if not decision.deterministic:
                        actor, kl = self._actor_terms(decision, advantage, reference=references[0])
                        loss = (actor + self.config.kl_weight * kl) / len(group)  # Sum decisions, mean searches.
                        actor_total += float(actor.detach()) / len(group)
                        kl_total += float(kl.detach()) / len(group)
                        decisions_count += 1
                    if decision.kind in {"outer", "batch", "plan"} and self.config.value_weight:
                        values = self.model.decision_values(decision)
                        prediction = values[decision.selected_index]
                        target = prediction.new_tensor(trajectory.item_return(self.config.cost_coefficient, decision)
                            if self.config.reward_mode=='completed_items' else trajectory.target(decision, self.config.cost_coefficient, self.config.time_coefficient))
                        value = regression_loss(prediction, target, self.config.regression)
                        weighted_value = self.config.value_weight * value / max(1, value_count)
                        loss = weighted_value if loss is None else loss + weighted_value
                        value_total += float(value.detach()) / max(1, value_count)
                    if loss is not None and loss.requires_grad:
                        loss.backward()
            for audit in completed:
                if not self.config.surface_weight:
                    continue
                logits = self.model.surface_logits(audit.features, audit.context_text)
                surface = F.binary_cross_entropy_with_logits(logits, logits.new_tensor(float(audit.label)))
                (self.config.surface_weight * surface / len(completed)).backward()
            self._step()
        critic_metrics = {}
        ordinary_signal = merge_learning_signals(self._update_events[update_start:])
        if forced_trajectories or pairs:
            critic_metrics = self.warmup_epoch(trajectories=forced_trajectories, pairs=pairs, _record_evidence=False)
        signal = merge_learning_signals(self._update_events[update_start:])
        signal["ordinary_search_updates"] = ordinary_signal
        signal["forced_critic_updates"] = critic_metrics.get("learning_signal")
        signal["group_dispersion"] = dispersion
        signal["groups_with_zero_loo_advantage"] = sum(row["loo_nonzero_advantages"] == 0 for row in dispersion)
        ordinary_counts = [sum(d.kind in {"outer", "batch", "plan"} for t in group for d in t.decisions)
                           for group in groups]
        forced_count = sum(d.kind in {"outer", "batch", "plan"} for t in forced_trajectories for d in t.decisions)
        schedule = {"name": "legacy_group_then_forced_v1", "actual_optimizer_steps": signal["optimizer_steps"],
            "actor_normalization": "sum_sampled_choices_mean_searches_separately_per_group_step",
            "actor_group_sizes": [len(group) for group in groups],
            "q_normalization": "ordinary_group_means_then_separate_forced_component_minibatches",
            "q_tuple_counts": {"ordinary": sum(ordinary_counts), "forced": forced_count,
                               "total": sum(ordinary_counts) + forced_count},
            "ordinary_group_q_counts": ordinary_counts,
            "pair_count": len(pairs), "forced_batch_size": self.config.batch_size,
            "source_weighting": "historical_source_dependent_schedule_not_the_uniform_combined_DQ_mean",
            **self._update_contract()}
        self.learning_history.append({"stage": "gvo", "update_schedule": schedule, "learning_signal": signal})
        return {"stage": "gvo", "estimator": self.config.estimator, "reward_mode":self.config.reward_mode, "steps": self.steps,
                "learning_signal": signal, "update_schedule": schedule,
                "first_hf_cuda_optimizer_step": self.cuda_memory_evidence,
                "actor_loss": actor_total / len(groups), "value_loss": value_total / len(groups),
                "kl": kl_total / len(groups), "mean_search_return": sum(all_returns) / len(all_returns),
                "groups": len(groups), "searches": len(all_returns), "stochastic_decisions": decisions_count,
                "forced_searches": len(forced_trajectories), "pair_examples": len(pairs),
                "pair_loss": critic_metrics.get("pair_loss", 0.)}

    @torch.no_grad()
    def evaluate(self, *, audits=(), trajectories=()):
        self.model.eval()
        complete = [a for a in audits if a.completed]
        if complete:
            logits = torch.stack([self.model.surface_logits(a.features, a.context_text) for a in complete])
            labels = logits.new_tensor([a.label for a in complete])
            calibration = binary_calibration(logits, labels)
        else:
            calibration = {"count": 0}
        returns = [t.full_return(self.config.cost_coefficient, self.config.time_coefficient) for t in trajectories]
        return {"surface_completed_population": calibration, "interrupted_checks": sum(not a.completed for a in audits),
                "completed_searches": len(returns), "mean_realized_return": sum(returns) / len(returns) if returns else None}

    def save(self, directory: str | Path):
        directory = Path(directory)
        self.model.save(directory)
        if self.actor_reference is not None:
            self.actor_reference.save(directory)
        (directory / "training_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        evidence = {"schema": "role_training_evidence_v1",
            "architecture_version": getattr(getattr(self.model, "config", None), "architecture_version", None),
            "inherited_evidence_status": self.inherited_evidence_status,
            "gvo_update_schedule": self.config.gvo_update_schedule,
            "schedule_restore_history": self.schedule_restore_history,
            "optimizer_restore_history": self.optimizer_restore_history,
            "update_contract": self._update_contract(),
            "actor_reference": self.actor_reference.manifest() if self.actor_reference is not None else None,
            "first_hf_cuda_optimizer_step": self.cuda_memory_evidence,
            "history": self.learning_history,
            "learning_signal": merge_learning_signals(row["learning_signal"] for row in self.learning_history),
            "gvo_learning_signal": merge_learning_signals(row["learning_signal"] for row in self.learning_history if row["stage"] == "gvo")}
        (directory / "training_evidence.json").write_text(json.dumps(evidence, indent=2))
        torch.save({"optimizer": self.optimizer.state_dict(), "optimizer_rule": self.config.optimizer_rule, "steps": self.steps,
                    "torch_rng_state": torch.get_rng_state()}, directory / "trainer.pt")

    def restore_optimizer(self, directory: str | Path, *, gvo_update_schedule_override: str | None = None,
                          optimizer_rule_override: str | None = None, kl_reference_override: str | None = None):
        directory = Path(directory)
        config_path = directory / "training_config.json"
        checkpoint_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        inherited_schedule = checkpoint_config.get("gvo_update_schedule", "legacy_group_then_forced_v1")
        schedule = inherited_schedule if gvo_update_schedule_override is None else gvo_update_schedule_override
        if inherited_schedule not in _GVO_SCHEDULES or schedule not in _GVO_SCHEDULES:
            raise ValueError("Unknown checkpoint or explicitly overridden GVO update schedule")
        self.config.gvo_update_schedule = schedule
        inherited_rule = checkpoint_config.get("optimizer_rule", "adamw_legacy_v1")
        rule = inherited_rule if optimizer_rule_override is None else optimizer_rule_override
        inherited_reference = checkpoint_config.get("kl_reference", "behavior_snapshot")
        reference = inherited_reference if kl_reference_override is None else kl_reference_override
        if (inherited_rule not in _OPTIMIZER_RULES or rule not in _OPTIMIZER_RULES
                or inherited_reference not in _KL_REFERENCES or reference not in _KL_REFERENCES):
            raise ValueError("Unknown checkpoint or explicitly overridden optimizer/KL rule")
        self.config.optimizer_rule, self.config.kl_reference = rule, reference
        self.config.__post_init__()
        state = torch.load(directory / "trainer.pt", map_location=self.model.device, weights_only=True)
        if state.get("optimizer_rule", "adamw_legacy_v1") != inherited_rule:
            raise ValueError("Optimizer state disagrees with the checkpoint rule")
        self.optimizer = self._make_optimizer()
        if inherited_rule == rule:
            self.optimizer.load_state_dict(state["optimizer"])
            if rule == "pdf_gradient_v1" and any(
                    group.get("momentum", 0) or group.get("dampening", 0) or group.get("nesterov", False)
                    or group.get("weight_decay", 0) for group in self.optimizer.param_groups):
                raise ValueError("Eq23 optimizer checkpoint contains unregistered momentum or decay")
        # An explicitly registered AdamW -> Eq23 transition discards moments;
        # loading them into a plain-gradient optimizer would change Eq23.
        # A stage transition may prospectively register a different rate/decay.
        # Preserve optimizer moments without silently restoring the old schedule.
        for group in self.optimizer.param_groups:
            update_role = group.get("update_role")
            group["lr"] = ((self.config.actor_learning_rate if update_role == "actor" else
                            self.config.value_learning_rate if update_role == "value" else None)
                           or self.config.learning_rate)
            group["weight_decay"] = self.config.weight_decay if rule == "adamw_legacy_v1" else 0.
        self.steps = state["steps"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        evidence = directory / "training_evidence.json"
        recorded = {}
        if evidence.exists():
            recorded = json.loads(evidence.read_text())
            if recorded.get("schema") != "role_training_evidence_v1":
                raise ValueError("Unknown training evidence schema")
            self.learning_history = recorded["history"]
            self.inherited_evidence_status = recorded.get("inherited_evidence_status", "recorded")
            self.schedule_restore_history = list(recorded.get("schedule_restore_history", ()))
            self.optimizer_restore_history = list(recorded.get("optimizer_restore_history", ()))
        else:
            self.inherited_evidence_status = "legacy_checkpoint_without_role_update_evidence"
        self.schedule_restore_history.append({"checkpoint_schedule": inherited_schedule,
            "missing_checkpoint_schedule_uses_legacy": "gvo_update_schedule" not in checkpoint_config,
            "explicit_override": gvo_update_schedule_override, "resolved_schedule": schedule,
            "optimizer_steps_at_restore": self.steps})
        self.optimizer_restore_history.append({"checkpoint_rule": inherited_rule,
            "missing_checkpoint_rule_uses_legacy": "optimizer_rule" not in checkpoint_config,
            "explicit_override": optimizer_rule_override, "resolved_rule": rule,
            "state_transfer": "preserved" if inherited_rule == rule else "discarded_on_registered_rule_change",
            "checkpoint_kl_reference": inherited_reference, "explicit_kl_reference_override": kl_reference_override,
            "resolved_kl_reference": reference, "optimizer_steps_at_restore": self.steps})
        manifest_path, weights_path = directory / "actor_reference.json", directory / "actor_reference.pt"
        self.actor_reference = None
        if manifest_path.exists() != weights_path.exists():
            raise ValueError("Incomplete frozen actor reference checkpoint")
        if manifest_path.exists():
            from .reference_policy import FrozenActorReference
            self.actor_reference = FrozenActorReference.load(directory, self.model)
            if recorded.get("actor_reference") != self.actor_reference.manifest():
                raise ValueError("Training evidence disagrees with frozen actor reference identity")
        elif evidence.exists() and recorded.get("actor_reference") is not None:
            raise ValueError("Checkpoint lost its registered frozen actor reference")
