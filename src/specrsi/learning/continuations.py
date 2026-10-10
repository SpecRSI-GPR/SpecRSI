"""Paid same-state Monte Carlo acquisition for explicitly isolated runtimes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from itertools import combinations
import json
import hashlib
import math
from pathlib import Path
import random
import time
from urllib.parse import urlparse

from .records import PairwiseValueExample


@dataclass(frozen=True)
class ForcedCollectionConfig:
    repetitions: int = 0
    state_indices: tuple[int, ...] = (0,)
    arm_count: int = 2
    arm_rule: str = "canonical"

    @classmethod
    def from_settings(cls, settings, *, teacher=False):
        config = cls(settings.get("forced_repetitions", 0), tuple(settings.get("forced_state_indices", (0,))),
                     settings.get("forced_arm_count", 2), settings.get("forced_arm_rule", "canonical"))
        if type(config.repetitions) is not int or config.repetitions < 0:
            raise ValueError("forced_repetitions must be a prospectively fixed nonnegative integer")
        if type(config.arm_count) is not int or config.arm_count < 2:
            raise ValueError("forced_arm_count must register at least two legal arms")
        if not config.state_indices or any(type(i) is not int or i < 0 for i in config.state_indices) or len(set(config.state_indices)) != len(config.state_indices):
            raise ValueError("forced_state_indices must be distinct nonnegative paid-choice indices")
        if config.arm_rule not in {"canonical", "non_accept_first"}:
            raise ValueError("Unknown prospective forced-arm selection rule")
        if teacher and config.repetitions:
            raise ValueError("Demonstration warm-up cannot collect forced GVO continuations; set forced_repetitions=0")
        return config


class FrozenRoleGuard:
    """Keep Eq. (4)'s entire inner collection under one fixed improver.

    Theta_k=(theta_k,psi_k,phi_k,nu), with bar_phi_k=phi_k during collection.
    An in-place optimizer/parameter update before ordinary and forced searches
    finish would mix continuation policies in D_Q and invalidate stored actor
    behavior probabilities. Parameter object identities and mutation counters
    therefore stay fixed until the external Eq. (23) update.
    """
    def __init__(self, model):
        self.model = model
        self.signature = self._signature()

    def _signature(self):
        named = getattr(self.model, "named_parameters", None)
        return tuple((name, id(parameter), parameter._version) for name, parameter in named()) if named else ()

    def check(self):
        if self.signature != self._signature():
            raise RuntimeError("Collection roles changed before all ordinary/forced searches completed")


def role_snapshot_id(model, label):
    """Name a collection by actual role weights and the frozen-base identity.

    The full multi-billion-parameter frozen backbone is identified by its
    registered model configuration/revision; only trainable role tensors are
    transferred for hashing. Parameter versions separately guard all weights.
    Lightweight test doubles without model parameters retain their test label.
    """
    named = getattr(model, "named_parameters", None)
    if named is None:
        return label
    import torch
    config = model.config
    config = asdict(config) if is_dataclass(config) else config.to_dict() if hasattr(config, "to_dict") else vars(config)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode())
    for name, parameter in named():
        if not parameter.requires_grad:
            continue
        digest.update(json.dumps((name, list(parameter.shape), str(parameter.dtype)), separators=(",", ":")).encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return f"{label}:sha256:{digest.hexdigest()}"


def require_local_prompt_forks(executor_class, settings, health):
    """Only the inspected local request-isolated Qwen path has this adapter.

    Each request supplies its full messages and contract-fixed evaluation seed;
    the local server resets its RNG and serializes inference. A new HTTP client
    therefore carries no conversational/tool state between suffixes.
    """
    from specrsi.experiments import QwenExecutor
    host = urlparse(settings["executor"]["base_url"]).hostname
    if executor_class is not QwenExecutor or settings.get("task_pipeline") or "initial_workflow" in settings:
        raise ValueError("Forced prompt continuations require the plain local QwenExecutor adapter; repositories, tools and workflows need an explicit restore adapter")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Forced prompt continuations require the inspected local stateless inference service")
    if health.get("protocol_version", 0) < 2 or not health.get("server_enforces_total_token_cap") or not health.get("revision"):
        raise ValueError("Local forced execution requires the locked metered protocol-v2 service")


def _validate_executor(executor, contract, snapshot):
    from specrsi.core.execution import PureIntegerExecutor
    from specrsi.experiments import QwenExecutor
    from specrsi.science.executor import ScienceExecutor
    classes = {"pure_integer": PureIntegerExecutor, "local_qwen_prompt": QwenExecutor, "native_science": ScienceExecutor}
    if contract not in classes or type(executor) is not classes[contract]:
        raise ValueError("No proven independent-restore adapter for this forced execution environment")
    if contract == "local_qwen_prompt":
        if any("workflow" in candidate.artifacts for candidate in snapshot.state.candidates.values()):
            raise ValueError("Workflow state cannot be restored by the plain prompt adapter")
        expected = snapshot.state.incumbent.runtime["model_revision"]
        health = executor.client.health()
        if health.get("revision") != expected or health.get("protocol_version", 0) < 2 or not health.get("server_enforces_total_token_cap"):
            raise ValueError("The frozen task-model service changed before a forced continuation")
    if snapshot.resume_tokens:
        raise ValueError("The selected stateless runtime cannot restore opaque external resume tokens")


class _RecordedExecutor:
    def __init__(self, executor):
        self.executor = executor
        self.events = []

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def run(self, candidate, task, scope, cap, resume_token=None):
        from specrsi.core.models import thaw
        started = time.monotonic()
        try:
            result = self.executor.run(candidate, task, scope, cap, resume_token=resume_token)
        except Exception as exc:
            self.events.append({"candidate": candidate.uid, "task": task.uid, "scope": scope.uid,
                "status": "missing_return", "elapsed_seconds": time.monotonic() - started,
                "cost": None, "error": str(exc)})
            raise
        self.events.append({"candidate": candidate.uid, "task": task.uid, "scope": scope.uid,
            "status": result.status, "cost": result.actual_cost, "cap": cap,
            "elapsed_seconds": time.monotonic() - started, "receipt": thaw(result.receipt), "error": result.error})
        return result


def collect_paid_continuations(snapshots, *, model, snapshot_id, bridge_factory, executor_factory,
                               drafter_factory, checker_factory, environment_contract, config: ForcedCollectionConfig,
                               seed, lineage_id, training, guard=None, acquisition_path=None):
    """Execute registered legal arms with independent state and RNG per repeat.

    Appendix F.5, Eq. (22) and Algorithm 3, lines 7--14:
        G_bar_a = (1/n_a) sum_{s=1}^{n_a} G_{a,s},
        delta_hat_ab = G_bar_a-G_bar_b,
        Var(delta_hat_ab | S) = sigma_a^2/n_a + sigma_b^2/n_b,
        sigma_a^2 = Var(G_{a,s} | S,a,Theta_k).
    The variance identity requires independent repetitions and independent
    arms conditional on S and Theta_k. Coupled arm sampling instead subtracts
    2*Cov(G_bar_a,G_bar_b | S); matching the starting state alone does not imply
    that this covariance is zero or that variance decreases. The collector
    executes repetitions and stores observed means/differences; the variance
    equation states the statistical assumptions, rather than a computed label.
    ``config.repetitions`` fixes n_a before any arm outcome is known. Distinct
    RNG seeds produce independent suffixes under the same immutable Theta_k;
    each arm begins after the *same already paid* solver/menu construction.

    Clone setup belongs to acquisition, not to the saved counterfactual budget.
    The first menu is sunk; subsequent preparation, failed work and terminal fees
    remain in every actual suffix. Missing rollouts abort the batch without a
    fabricated label, with partial charges retained in an append-only journal.
    Each G_{a,s} is the Eq. (11) observed return from the forced first plan to
    termination: F(S_end)-F(S)-lambda*C_suffix-mu*L_suffix. Forcing only the
    first plan does not authorize actor replay of later stochastic choices:
    the entire forced trajectory contributes to D_Q/D_pair only. Copies are
    charged once to acquisition, without another subtraction from the saved
    counterfactual continuation budget. Fully logged unsuccessful repetitions
    remain in their prospectively fixed n_a denominator.
    """
    from specrsi.core.vats import VATSEngine

    guard = guard or FrozenRoleGuard(model)
    output = {"forced_trajectories": [], "pairs": [], "physical": [], "metadata": {
        "environment_contract": environment_contract, "registered_repetitions": config.repetitions,
        "state_indices": list(config.state_indices), "arm_count": config.arm_count, "arm_rule": config.arm_rule,
        "forced_search_cost": 0., "forced_copy_elapsed_seconds": 0., "forced_copy_cost": 0,
        "raw_search_results": [], "acquisition_records": [], "skipped_states": [],
        "copy_charge_location": "acquisition_only_not_counterfactual_suffix_budget",
        "randomness": "independent_proposer_policy_streams; evaluator_replay_variables_fixed_by_contract"}}
    metadata = output["metadata"]
    if not config.repetitions:
        return output

    def log(event):
        metadata["acquisition_records"].append(event)
        if acquisition_path is not None:
            path = Path(acquisition_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    rng = random.Random(seed)
    used_seeds, executors, bridges, drafters = set(), [], [], []
    model.eval()
    for state_index in config.state_indices:
        if state_index >= len(snapshots):
            metadata["skipped_states"].append({"state_index": state_index, "reason": "registered paid state was not visited"})
            continue
        snapshot = snapshots[state_index]
        state_id = snapshot.uid
        forbidden_splits = {"audit", "test", "selection", "heldout"}
        if snapshot.state.scope.split in forbidden_splits or any(task.metadata.get("split") in forbidden_splits for task in snapshot.state.scope.tasks):
            raise ValueError("Paid training continuations must come from development, never final audit or selection")
        if not snapshot.config.all_u or snapshot.config.max_sources:
            raise ValueError("This forced runtime adapter supports the PDF direct-execution contract only")
        if {snapshot.config.scorer_snapshot, snapshot.config.policy_snapshot} != {snapshot_id}:
            raise ValueError("Forced arms must retain the paid state's exact scorer and policy snapshot")
        plans = sorted(snapshot.plans, key=(lambda p: (p.kind == "accept", p.uid))
                       if config.arm_rule == "non_accept_first" else (lambda p: p.uid))[:config.arm_count]
        if len(plans) < 2:
            metadata["skipped_states"].append({"state_index": state_index, "reason": "fewer than two legal arms"})
            continue
        plan_returns, plan_seeds, decisions = {}, {}, {}
        log({"event": "registered", "paid_state_id": state_id, "snapshot_id": snapshot_id,
             "state_index": state_index, "arms": [plan.uid for plan in plans], "repetitions": config.repetitions})
        for plan in plans:
            plan_returns[plan.uid], plan_seeds[plan.uid] = [], []
            for repeat in range(config.repetitions):
                guard.check()
                if snapshot.uid != state_id:
                    raise RuntimeError("A previous forced branch mutated the saved paid state")
                member_seed = rng.randrange(2**63 - 1)
                while member_seed in used_seeds:
                    member_seed = rng.randrange(2**63 - 1)
                used_seeds.add(member_seed)
                started = time.monotonic()
                engine = None
                bridge = executor = recorded = None
                try:
                    bridge = bridge_factory()
                    executor = executor_factory()
                    _validate_executor(executor, environment_contract, snapshot)
                    recorded = _RecordedExecutor(executor)
                    drafter = drafter_factory(bridge, recorded)
                    if any(executor is x for x in executors) or any(bridge is x for x in bridges) or any(drafter is x for x in drafters):
                        raise ValueError("Each forced suffix needs newly restored executor, bridge and drafter instances")
                    executors.append(executor); bridges.append(bridge); drafters.append(drafter)
                    if bridge.model is not model or bridge.snapshot_id != snapshot_id:
                        raise ValueError("Forced branch bridge does not use the frozen collection roles")
                    first = bridge._record(snapshot.state, snapshot.plans)
                    first = replace(first, selected_index=first.support.index(plan.uid), transition_index=0,
                                    deterministic=True, behavior_logprob=None)
                    engine = VATSEngine.from_paid_state(snapshot, plan, executor=recorded, drafter=drafter,
                        seed=member_seed, scorer=bridge.scorer, policy_menu=bridge.policy_menu)
                    checker = checker_factory()
                    if checker.version != snapshot.state.ledger.checker.version:
                        raise ValueError("Restored checker protocol differs from the frozen paid state")
                    engine.state.ledger.checker = checker
                    if engine.state.ledger is snapshot.state.ledger or engine.state.budget is snapshot.state.budget:
                        raise RuntimeError("Forced continuation failed to isolate the host ledger and budget")
                    copy_seconds = time.monotonic() - started
                    copy_cost = math.ceil(copy_seconds * 1000) if bridge.cost_unit == "wall-ms" else 0
                    metadata["forced_copy_elapsed_seconds"] += copy_seconds
                    metadata["forced_copy_cost"] += copy_cost
                    log({"event": "clone_acquisition", "paid_state_id": state_id, "arm": plan.uid,
                         "repeat": repeat, "seed": member_seed, "elapsed_seconds": copy_seconds, "cost": copy_cost,
                         "cost_unit": bridge.cost_unit, "counterfactual_budget_debit": 0})
                    bridge.initial_budget, bridge.workers, bridge.contract_id = (
                        snapshot.state.budget.total, snapshot.state.workers, snapshot.state.scope.uid)
                    result = engine.run()
                    guard.check()
                    if snapshot.uid != state_id:
                        raise RuntimeError("A forced rollout mutated the source paid state")
                    trajectory = bridge.convert(result, lineage_id=lineage_id)
                    trajectory = replace(trajectory, forced=True, decisions=(first,) + trajectory.decisions)
                    # Eq. (11)/(17): this restored trajectory starts at paid S,
                    # so its full_return IS the first arm's suffix return G_a,s.
                    # Initial menu preparation is absent from the suffix log;
                    # later preparation and unsuccessful work are still present.
                    # Even though later choices are sampled, the WHOLE forced
                    # trajectory is scorer-only (Algorithm 3, line 11).
                    observed_return = trajectory.full_return(training.get("cost_coefficient", .01), training.get("time_coefficient", 0.))
                    output["forced_trajectories"].append(trajectory)
                    plan_returns[plan.uid].append(observed_return)
                    plan_seeds[plan.uid].append(member_seed)
                    decisions[plan.uid] = first
                    metadata["forced_search_cost"] += result.total_cost
                    metadata["raw_search_results"].append(result.to_dict())
                    output["physical"].extend(getattr(executor, "physical_records", []))
                    output["physical"].extend(bridge.physical_inferences)
                    if not hasattr(executor, "physical_records"):
                        output["physical"].extend({"kind": "native_execution", "gpu_count": 0, **event} for event in recorded.events)
                    log({"event": "completed", "paid_state_id": state_id, "snapshot_id": snapshot_id,
                         "arm": plan.uid, "repeat": repeat, "seed": member_seed, "search_id": result.search_id,
                         "cost": result.total_cost, "elapsed_seconds": trajectory.total_elapsed,
                         "observed_return": observed_return, "executor_events": recorded.events,
                         "actor_eligible": False})
                except Exception as exc:
                    log({"event": "missing_rollout", "paid_state_id": state_id, "arm": plan.uid,
                         "repeat": repeat, "seed": member_seed, "error": str(exc), "return_label": None,
                         "settled_suffix_cost": engine.state.budget.spent - snapshot.state.budget.spent if engine else None,
                         "pending_suffix_cost": engine.state.budget.pending if engine else None,
                         "elapsed_seconds": time.monotonic() - started, "executor_events": recorded.events if recorded else [],
                         "physical_executions": getattr(executor, "physical_records", []),
                         "physical_inferences": getattr(bridge, "physical_inferences", [])})
                    raise
        for left, right in combinations(plans, 2):
            output["pairs"].append(PairwiseValueExample(decisions[left.uid], decisions[right.uid],
                tuple(plan_returns[left.uid]), tuple(plan_returns[right.uid]), state_id, snapshot_id,
                tuple(plan_seeds[left.uid]), tuple(plan_seeds[right.uid]), config.repetitions, config.repetitions,
                lineage_id=lineage_id))
    metadata["total_acquisition_cost"] = metadata["forced_search_cost"] + metadata["forced_copy_cost"]
    return output
