"""Development demonstrations from executed, verified VATS work.

The collection policy completes observed children before requesting another edit.
It receives observed execution feedback without expected task answers or
unexecuted outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math

from specrsi.core.models import Candidate, thaw

from .paper_bridge import VATSSearchBridge
from .records import EditExample


DEMONSTRATION_POLICIES = frozenset({"complete_then_draft", "observed_plateau"})


@dataclass(frozen=True)
class DemonstrationStopping:
    """A registered local teacher rule, using only completed observed edits.

    Collection stopping uses observed child gains and a configurable patience.
    Demonstrations and return labels require executed actions.
    """
    patience: int = 1
    min_completed_edits: int = 1
    min_gain: float = 0.

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.patience, self.min_completed_edits)):
            raise ValueError("Demonstration stopping counts must be positive integers")
        if isinstance(self.min_gain, bool) or not math.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError("Demonstration stopping min_gain must be finite and nonnegative")

    @classmethod
    def from_settings(cls, settings=None):
        if isinstance(settings, cls):
            return settings
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("demonstration_stopping must be a registered mapping")
        try:
            return cls(**(settings or {}))
        except TypeError as exc:
            raise ValueError("Unknown demonstration_stopping setting") from exc


def demonstration_bridge_kwargs(settings):
    name = settings.get("demonstration_policy", "complete_then_draft")
    if not isinstance(name, str) or name not in DEMONSTRATION_POLICIES:
        raise ValueError("Unknown registered demonstration policy")
    stopping = settings.get("demonstration_stopping")
    if stopping is not None and name != "observed_plateau":
        raise ValueError("demonstration_stopping requires observed_plateau policy")
    DemonstrationStopping.from_settings(stopping)
    DemonstrationQuality.from_settings(settings.get("demonstration_quality"))
    return {"demonstration_policy": name, "demonstration_stopping": stopping,
            "demonstration_quality": settings.get("demonstration_quality")}


@dataclass(frozen=True)
class DemonstrationQuality:
    """Registered SFT dataset selection, independent of search/GVO objectives.

    The default accepts complete, admissible executed edits. Quality comparisons
    use the fixed host utility, including its operating penalty,
    on the parent's and child's complete records from the same contract.
    """
    edit_rule: str = "verified"
    min_gain: float = 0.
    comparison_epsilon: float = 1e-12
    require_changed_artifacts: bool = False
    plan_rule: str = "executed"

    def __post_init__(self):
        if self.edit_rule not in {"verified", "parent_nondegrading", "parent_improving"}:
            raise ValueError("Unknown demonstration edit quality rule")
        if not math.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError("Demonstration min_gain must be finite and nonnegative")
        if not math.isfinite(self.comparison_epsilon) or self.comparison_epsilon < 0:
            raise ValueError("Demonstration comparison_epsilon must be finite and nonnegative")
        if self.edit_rule == "verified" and self.min_gain:
            raise ValueError("A nonzero demonstration min_gain requires a parent quality rule")
        if not isinstance(self.require_changed_artifacts, bool):
            raise ValueError("require_changed_artifacts must be a boolean")
        if self.plan_rule not in {"executed", "qualified_edits"}:
            raise ValueError("Unknown demonstration plan quality rule")

    @classmethod
    def from_settings(cls, settings=None):
        if isinstance(settings, cls):
            return settings
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("demonstration_quality must be a registered mapping")
        try:
            return cls(**(settings or {}))
        except TypeError as exc:
            raise ValueError("Unknown demonstration_quality setting") from exc

    def decision(self, *, verified, parent_verified, changed, child_score, parent_score):
        if not verified:
            return False, "child_not_complete_admissible_in_executed_tree"
        if self.require_changed_artifacts and not changed:
            return False, "unchanged_artifacts"
        if self.edit_rule == "verified":
            return True, "complete_admissible_edit"
        if not parent_verified:
            return False, "parent_not_complete_admissible"
        gain = child_score - parent_score
        allowed = (gain >= self.min_gain - self.comparison_epsilon if self.edit_rule == "parent_nondegrading"
                   else gain > self.min_gain + self.comparison_epsilon)
        return allowed, "observed_parent_gain_qualified" if allowed else "observed_parent_gain_below_rule"


class DemonstrationVATSBridge(VATSSearchBridge):
    demonstration_policy = "complete_then_draft"

    def __init__(self, *args, demonstration_quality=None, demonstration_policy="complete_then_draft",
                 demonstration_stopping=None, **kwargs):
        super().__init__(*args, **kwargs)
        demonstration_bridge_kwargs({"demonstration_policy": demonstration_policy,
            "demonstration_stopping": demonstration_stopping, "demonstration_quality": demonstration_quality})
        self.demonstration_policy = demonstration_policy
        self.demonstration_stopping = DemonstrationStopping.from_settings(demonstration_stopping)
        self.demonstration_quality = DemonstrationQuality.from_settings(demonstration_quality)
        self.executed_edits = []
        self.edit_evidence = []
        self.plan_evidence = []
        self._proposals = []
        self._quality_context = None
        self.teacher_choice_evidence = []

    def observed_stopping_evidence(self, state):
        """Describe actual complete edits in immutable tree insertion order."""
        history = []
        for child in state.candidates.values():
            parent = state.candidates.get(child.parent_id)
            if (parent is None or not state.ledger.complete(child, state.scope) or
                    not state.ledger.complete(parent, state.scope) or
                    not state.ledger.admissible(parent, state.scope)):
                continue
            admissible = state.ledger.admissible(child, state.scope)
            parent_score = state.ledger.score(parent, state.scope)
            child_score = state.ledger.score(child, state.scope) if admissible else None
            gain = child_score - parent_score if admissible else None
            history.append({"child_id": child.uid, "parent_id": parent.uid,
                "complete": True, "admissible": admissible, "parent_utility": parent_score,
                "child_utility": child_score, "gain": gain,
                "progress": bool(admissible and gain > self.demonstration_stopping.min_gain)})
        plateau = 0
        for row in reversed(history):
            if row["progress"]:
                break
            plateau += 1
        return {"completed_edits": history, "trailing_nonprogress_edits": plateau,
                "stop": (len(history) >= self.demonstration_stopping.min_completed_edits and
                         plateau >= self.demonstration_stopping.patience)}

    def policy_menu(self, state, plans):
        # Keep the live trusted ledger to use its registered native aggregation
        # after execution. Do not reconstruct utility from forecasts or means.
        if state.scope.split == "audit":
            raise ValueError("Demonstrations cannot use untouched audit tasks")
        self._quality_context = (state.ledger, state.scope)
        # Retain the exact paid-state inputs and meter the role forward normally.
        # The resulting learned logits are replaced by this registered teacher.
        super().policy_menu(state, plans)
        stopping = self.observed_stopping_evidence(state) if self.demonstration_policy == "observed_plateau" else None
        verify = [p for p in plans if p.kind == "verify"]
        if verify:
            def order(plan):
                coverage = {key for job in plan.jobs for key in job.keys}
                completed = 0
                for candidate in state.candidates.values():
                    missing = {(candidate.uid, t.uid) for t in state.ledger.missing(candidate, state.scope)}
                    completed += bool(missing) and missing.issubset(coverage)
                return (-completed, plan.work_cap, -len(coverage), plan.uid)
            selected = min(verify, key=order)
            reason = "complete_observed_child_first"
        elif stopping is not None and stopping["stop"]:
            selected = next(p for p in plans if p.kind == "accept")
            reason = "observed_plateau"
        else:
            drafts = [p for p in plans if p.kind == "draft"]
            def order(plan):
                scores = [state.ledger.score(state.candidates[uid], state.scope)
                    if state.ledger.complete(state.candidates[uid], state.scope) and
                       state.ledger.admissible(state.candidates[uid], state.scope) else float("-inf")
                    for uid in plan.parent_ids]
                return (plan.depth, len(plan.parent_ids), -max(scores), plan.work_cap, plan.uid)
            selected = min(drafts, key=order) if drafts else next(p for p in plans if p.kind == "accept")
            reason = "legal_draft" if drafts else "no_draft_or_verify_available"
        self.teacher_choice_evidence.append({"transition_index": getattr(state, "current_transition_index", state.version),
            "selected_id": selected.uid, "support": [p.uid for p in plans],
            "rule": self.demonstration_policy, "reason": reason,
            "stopping_settings": asdict(self.demonstration_stopping) if stopping is not None else None,
            "observed_stopping_evidence": stopping})
        return [0. if p.kind == "accept" else 1000. if p.uid == selected.uid else -1000. for p in plans]

    def record_executed_edit(self, child, parent, sample):
        # Construction alone is not verification. convert() checks the final
        # immutable tree and complete actual ledger before assigning a label.
        self._proposals.append((child, parent, sample))

    def convert(self, result, *, lineage_id):
        trajectory = super().convert(result, lineage_id=lineage_id)
        self.teacher_choice_evidence = [{**row, "search_id": trajectory.search_id,
            "lineage_id": lineage_id} for row in self.teacher_choice_evidence]
        plans = [d for d in trajectory.decisions if d.kind == "plan"]
        if any(d.exploration != 0 for d in plans):
            raise ValueError("The registered demonstration policy requires exploration=0")
        if any(not d.deterministic or d.behavior_logprob is not None for d in plans):
            raise ValueError("Teacher plans must execute deterministically; sampled work cannot be relabelled")
        if len(plans) != len(self.teacher_choice_evidence) or any(
                row["transition_index"] != decision.transition_index or
                tuple(row["support"]) != decision.support or
                row["selected_id"] != decision.support[decision.selected_index]
                for row, decision in zip(self.teacher_choice_evidence, plans)):
            raise ValueError("Teacher choice evidence must exactly match every actually executed plan")
        decisions = trajectory.decisions
        raw = json.loads(result.ledger_json)
        records = raw["records"]
        scope_ids = {r["scope_id"] for r in records}
        if len(scope_ids) != 1:
            raise ValueError("Demonstration verification requires one fixed development contract")
        by_candidate = {}
        for record in records:
            by_candidate.setdefault(record["candidate_id"], []).append(record)
        required_count = len(raw["scopes"][0]["tasks"])
        frozen = {e["candidate_id"] for transition in result.transitions for e in transition.events
            if e.get("event") == "freeze"}
        candidates = {candidate.uid: candidate for candidate in (Candidate(**r) for r in raw["candidates"])}
        proposals = {child.uid: (child, parent, sample) for child, parent, sample in self._proposals}
        for child_id in frozen:
            child = candidates[child_id]
            if child.parent_id in candidates:
                proposals.setdefault(child_id, (child, candidates[child.parent_id], None))
        self.executed_edits, self.edit_evidence, self.plan_evidence = [], [], []
        qualified = set()
        complete_ids = set(result.acceptance.complete_candidates)
        ledger, scope = self._quality_context if self._quality_context else (None, None)
        if proposals and ledger is None:
            raise ValueError("Demonstration quality needs its actual execution ledger")
        for child, parent, sample in proposals.values():
            actual = by_candidate.get(child.uid, [])
            # Host acceptance checked quarantine/provenance, all required
            # outcomes, admissibility and any additional hard constraints.
            verified = child.uid in frozen and child.uid in complete_ids
            parent_verified = parent.uid in complete_ids
            changed = thaw(child.artifacts) != thaw(parent.artifacts)
            # Ledger.score already subtracts the candidate operating penalty.
            child_score = ledger.score(child, scope) if verified else None
            parent_score = ledger.score(parent, scope) if parent_verified else None
            eligible, reason = self.demonstration_quality.decision(verified=verified,
                parent_verified=parent_verified, changed=changed,
                child_score=child_score, parent_score=parent_score)
            evidence = {"search_id": trajectory.search_id, "lineage_id": lineage_id,
                "child_id": child.uid, "parent_id": parent.uid,
                "executed_in_tree": child.uid in frozen, "complete_admissible_verification": verified,
                "parent_complete_admissible_verification": parent_verified,
                "generation_stop_reason": sample.stop_reason if sample else None,
                "admitted_record_count": len(actual), "required_record_count": required_count,
                "changed_artifacts": changed, "child_utility": child_score, "parent_utility": parent_score,
                "observed_parent_gain": child_score - parent_score if verified and parent_verified else None,
                "utility_contract": "ledger.score_including_operating_penalty_once",
                "demonstration_quality": asdict(self.demonstration_quality),
                "sft_eligible": eligible, "sft_selection_reason": reason,
                "autoregressive_example": bool(sample and sample.generated_ids)}
            self.edit_evidence.append(evidence)
            if eligible:
                qualified.add(child.uid)
            if eligible and sample and sample.generated_ids:
                ids = sample.prompt_ids + sample.generated_ids
                labels = (-100,) * len(sample.prompt_ids) + sample.generated_ids
                self.executed_edits.append(EditExample(lineage_id, ids, labels, True,
                    token_decisions=sample.decisions))
        for index, decision in enumerate(decisions):
            if decision.kind not in {"plan", "edit"}:
                continue
            transition = result.transitions[decision.transition_index]
            targets = {event["candidate_id"] for event in transition.events
                if event.get("event") in {"freeze", "acquisition"} and "candidate_id" in event}
            is_accept = decision.kind == "plan" and decision.support[decision.selected_index] == "ACCEPT"
            eligible = self.demonstration_quality.plan_rule == "executed" or is_accept or bool(targets & qualified)
            self.plan_evidence.append({"search_id": trajectory.search_id, "lineage_id": lineage_id,
                "decision_index": index, "transition_index": decision.transition_index,
                "kind": decision.kind, "selected_id": decision.support[decision.selected_index],
                "target_candidate_ids": sorted(targets), "qualified_candidate_ids": sorted(targets & qualified),
                "sft_eligible": eligible, "plan_rule": self.demonstration_quality.plan_rule})
        # Every original transition, cost and Q target is retained, including
        # failed/unverified/quality-rejected edits. Selection changes only SFT.
        return replace(trajectory, decisions=decisions, demonstration=True)


def warmup_records(collection):
    """Keep every critic trajectory; select only registered SFT demonstrations."""
    trajectories = [t for group in collection["groups"] for t in group]
    if any(not t.demonstration or t.audit_only or t.forced for t in trajectories):
        raise ValueError("Collected SFT requires explicitly marked development demonstrations")
    metadata = collection.get("metadata", {})
    quality = DemonstrationQuality.from_settings(metadata.get("demonstration_quality"))
    evidence = metadata.get("sft_plan_evidence")
    if quality.plan_rule != "executed" and evidence is None:
        raise ValueError("Filtered SFT plans require complete selection evidence")
    selection = None if evidence is None else {(row["search_id"], row["decision_index"]): row for row in evidence}
    if selection is not None and len(selection) != len(evidence):
        raise ValueError("Duplicate SFT plan selection evidence")
    records = []
    for trajectory in trajectories:
        records.append({"record_type": "trajectory", **asdict(trajectory)})
        for index, decision in enumerate(trajectory.decisions):
            # Discrete reference edits share the policy head; real HF edits use
            # their verified autoregressive demonstration below, exactly once.
            if decision.kind not in {"plan", "edit"}:
                continue
            if selection is not None:
                row = selection.get((trajectory.search_id, index))
                if (row is None or row["lineage_id"] != trajectory.lineage_id or
                        row["selected_id"] != decision.support[decision.selected_index] or
                        row["transition_index"] != decision.transition_index or row["kind"] != decision.kind):
                    raise ValueError("SFT plan selection must identify its actual executed decision")
                if not row["sft_eligible"]:
                    continue
            records.append({"record_type": "plan", "lineage_id": trajectory.lineage_id,
                "decision": asdict(decision)})
    records.extend({"record_type": "edit", **asdict(edit)} for edit in collection.get("edits", ()))
    if collection.get("forced_trajectories") or collection.get("pairs"):
        raise ValueError("Collect warm-up demonstrations separately from forced GVO continuations")
    return records
