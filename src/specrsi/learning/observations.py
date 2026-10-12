"""Pure, recorded observation interventions; never edit the physical SpecRSI state.

NeuralSearchBridge supplies its eight state features, structured visible task
and candidate context, and action-specific option vectors. State positions are
available-budget fraction, candidate count, complete-candidate count,
normalized missing-evaluation count, pending-budget fraction, state version,
normalized ledger-record count, and history length. The registered
ablations mask evidence, retain coarse counts, blind outer or inner decisions,
relabel identifiers, or substitute an explicitly frozen donor observation.
Each view records its schema and intervention provenance for likelihood replay;
the trusted ledger, remaining resources and executable search state are not
modified by these observation transformations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Mapping, Sequence


MODES = frozenset({"full", "evidence_blind", "coarse_counts", "outer_blind",
                   "inner_blind", "sham_relabel", "mismatched_state"})
_EVIDENCE_KEYS = frozenset({"missing", "sources", "allowed_sources", "source_ids", "operations",
    "remaining_work", "dependencies", "conflicts", "setup_groups", "forecasts", "coverage",
    "receipts", "observed_outcomes", "remaining_completion_cost", "per_item_records"})
_ID_KEYS = frozenset({"id", "uid", "candidate_id", "task_id", "source_id", "source_ids",
    "operation_id", "operation_ids", "option_ids", "missing", "dependencies", "conflicts",
    "group", "setup_group", "scope_id"})
_PAYLOAD_KEYS = frozenset({"answer", "output", "outcome", "observed_outcomes", "metrics", "reference", "expected"})
_QUALITY_KEYS = _PAYLOAD_KEYS | {"score", "complete_score", "quality", "reward", "partial_score"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _context(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("Observation masking requires complete structured JSON before text truncation") from error
    if not isinstance(value, Mapping):
        raise ValueError("Observation context must be a JSON object")
    return json.loads(_json(dict(value)))  # Deep copy; no mutable input aliases.


def _vector(values):
    result = tuple(float(x) for x in values)
    if len(result) < 8 or not all(math.isfinite(x) for x in result):
        raise ValueError("Observation vectors require at least eight finite features")
    return result


@dataclass(frozen=True)
class ObservationView:
    features: tuple[float, ...]
    context_text: str
    options: tuple[tuple[float, ...], ...]
    kind: str
    mode: str
    provenance: tuple[tuple[str, str], ...] = ()

    @property
    def context(self):
        """Return a fresh copy, never an alias to the frozen observation."""
        return json.loads(self.context_text)


@dataclass(frozen=True)
class FrozenDonor:
    donor_id: str
    view: ObservationView

    def __post_init__(self):
        if not isinstance(self.donor_id, str) or not self.donor_id.strip() or self.view.mode != "full" or not isinstance(self.view.features, tuple) or not isinstance(self.view.options, tuple) or any(not isinstance(x, tuple) for x in self.view.options):
            raise ValueError("Donor must contain an immutable full view")
        if _vector(self.view.features) != self.view.features or any(_vector(row) != row for row in self.view.options):
            raise ValueError("Frozen donor contains invalid numeric vectors")
        _context(self.view.context_text)

    @classmethod
    def from_view(cls, view: ObservationView, *, donor_id: str):
        if not donor_id or view.mode != "full":
            raise ValueError("Donor requires a nonempty external identity and a full frozen observation")
        # Reconstruct immutable primitive tuples even if caller manually made a view.
        frozen = views(view.features, view.context_text, view.options, view.kind)
        return cls(str(donor_id), frozen)


def _masked(vector, keep):
    return tuple(v if i in keep else 0. for i, v in enumerate(vector))


def _blind_context(context, coarse=False):
    # An allowlist prevents a newly added evidence/history field bypassing masking.
    result = {k: context[k] for k in ("scope", "budget", "quality", "incumbent_score", "initial_context_id")
              if k in context}
    if "budget" in result:
        result["budget"] = {k: result["budget"][k] for k in ("available", "pending", "total", "reserved", "spent") if k in result["budget"]}
    if "tasks" in context:
        result["tasks"] = [{k: task[k] for k in ("id", "inputs") if k in task}
                           for task in context["tasks"]]
    if "candidates" in context:
        result["candidates"] = []
        for candidate in context["candidates"]:
            row = {k: candidate[k] for k in ("id", "complete_score", "quality", "operating_penalty") if k in candidate}
            if coarse:
                row["missing_count"] = len(candidate.get("missing", ()))
                for field in ("sources", "allowed_sources", "operations", "dependencies", "conflicts", "setup_groups"):
                    if field in candidate:
                        row[field + "_count"] = len(candidate[field])
            result["candidates"].append(row)
    if coarse:
        for field in ("operations", "dependencies", "conflicts", "setup_groups", "sources"):
            if field in context:
                result[field + "_count"] = len(context[field])
    return result


def _sham(context, seed):
    identifiers = set()

    def collect(node, identity=False):
        if isinstance(node, str) and identity:
            identifiers.add(node)
        elif isinstance(node, list):
            for value in node:
                collect(value, identity)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in _PAYLOAD_KEYS:
                    continue
                if identity:
                    identifiers.add(key)
                collect(value, key in _ID_KEYS or identity)
    collect(context)
    mapping = {value: "entity_" + hashlib.sha256((str(seed) + "\0" + value).encode()).hexdigest()
               for value in sorted(identifiers)}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Sham identity hash collision")

    def rewrite(node):
        if isinstance(node, str):
            return mapping.get(node, node)
        if isinstance(node, list):
            return [rewrite(x) for x in node]
        if isinstance(node, dict):
            return {mapping.get(k, k): v if k in _PAYLOAD_KEYS else rewrite(v) for k, v in node.items()}
        return node
    return rewrite(context), hashlib.sha256(_json(mapping).encode()).hexdigest()


def views(features: Sequence[float], context, options: Sequence[Sequence[float]], kind: str, *,
          mode="full", donor: FrozenDonor | None = None, relabel_seed="specrsi-observations-v1") -> ObservationView:
    """Freeze a controller-only view. Legal support IDs/order are not inputs here.

    Apply this before HF text truncation and record its returned inputs for replay.
    A mismatch aligns candidate/task rows by their already frozen order; it never
    grants the donor's sources, operations, receipts, budget or quality to SpecRSI.
    """
    if mode not in MODES or kind not in {"outer", "batch", "edit", "surface"}:
        raise ValueError("Unknown observation mode or decision kind")
    feature = _vector(features)
    option = tuple(_vector(row) for row in options)
    body = _context(context)
    provenance = (("definition", "prospective-bridge-eight-features-v1"),)
    effective = "evidence_blind" if (mode == "outer_blind" and kind == "outer") or (
        mode == "inner_blind" and kind == "batch") else mode
    if effective in {"outer_blind", "inner_blind"}:
        effective = "full"
    if effective in {"evidence_blind", "coarse_counts"}:
        coarse = effective == "coarse_counts"
        feature = _masked(feature, {0, 1, 2, 3, 4, 5, 6, 7} if coarse else {0, 1, 4})
        keep = ({0, 2, 3, 4, 5, 6} if coarse else {0, 3, 5, 6}) if kind == "outer" else (
            {0, 1, 2, 4, 5, 6, 7} if coarse else set())
        option = tuple(_masked(row, keep) for row in option)
        body = _blind_context(body, coarse)
    elif effective == "sham_relabel":
        body, fingerprint = _sham(body, relabel_seed)
        provenance += (("relabel_map_sha256", fingerprint), ("seed", str(relabel_seed)))
    elif effective == "mismatched_state":
        if not isinstance(donor, FrozenDonor) or not donor.donor_id or donor.view.mode != "full":
            raise ValueError("mismatched_state requires an explicitly frozen full donor observation")
        other = donor.view
        if other.kind != kind or len(other.features) != len(feature) or tuple(map(len, other.options)) != tuple(map(len, option)):
            raise ValueError("Frozen donor feature/option shapes or decision kind differ")
        donor_body = other.context
        for field in ("tasks", "candidates"):
            if len(body.get(field, ())) != len(donor_body.get(field, ())):
                raise ValueError("Frozen donor requires matching candidate/task row counts")
        # Keep current remaining resources, candidate count, quality and action kind.
        current = ObservationView(feature, _json(body), option, kind, "full")
        feature = tuple(x if i in {0, 1, 4} else other.features[i] for i, x in enumerate(feature))
        preserve = {0, 3, 5, 6} if kind == "outer" else set()
        option = tuple(tuple(x if i in preserve else borrowed[i] for i, x in enumerate(row))
                       for row, borrowed in zip(option, other.options))
        ids = {}
        for field in ("tasks", "candidates"):
            ids.update({old["id"]: new["id"] for old, new in zip(donor_body.get(field, ()), body.get(field, ()))
                        if "id" in old and "id" in new})

        def aligned(value):
            if isinstance(value, str): return ids.get(value, value)
            if isinstance(value, list): return [aligned(v) for v in value]
            if isinstance(value, dict): return {ids.get(k, k): aligned(v) for k, v in value.items()}
            return value

        def evidence_only(value):
            if isinstance(value, list): return [evidence_only(v) for v in value]
            if isinstance(value, dict): return {k: evidence_only(v) for k, v in value.items() if k not in _QUALITY_KEYS}
            return value

        retained = [{k: v for k, v in row.items() if k in _QUALITY_KEYS | {"candidate_id", "task_id"}}
                    for row in body.get("per_item_records", ()) if isinstance(row, dict)]
        if any(any(k in _QUALITY_KEYS for k in row) for row in retained):
            body["retained_observed_quality"] = retained
        for field in (_EVIDENCE_KEYS - {"observed_outcomes"}) | {"recent_history"}:
            body.pop(field, None)
            if field in donor_body:
                body[field] = evidence_only(aligned(donor_body[field]))
        for row, borrowed in zip(body.get("candidates", ()), donor_body.get("candidates", ())):
            for field in _EVIDENCE_KEYS - {"observed_outcomes"}:
                row.pop(field, None)
                if field in borrowed:
                    row[field] = evidence_only(aligned(borrowed[field]))
        comparable_current = _context(current.context_text)
        if retained and "retained_observed_quality" in body:
            comparable_current["retained_observed_quality"] = retained
        for field in (_EVIDENCE_KEYS - {"observed_outcomes"}) | {"recent_history"}:
            if field in comparable_current:
                comparable_current[field] = evidence_only(comparable_current[field])
        for row in comparable_current.get("candidates", ()):
            for field in _EVIDENCE_KEYS - {"observed_outcomes"}:
                if field in row: row[field] = evidence_only(row[field])
        if (feature, _json(body), option) == (current.features, _json(comparable_current), current.options):
            raise ValueError("Frozen donor does not change any permitted evidence observation")
        frozen_record = {"context": other.context, "features": other.features, "options": other.options, "kind": other.kind}
        provenance += (("donor_id", donor.donor_id), ("donor_sha256", hashlib.sha256(_json(frozen_record).encode()).hexdigest()),
                       ("alignment", "frozen_candidate_and_task_row_order"))
    return ObservationView(feature, _json(body), option, kind, mode, provenance)


def rule_values(qualities, completion_costs, kinds, *, rule, total_budget, spent, draft_spent,
                initial_fraction=.05, draft_fraction=.35):
    """Prospective non-neural scores over an already legal ordered support.

    Initial fraction is a REAL-spending draft phase, never a training charge.
    Thereafter follow a draft-spending share when the corresponding legal action
    exists. Unknown quality ranks last; host action legality/reserves remain binding.
    Returned finite ranks preserve utility order and ties for greedy rule choice.
    """
    if rule not in {"quality_only", "completion_cost"} or len(qualities) != len(completion_costs) or len(kinds) != len(qualities):
        raise ValueError("Invalid rule or action arrays")
    if not kinds or any(k not in {"draft", "verify"} for k in kinds):
        raise ValueError("Rules require a nonempty already legal draft/verify support")
    numbers = (total_budget, spent, draft_spent, initial_fraction, draft_fraction, *completion_costs)
    if not all(math.isfinite(float(x)) and x >= 0 for x in numbers) or total_budget <= 0 or draft_spent > spent or spent > total_budget:
        raise ValueError("Invalid actual spending, costs or budget")
    if not 0 <= initial_fraction <= 1 or not 0 <= draft_fraction <= 1:
        raise ValueError("Rule fractions must lie in [0,1]")
    if any(q is not None and not math.isfinite(float(q)) for q in qualities):
        raise ValueError("Known quality must be finite")
    wants_draft = spent < initial_fraction * total_budget or draft_spent < draft_fraction * spent
    preferred = "draft" if wants_draft else "verify"
    allowed = [i for i, kind in enumerate(kinds) if kind == preferred] or list(range(len(kinds)))
    utility = {}
    for i in allowed:
        utility[i] = (float(qualities[i]) if qualities[i] is not None else float("-inf")) if rule == "quality_only" else -float(completion_costs[i])
    ordering = {value: float(rank) for rank, value in enumerate(sorted(set(utility.values())))}
    result = [-1.] * len(kinds)
    for i in allowed:
        result[i] = ordering[utility[i]]
    return tuple(result)


class RuleController:
    """Non-neural callbacks with real spending and conservative completion caps.

    Ranking the full frozen candidate pool preserves order on every legal subset;
    the host alone chooses and enforces that subset. No fabricated partial quality
    or predicted P success is used. Reserved but unpaid work still has a cost.
    """
    def __init__(self, config, *, rule="quality_only", initial_fraction=.05, draft_fraction=.35,
                 completion_bound=None, cost_meter=None):
        if rule not in {"quality_only", "completion_cost"}:
            raise ValueError("Unknown non-neural allocation rule")
        if not 0 <= initial_fraction <= 1 or not 0 <= draft_fraction <= 1:
            raise ValueError("Rule fractions must lie in [0,1]")
        if cost_meter is None and config.cost_unit not in {"tokens", "logical-work", "wall-ms"}:
            raise ValueError("Native units require an explicit scorer cost_meter")
        self.config, self.rule = config, rule
        self.initial_fraction, self.draft_fraction = initial_fraction, draft_fraction
        self.completion_bound = completion_bound
        self.cost_meter = cost_meter
        self.physical_inferences = []
        self.variant = "caller_supplied_completion_bound" if completion_bound else "all_U_remaining_attempt_upper_bound"

    def _quality(self, state, candidate):
        if not state.ledger.complete(candidate, state.scope):
            return None
        # Eq. (2)/(3): the contract defines J_Gamma(c) once.
        # Ledger.score already subtracts the registered operating penalty from
        # its aggregation. Subtracting it again here changes the baseline's
        # candidate ordering and compares a different utility from VATS.
        return state.ledger.score(candidate, state.scope)

    def _direct_cap(self, task):
        return self.config.caps(task, direct=True).total()

    def _completion(self, state, candidate, kind):
        if self.completion_bound is not None:
            cap = self.completion_bound(state, candidate, kind)
        else:
            tasks = state.scope.tasks if kind == "draft" else state.ledger.missing(candidate, state.scope)
            cap = sum(self._direct_cap(t) for t in tasks) + (self.config.draft_cap if kind == "draft" else 0)
        if type(cap) is not int or cap < 0:
            raise ValueError("Completion rule needs a nonnegative enforced integer attempt cap")
        return cap

    def _receipt(self, started, kind, uid, value, **metadata):
        from specrsi.core.search import ValueEstimate
        elapsed = time.perf_counter() - started
        actual = (self.cost_meter(elapsed, kind) if self.cost_meter else
                  0 if self.config.cost_unit == "tokens" else
                  max(1, math.ceil(elapsed * 1000)) if self.config.cost_unit == "wall-ms" else 1)
        if type(actual) is not int or actual < 0:
            raise ValueError("Rule cost meter returned invalid actual spending")
        self.physical_inferences.append({"kind": "rule_control", "role": kind, "option_id": uid,
            "elapsed_seconds": elapsed, "gpu_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "charged_cost": actual, "cost_unit": self.config.cost_unit, "rule": self.rule,
            "completion_cost_variant": self.variant, **metadata})
        return ValueEstimate(float(value), actual)

    def outer_value(self, state, action):
        started = time.perf_counter()
        if action.kind not in {"draft", "verify"} or action.candidate_id not in state.candidates:
            raise ValueError("Unknown candidate or outer action")
        if not state.budget.cost_valid:
            raise ValueError("Rule cannot rank a state with invalid actual cost accounting")
        candidate = state.candidates[action.candidate_id]
        spent = state.budget.spent
        draft_spent = sum(c.actual for c in state.budget.charges if c.category == "draft")
        prefer_draft = spent < self.initial_fraction * state.budget.total or draft_spent < self.draft_fraction * spent
        preferred = "draft" if prefer_draft else "verify"
        if self.rule == "quality_only":
            quality = self._quality(state, candidate)
            values = [self._quality(state, c) for c in state.candidates.values()]
            ordered = sorted({q for q in values if q is not None})
            utility = -1. if quality is None else ordered.index(quality) / max(1, len(ordered))
            metadata = {"known_completed_quality": quality}
        else:
            cap = self._completion(state, candidate, action.kind)
            caps = sorted({self._completion(state, c, kind) for c in state.candidates.values()
                           for kind in ("draft", "verify")}, reverse=True)
            utility = caps.index(cap) / max(1, len(caps))
            metadata = {"remaining_attempt_cap": cap}
        # Preferred kind always dominates; if no such legal action exists, the
        # other kind retains its quality/cost ordering instead of collapsing ties.
        value = (3. if action.kind == preferred else 0.) + utility
        return self._receipt(started, "outer", action.uid, value, actual_spent=spent,
                             actual_draft_spent=draft_spent, preferred_kind=preferred, **metadata)

    def batch_value(self, state, candidate, batch, problem):
        started = time.perf_counter()
        selected = [problem.by_id[uid] for uid in batch.option_ids]
        missing = {t.uid for t in state.ledger.missing(candidate, state.scope)}
        coverage = set().union(*(set(option.task_ids) for option in selected)) if selected else set()
        # problem.cost subtracts protected reservations for admission. They are
        # unpaid physical work, so use full caps when estimating completion cost.
        physical_cap = sum(option.cap for option in selected) + sum(problem.setup_costs.get(group, 0) for group in {o.group for o in selected})
        residual = sum(self._direct_cap(task) for task in state.scope.tasks if task.uid in missing - coverage)
        if self.rule == "completion_cost":
            value = 1. / (1. + (physical_cap + residual) / max(1, state.budget.total))
        else:
            n = max(1, len(missing))
            cheaper = 1. / (1. + physical_cap / max(1, state.budget.total))
            value = 3. * missing.issubset(coverage) + len(missing & coverage) / n + cheaper / (n+1)
        return self._receipt(started, "batch", batch.uid, value, physical_attempt_cap=physical_cap,
                             remaining_direct_attempt_cap=residual)


def make_rule_controller(config, *, rule="quality_only", **kwargs):
    return RuleController(config, rule=rule, **kwargs)
