"""Executable revisions, observed history, and fixed external acceptance.

Eq. (1), the general recursive improvement loop:
    c_tilde_t ~ Propose_{Theta_t}(. | V_t, H_t),
    z_t = f_real(c_tilde_t),
    (V_{t+1}, H_{t+1}) = Update(V_t, H_t, c_tilde_t, z_t).
V_t contains retained executable versions (``state.candidates``); H_t contains
observed execution feedback (``state.history`` and admitted ledger records).
Propose selects an actual parent and produces an executable revision.
The evaluator supplies z_t; a predicted value is not z_t. Update records both
successful and failed observations, so retaining a version does not imply that
it improves the incumbent or qualifies for adoption. ``_draft`` handles the
version-producing part, acquisition handles real observations, and ``run``
records transitions and selects the final complete admissible candidate.
Eq. (1) states the general propose/evaluate/update setting, not a mandatory
evaluation barrier after every VATS draft. In VATS, DRAFT creates an immutable
version and VERIFY contributes observed partial records; successors can be
drafted before all their parent's required items have been evaluated.

Eq. (2), the real-evaluation interface and best observed valid candidate:
    f_real(c) = (J(c), v(c)),
    c_star_n in argmax_{(c,J,v) in H_n : v=1} J.
J is higher-is-better utility; v is a binary hard-constraint validity flag.
The subscript n counts real-evaluation attempts; it is neither monetary expense,
a token count, nor the number G of independent grouped searches. In the itemized
contract, the complete f_real pair becomes available only when Ledger reconstructs J_Gamma(c)
and admissibility from the complete required outcome vector. Partial coverage
does not define a complete aggregate, even if every observed item passed.

The VATS subclass reuses immutable drafts, acquisition, and result structures
while providing its own unified work menu and separate preparation/work clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import math
import random
from typing import Callable, Mapping, Protocol
import uuid

from .acceptance import AcceptanceResult, accept, complete_improvement
from .acquisition import AcquisitionCaps, ResultAcquirer
from .budget import BudgetExceeded, BudgetLedger, CostCapViolation, cost_int
from .execution import Executor
from .ledger import Ledger
from .models import Candidate, EvaluationScope, FrozenMap, Task, digest, thaw
from specrsi.planning.batches import Batch, BatchProblem, CheckOption, MenuBuilder
from specrsi.planning.policy import TraceDecision, choose


@dataclass(frozen=True)
class ValueEstimate:
    value: float
    actual_cost: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("nonfinite value estimate")
        cost_int(self.actual_cost)


@dataclass(frozen=True)
class DraftResult:
    candidate: Candidate | None
    actual_cost: int
    forecasts: FrozenMap = field(default_factory=FrozenMap)
    status: str = "complete"
    error: str | None = None
    edit_decision: TraceDecision | None = None

    def __post_init__(self) -> None:
        cost_int(self.actual_cost)
        object.__setattr__(self, "forecasts", FrozenMap(self.forecasts))
        if self.status not in ("complete", "interrupted", "error"):
            raise ValueError("invalid draft status")
        if (self.status == "complete") != (self.candidate is not None):
            raise ValueError("only complete drafts freeze a child")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.forecasts.values()):
            raise ValueError("surface forecasts must be probabilities")


class Drafter(Protocol):
    def draft(self, parent: Candidate, state: "SearchState", cap: int, rng: random.Random) -> DraftResult: ...


@dataclass(frozen=True)
class Action:
    kind: str
    candidate_id: str
    work_cap: int
    # Verification plans reserve their suboperations after paid planning. This
    # conservative whole-transition bound is also used by deployment exploration.
    full_transition_cap: int | None = None

    @property
    def uid(self) -> str:
        return f"{self.kind}:{self.candidate_id}"


@dataclass(frozen=True)
class SearchConfig:
    execution_cap: int = 100
    execution_caps: FrozenMap = field(default_factory=FrozenMap)
    draft_cap: int = 10
    lookup_cost: int = 0
    check_cap: int = 1
    max_sources: int = 1
    planning_cost: int = 1
    outer_value_cap: int = 1
    batch_value_cap: int = 1
    setup_cost: int = 0
    max_actions: int = 100
    max_candidates: int = 16
    max_children: int = 2
    mode: str = "deploy"
    seed: int = 0
    margin: float = 0.
    exploration: float = .1
    temperature: float = 1.
    exploration_budget: int = 0
    reserve_completion: bool = True
    all_u: bool = False
    cost_unit: str = "logical-work"
    group_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_caps", FrozenMap(self.execution_caps))
        for name in ("execution_cap", "draft_cap", "lookup_cost", "check_cap", "max_sources", "planning_cost",
                     "outer_value_cap", "batch_value_cap", "setup_cost", "max_actions", "max_candidates", "max_children", "exploration_budget"):
            cost_int(getattr(self, name))
        for value in self.execution_caps.values():
            cost_int(value)
        if self.max_candidates < 1 or self.mode not in ("deploy", "collect") or not 0 <= self.exploration <= 1:
            raise ValueError("invalid search configuration")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("invalid temperature or margin")

    def caps(self, task: Task, *, direct: bool = False) -> AcquisitionCaps:
        execution = self.execution_caps.get(task.uid, self.execution_caps.get(task.name, self.execution_cap))
        return AcquisitionCaps(execution, self.lookup_cost, self.check_cap, 0 if direct or self.all_u else self.max_sources)


@dataclass
class SearchState:
    """Observed state and immutable versions shared by execution adapters.

    The search state is S=(T,E,H,B,N,w), with T=(C,A) anchored at root r.
    ``candidates`` is C; each candidate's immutable parent_id represents its edge
    in A; ``ledger`` is E; ``history`` is H; ``budget`` accounts for B. VATSState
    adds remaining primitive attempts N and workers w. The incumbent remains the
    root r for gain measurement even after a better complete candidate is observed.

    For required task set I, m_ci=1 iff valid contracted evidence for (c,i) exists;
        U_c(S)={i in I : m_ci=0},
        K_c(S)={(c,i) : i in U_c(S)}.
    Neither ``forecasts`` nor planned job coverage changes m_ci. A parent may be
    incomplete or inadmissible and still be editable. Only complete admissible
    versions contribute to F(S) or can be returned by ACCEPT.

    B uses the experiment's declared integer expense unit; N counts primitive
    attempts, including failed attempts; w counts concurrent workers. Candidate
    count, decision count, primitive attempts and expense are distinct quantities.
    """
    scope: EvaluationScope
    incumbent: Candidate
    candidates: dict[str, Candidate]
    ledger: Ledger
    budget: BudgetLedger
    forecasts: dict[str, dict[str, float]] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    version: int = 0
    children_drafted: dict[str, int] = field(default_factory=dict)
    finalist_id: str | None = None

    def features(self) -> tuple[float, ...]:
        n = max(1, len(self.scope.tasks))
        counts = [len(self.ledger.missing(c, self.scope)) for c in self.candidates.values()]
        return (self.budget.available / max(1, self.budget.total), len(self.candidates),
                sum(x == 0 for x in counts), sum(counts) / n, self.budget.pending / max(1, self.budget.total))


@dataclass(frozen=True)
class TraceTransition:
    index: int
    action: str
    candidate_id: str
    state_version_before: int
    state_version_after: int
    cost: int
    f_before: float
    f_after: float
    events: tuple[dict, ...]
    category: str = "search"
    elapsed_seconds: float = 0.


@dataclass(frozen=True)
class SearchResult:
    acceptance: AcceptanceResult
    initial_context_id: str
    search_id: str
    seed: int
    group_id: str
    initial_f: float
    final_f: float
    total_cost: int
    cost_unit: str
    transitions: tuple[TraceTransition, ...]
    decisions: tuple[TraceDecision, ...]
    charges: tuple[dict, ...]
    stop_reason: str
    candidate_ids: tuple[str, ...]
    budget_remaining: int
    ledger_json: str
    value_source: str
    elapsed_seconds: float = 0.
    algorithm: str = "legacy-two-level"

    @property
    def selected(self) -> Candidate:
        return self.acceptance.selected

    @property
    def objective_elapsed_seconds(self) -> float:
        """Measured preparation/work time used by the recorded return targets.

        ACCEPT has zero continuation time. Initial state hashing, final result
        serialization and other host bookkeeping remain in overall elapsed time.
        """
        return math.fsum(transition.elapsed_seconds for transition in self.transitions)

    @property
    def bookkeeping_elapsed_seconds(self) -> float:
        return max(0., self.elapsed_seconds - self.objective_elapsed_seconds)

    def to_dict(self) -> dict:
        result = {k: getattr(self, k) for k in ("initial_context_id", "search_id", "seed", "group_id", "initial_f", "final_f",
                  "total_cost", "cost_unit", "stop_reason", "candidate_ids", "budget_remaining", "value_source",
                  "elapsed_seconds", "algorithm")}
        result["acceptance"] = {"selected": self.selected.manifest(), "selected_id": self.selected.uid,
              "incumbent_score": self.acceptance.incumbent_score, "selected_score": self.acceptance.selected_score,
              "best_complete_score": self.acceptance.best_complete_score, "improved": self.acceptance.improved,
              "margin": self.acceptance.margin, "complete_candidates": self.acceptance.complete_candidates}
        result["timing"] = {"overall_elapsed_seconds": self.elapsed_seconds,
                            "objective_elapsed_seconds": self.objective_elapsed_seconds,
                            "bookkeeping_elapsed_seconds": self.bookkeeping_elapsed_seconds,
                            "objective_scope": "measured preparation and work phases; ACCEPT continuation is zero",
                            "overall_scope": "run entry through assembled result payload; excludes caller serialization"}
        result["transitions"] = [asdict(x) for x in self.transitions]
        result["decisions"] = [asdict(x) for x in self.decisions]
        result["charges"] = list(self.charges)
        import json
        result["ledger"] = json.loads(self.ledger_json)
        return result


OuterValue = Callable[[SearchState, Action], float | ValueEstimate]
BatchValue = Callable[[SearchState, Candidate, Batch, BatchProblem], float | ValueEstimate]


def heuristic_outer_value(state: SearchState, action: Action) -> float:
    """Heuristic baseline that scores candidates and verification work."""
    if action.kind == "draft":
        return .3 / (1 + state.children_drafted.get(action.candidate_id, 0))
    candidate = state.candidates[action.candidate_id]
    missing = len(state.ledger.missing(candidate, state.scope))
    return 1. + 1. / max(1, missing)


def heuristic_batch_value(state: SearchState, candidate: Candidate, batch: Batch, problem: BatchProblem) -> float:
    """Completion bonus illustrates complementarity; it is explicitly a heuristic."""
    missing = {t.uid for t in state.ledger.missing(candidate, state.scope)}
    coverage = set().union(*(set(problem.by_id[i].task_ids) for i in batch.option_ids)) if batch.option_ids else set()
    progress = len(coverage & missing) / max(1, len(missing))
    reuse_forecast = sum(state.forecasts.get(candidate.uid, {}).get(t, .5)
                         for i in batch.option_ids if problem.by_id[i].kind == "acquire" for t in problem.by_id[i].task_ids)
    return progress + 2. * float(missing.issubset(coverage)) + .05 * reuse_forecast - .001 * problem.cost(batch.option_ids)


class SearchEngine:
    def __init__(self, scope: EvaluationScope, incumbent: Candidate, ledger: Ledger,
                 budget: BudgetLedger, executor: Executor, drafter: Drafter,
                 config: SearchConfig | None = None, *, outer_value: OuterValue | None = None,
                 batch_value: BatchValue | None = None, menu_builder: MenuBuilder | None = None,
                 hard_constraint: Callable[[Candidate], bool] | None = None):
        self.config = config or SearchConfig()
        self.hard_constraint = hard_constraint
        # Validate incumbent before any paid or learned action.
        accept(incumbent, [incumbent], ledger, scope, self.config.margin, hard_constraint)
        self.state = SearchState(scope, incumbent, {incumbent.uid: incumbent}, ledger, budget)
        self.drafter, self.executor = drafter, executor
        self.outer_value = outer_value or heuristic_outer_value
        self.batch_value = batch_value or heuristic_batch_value
        self.value_source = "learned-or-custom-hooks" if outer_value is not None or batch_value is not None else "heuristic-fallback"
        self.menu_builder = menu_builder or MenuBuilder()
        self.acquirer = ResultAcquirer(ledger, budget, executor)
        self.rng = random.Random(self.config.seed)
        self.search_id = uuid.uuid4().hex
        self.decisions: list[TraceDecision] = []
        self.transitions: list[TraceTransition] = []
        self._finalist_jobs: dict[str, str] = {}
        self._used = False
        self._exploration_spent = 0
        self._transition_spent_before = budget.spent

    def _job(self, name: str) -> str:
        return f"{self.search_id}:{name}:{uuid.uuid4().hex}"

    def _f(self) -> float:
        s = self.state
        return complete_improvement(s.incumbent, s.candidates.values(), s.ledger, s.scope, self.hard_constraint)

    def _sources(self, candidate: Candidate, task: Task) -> tuple[Candidate, ...]:
        if self.config.all_u:
            return ()
        s = self.state
        # Parent first; other paid branch records remain valid sources.
        options = sorted((c for c in s.candidates.values() if c.uid != candidate.uid and s.ledger.get(c, task, s.scope)),
                         key=lambda c: (c.uid != candidate.parent_id, c.uid))
        return tuple(options[:self.config.max_sources])

    def _reserve_finalist(self) -> None:
        if not self.config.reserve_completion:
            return
        s = self.state
        if s.finalist_id is not None:
            current = s.candidates[s.finalist_id]
            for task in s.scope.tasks:
                if task.uid in self._finalist_jobs and s.ledger.get(current, task, s.scope) is not None:
                    job = self._finalist_jobs.pop(task.uid)
                    if job in s.budget.reserved:
                        s.budget.release(job)
            if self._finalist_jobs:
                return
            s.finalist_id = None
        for candidate in reversed(tuple(s.candidates.values())):
            missing = s.ledger.missing(candidate, s.scope)
            if not missing:
                continue
            # Every protected item uses direct all-U execution, never mean reuse cost.
            jobs = {t.uid: self._job("finalist") for t in missing}
            reservations = {jobs[t.uid]: (self.config.caps(t, direct=True).total(), "finalist-completion") for t in missing}
            if sum(v[0] for v in reservations.values()) <= s.budget.available:
                s.budget.reserve_many(reservations)
                self._finalist_jobs = jobs
                s.finalist_id = candidate.uid
                s.history.append({"event": "reserve_finalist", "candidate_id": candidate.uid, "cap": sum(v[0] for v in reservations.values())})
                return

    def _candidate_reserve(self, candidate: Candidate) -> int:
        if self.state.finalist_id != candidate.uid:
            return 0
        reservations = self.state.budget.reserved
        return sum(reservations.get(j, 0) for j in self._finalist_jobs.values())

    def _problem(self, candidate: Candidate, available: int) -> tuple[BatchProblem, tuple[str, ...]]:
        s = self.state
        options, conflicts, direct_ids = [], [], []
        for task in s.ledger.missing(candidate, s.scope):
            direct_id = f"direct:{task.uid}"
            direct_ids.append(direct_id)
            # Individual direct requests need no shared setup; executor cap includes
            # its own mandatory runtime setup, protecting final completion as well.
            cap = self.config.caps(task, direct=True).total()
            reserved_cap = cap if s.finalist_id == candidate.uid and task.uid in self._finalist_jobs else 0
            reservation_id = self._finalist_jobs.get(task.uid) if reserved_cap else None
            value = 1. + s.forecasts.get(candidate.uid, {}).get(task.uid, 0.)
            options.append(CheckOption(direct_id, candidate.uid, (task.uid,), cap, "direct", value, "direct", reserved_cap=reserved_cap,
                                       reservation_id=reservation_id))
            sources = self._sources(candidate, task)
            # The direct fallback is the exact physical work already protected;
            # checking adds a new separately affordable cap to that reservation.
            if sources:
                acquire_id = f"acquire:{task.uid}"
                cap = self.config.caps(task).total(len(sources))
                options.append(CheckOption(acquire_id, candidate.uid, (task.uid,), cap, "reuse", value + .1,
                                           "acquire", tuple(c.uid for c in sources), reserved_cap, reservation_id))
                conflicts.append((direct_id, acquire_id))
        problem = BatchProblem(tuple(options), {"direct": 0, "reuse": self.config.setup_cost}, max(0, available),
                               conflicts=tuple(conflicts), state_version=s.version)
        return problem, tuple(direct_ids)

    def _actions(self) -> list[Action]:
        s, cfg = self.state, self.config
        raw: list[Action] = []
        for candidate in s.candidates.values():
            if len(s.candidates) < cfg.max_candidates and s.children_drafted.get(candidate.uid, 0) < cfg.max_children:
                raw.append(Action("draft", candidate.uid, cfg.draft_cap))
            if s.ledger.missing(candidate, s.scope):
                overhead = cfg.planning_cost + self.menu_builder.max_options * cfg.batch_value_cap
                problem, _ = self._problem(candidate, s.budget.available)
                if problem.options:
                    cheapest = min(problem.cost((o.uid,)) for o in problem.options)
                    # Existing finalist caps fund exactly its direct operations.
                    raw.append(Action("verify", candidate.uid, overhead + cheapest))
        # Reserve enough selection cost even before the legal support is known.
        selection_cap = len(raw) * cfg.outer_value_cap
        legal = [a for a in raw if a.work_cap + selection_cap <= s.budget.available]
        return [Action(a.kind, a.candidate_id, a.work_cap,
                       selection_cap + a.work_cap if a.kind == "draft" else
                       s.budget.available + self._candidate_reserve(s.candidates[a.candidate_id])) for a in legal]

    def _estimate(self, result: float | ValueEstimate, cap: int) -> tuple[float, int]:
        if isinstance(result, ValueEstimate):
            value, actual = result.value, result.actual_cost
        else:
            value, actual = float(result), cap  # declared logical work cost
        if not math.isfinite(value):
            raise ValueError("nonfinite policy value")
        return value, actual

    def _select_outer(self, actions: list[Action]) -> tuple[Action, bool]:
        s, cfg = self.state, self.config
        job = self._job("outer-selection")
        s.budget.reserve(job, len(actions) * cfg.outer_value_cap, "outer-selection")
        decision_features = s.features()
        values, actual = [], 0
        try:
            for action in actions:
                value, cost = self._estimate(self.outer_value(s, action), cfg.outer_value_cap)
                actual += cost
                if cost > cfg.outer_value_cap:
                    s.budget.breached = True
                    raise CostCapViolation("outer head exceeded declared inference cap")
                values.append(value)
            s.budget.settle(job, actual)  # selection is paid before chosen work starts
        except Exception:
            if job in s.budget.reserved:
                s.budget.settle_unknown(job)
            raise
        support = tuple(a.uid for a in actions)
        selected, probabilities = choose(support, values, mode=cfg.mode, rng=self.rng,
                                         exploration=cfg.exploration, temperature=cfg.temperature)
        explored = False
        if cfg.mode == "deploy":
            # Fresh complete rescoring gives the exact current heap maximum.
            heap = [(-value, action.uid, s.version, action) for value, action in zip(values, actions)]
            heapq.heapify(heap)
            selected = heapq.heappop(heap)[3].uid
            exploratory = [a for a in actions if a.full_transition_cap <= cfg.exploration_budget - self._exploration_spent]
            if exploratory and self.rng.random() < cfg.exploration:
                selected = self.rng.choice(exploratory).uid
                explored = True
            # Deployment actions are not actor-training likelihood events.
            probabilities = tuple(float(x == selected) for x in support)
        self.decisions.append(TraceDecision("outer", support, selected, tuple(probabilities), tuple(values), s.version,
              len(self.transitions), cfg.temperature, cfg.exploration if cfg.mode == "collect" else 0.,
              state_features=decision_features, option_features=tuple((float(a.kind == "verify"), float(a.work_cap)) for a in actions),
              stochastic=cfg.mode == "collect", value_source=self.value_source))
        return next(a for a in actions if a.uid == selected), explored

    def _draft(self, action: Action) -> list[dict]:
        # Eq. (9), one physical edit in a bounded path:
        # e_{a,d} ~ pi_theta(. | c_{a,d-1}, H_{a,d}),
        # c_{a,d}=Freeze(Apply(c_{a,d-1}, e_{a,d})).
        # ``parent`` is the actual immutable c_{a,d-1}, and ``s`` exposes only
        # currently observed artifacts/results. The returned child must bind its
        # parent_id to this exact parent. Expense is settled before deciding
        # whether a child exists, so invalid sampled edits remain paid work.
        s = self.state
        parent = s.candidates[action.candidate_id]
        job = self._job("draft")
        s.budget.reserve(job, self.config.draft_cap, "draft")
        try:
            result = self.drafter.draft(parent, s, self.config.draft_cap, self.rng)
            s.budget.settle(job, result.actual_cost)
        except Exception:
            if job in s.budget.reserved:
                s.budget.settle_unknown(job)
            raise
        s.children_drafted[parent.uid] = s.children_drafted.get(parent.uid, 0) + 1
        events = [{"event": "draft", "parent_id": parent.uid, "status": result.status, "cost": result.actual_cost, "error": result.error}]
        if result.candidate is not None:
            child = result.candidate
            if child.parent_id != parent.uid:
                raise ValueError("frozen draft does not identify its actual parent")
            if not set(result.forecasts).issubset({t.uid for t in s.scope.tasks}):
                raise ValueError("surface forecast references task outside selection scope")
            if child.uid in s.candidates:
                events.append({"event": "duplicate_draft", "candidate_id": child.uid})
            else:
                s.candidates[child.uid] = child
                s.ledger.register(child, s.scope)
                s.forecasts[child.uid] = dict(result.forecasts)
                events.append({"event": "freeze", "candidate_id": child.uid, "forecasts": dict(result.forecasts)})
        if result.edit_decision is not None:
            self.decisions.append(result.edit_decision)
        return events

    def _verify(self, action: Action) -> list[dict]:
        s, cfg = self.state, self.config
        candidate = s.candidates[action.candidate_id]
        cost_before_origin = s.budget.spent - self._transition_spent_before
        planning_job = self._job("planning")
        s.budget.reserve(planning_job, cfg.planning_cost, "planning")
        # Menu feasibility includes pending planning/ranking expense, once each.
        ranking_cap = self.menu_builder.max_options * cfg.batch_value_cap
        available = s.budget.available - ranking_cap
        problem, completion = self._problem(candidate, available)
        try:
            menu = self.menu_builder.build(problem, completion_ids=completion)
        finally:
            s.budget.settle(planning_job, cfg.planning_cost)
        events = [{"event": "menu", "candidate_id": candidate.uid, "solver_status": menu.solver_status,
                   "proposal_snapshot": menu.proposal_snapshot, "state_version": menu.state_version,
                   "batches": [{"id": b.uid, "options": b.option_ids, "origin": b.origin} for b in menu.batches],
                   "problem": {"budget": problem.budget, "options": [asdict(o) for o in problem.options],
                               "setup_costs": thaw(problem.setup_costs), "dependencies": problem.dependencies, "conflicts": problem.conflicts}}]
        if not menu.batches:
            return events
        job = self._job("batch-ranking")
        s.budget.reserve(job, len(menu.batches) * cfg.batch_value_cap, "batch-ranking")
        decision_features = s.features()
        values, actual = [], 0
        try:
            for batch in menu.batches:
                value, cost = self._estimate(self.batch_value(s, candidate, batch, problem), cfg.batch_value_cap)
                actual += cost
                if cost > cfg.batch_value_cap:
                    s.budget.breached = True
                    raise CostCapViolation("batch head exceeded declared inference cap")
                values.append(value)
            s.budget.settle(job, actual)
        except Exception:
            if job in s.budget.reserved:
                s.budget.settle_unknown(job)
            raise
        support = tuple(b.uid for b in menu.batches)
        selected, probabilities = choose(support, values, mode=cfg.mode, rng=self.rng,
                                         exploration=cfg.exploration, temperature=cfg.temperature)
        self.decisions.append(TraceDecision("batch", support, selected, tuple(probabilities), tuple(values), s.version,
             len(self.transitions), cfg.temperature, cfg.exploration if cfg.mode == "collect" else 0., menu.proposal_snapshot,
             state_features=decision_features, option_features=tuple((float(len(b.option_ids)), float(problem.cost(b.option_ids))) for b in menu.batches),
             planning_cost_paid=cfg.planning_cost, stochastic=cfg.mode == "collect", value_source=self.value_source,
             cost_before_origin=cost_before_origin))
        batch = next(b for b in menu.batches if b.uid == selected)
        # Revalidate after ranking, including state version and current budget.
        available_now = s.budget.available
        if menu.state_version != s.version or not problem.feasible(batch.option_ids, budget=available_now):
            events.append({"event": "stale_or_infeasible_batch", "batch_id": batch.uid})
            return events
        return events + self._execute_batch(candidate, batch, problem)

    def _execute_batch(self, candidate: Candidate, batch: Batch, problem: BatchProblem) -> list[dict]:
        s, cfg = self.state, self.config
        by_id = problem.by_id
        order = problem.dependency_order(batch.option_ids)
        groups = {by_id[i].group for i in order}
        setup_jobs = {g: self._job("setup") for g in groups}
        option_jobs: dict[str, str] = {}
        transfers: dict[str, int] = {}
        new_reservations = {setup_jobs[g]: (problem.setup_costs[g], "shared-setup") for g in groups}
        for option_id in order:
            option = by_id[option_id]
            if len(option.task_ids) != 1 or option.kind == "prerequisite":
                raise ValueError("default runner executes single-item result requests; custom compound operations need a trusted adapter")
            task_id = option.task_ids[0]
            existing_job = self._finalist_jobs.get(task_id) if s.finalist_id == candidate.uid else None
            if existing_job:
                if s.budget.reserved.get(existing_job) != option.reserved_cap:
                    raise ValueError("reserved work transfer does not match direct completion")
                option_jobs[option_id] = existing_job
                transfers[existing_job] = option.cap
            else:
                job = self._job("request")
                option_jobs[option_id] = job
                new_reservations[job] = option.cap, "acquisition"
        s.budget.reserve_work(new_reservations, transfers)
        events = []
        try:
            for group in sorted(groups):
                s.budget.settle(setup_jobs[group], problem.setup_costs[group])
                events.append({"event": "shared_setup", "group": group, "cost": problem.setup_costs[group]})
            for option_id in order:
                option = by_id[option_id]
                task = next(t for t in s.scope.tasks if t.uid == option.task_ids[0])
                sources = tuple(s.candidates[x] for x in option.source_ids)
                result = self.acquirer.acquire(candidate, task, s.scope, sources, cfg.caps(task, direct=option.kind == "direct"),
                                               reservation=option_jobs[option_id])
                self._finalist_jobs.pop(task.uid, None) if s.finalist_id == candidate.uid else None
                events.append({"event": "acquisition", "candidate_id": candidate.uid, "task_id": task.uid,
                    "status": result.status, "route": result.record.route if result.record else None, "cost": result.actual_cost,
                    "from_cache": result.from_cache, "resume_token": result.resume_token, "error": result.error,
                    "checks": [{"status": c.status, "actual_cost": c.actual_cost, "reason": c.reason,
                                "source_id": c.certificate.source_id if c.certificate else None,
                                "checker_id": s.ledger.checker.version} for c in result.checks]})
        finally:
            # Cancel unstarted nonfinalist jobs after an exception. Reserved finalist
            # work stays protected if it has not physically started.
            for job in new_reservations:
                if job in s.budget.reserved:
                    s.budget.release(job)
        return events

    def run(self) -> SearchResult:
        if self._used:
            raise ValueError("SearchEngine instances are single-use; independent searches need independent mutable state")
        self._used = True
        s, cfg = self.state, self.config
        initial_spent = s.budget.spent
        initial_charge_count = len(s.budget.charges)
        initial_f = self._f()
        # A GVO group's fixed context includes observed outcomes, not just their
        # keys: stochastic incumbents can produce different receipts/results under
        # the same candidate and task identities. Local signing keys are excluded
        # because isolated host ledgers may authenticate the same receipt anew.
        initial_records = []
        for key in sorted(s.ledger.records):
            manifest = s.ledger.records[key].manifest()
            manifest.pop("signature")
            initial_records.append(manifest)
        initial_context = digest({"incumbent": s.incumbent.uid, "scope": s.scope.uid,
                                  "initial_records": initial_records, "budget": s.budget.total,
                                  "spent": s.budget.spent, "pending": s.budget.pending,
                                  "checker_version": s.ledger.checker.version})
        stop_reason = "action_limit"
        for _ in range(cfg.max_actions):
            self._reserve_finalist()
            actions = self._actions()
            if not actions:
                stop_reason = "no_affordable_legal_action"
                break
            before, f_before, version_before = s.budget.spent, self._f(), s.version
            self._transition_spent_before = before
            action, explored = self._select_outer(actions)
            events = self._draft(action) if action.kind == "draft" else self._verify(action)
            s.version += 1
            s.history.extend(events)
            self._reserve_finalist()
            cost = s.budget.spent - before
            if explored:
                self._exploration_spent += cost
                if self._exploration_spent > cfg.exploration_budget:
                    raise CostCapViolation("deployment exploration exceeded its separately declared budget")
            self.transitions.append(TraceTransition(len(self.transitions), action.kind, action.candidate_id, version_before,
                                                    s.version, cost, f_before, self._f(), tuple(events)))
            if not s.budget.invariant():
                raise CostCapViolation("budget invariant lost")
        # Once regular search stops, transfer each protected all-U reservation once.
        if s.finalist_id is not None:
            finalist = s.candidates[s.finalist_id]
            for task in s.scope.tasks:
                job = self._finalist_jobs.get(task.uid)
                if job is None:
                    continue
                before, f_before, version_before = s.budget.spent, self._f(), s.version
                result = self.acquirer.acquire(finalist, task, s.scope, (), cfg.caps(task, direct=True), reservation=job)
                del self._finalist_jobs[task.uid]
                s.version += 1
                event = {"event": "finalist_completion", "candidate_id": finalist.uid, "task_id": task.uid,
                         "status": result.status, "route": result.record.route if result.record else None, "cost": result.actual_cost}
                s.history.append(event)
                self.transitions.append(TraceTransition(len(self.transitions), "finalist_completion", finalist.uid,
                     version_before, s.version, s.budget.spent - before, f_before, self._f(), (event,), "completion"))
        outcome = accept(s.incumbent, s.candidates.values(), s.ledger, s.scope, cfg.margin, self.hard_constraint)
        if not s.budget.invariant():
            raise CostCapViolation("budget invariant lost at acceptance")
        total = s.budget.spent - initial_spent
        if total != sum(t.cost for t in self.transitions):
            raise RuntimeError("transition accounting omitted physical spending")
        return SearchResult(outcome, initial_context, self.search_id, cfg.seed, cfg.group_id, initial_f, self._f(), total,
                 cfg.cost_unit, tuple(self.transitions), tuple(self.decisions), tuple(asdict(c) for c in s.budget.charges[initial_charge_count:]),
                 stop_reason, tuple(s.candidates), s.budget.available, s.ledger.to_json(), self.value_source)
