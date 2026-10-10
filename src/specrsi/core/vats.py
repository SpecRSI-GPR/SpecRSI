"""Executable VATS inner search, Algorithms 1--2.

Eq. (4) defines an independent search under a frozen improver:
    (c_{k+1}^{(g)}, D_k^{(g)}) = Search_{Theta_k}(c_k), g=1,...,G,
    D_k = union_{g=1}^G D_k^{(g)},
    Theta_{k+1} = GVO(Theta_k, D_k).
Here k indexes outer updates, g indexes independent complete searches from the
same root, and G>=2 is the group size. A leaf of a shared tree is not an
independent search. Each D_k^{(g)} includes selected plans, sampled edits,
observations and all paid preparation/work; completion means the search has
terminated and returned its selected agent, not that every visited node passed.
Theta_k=(theta_k, psi_k, phi_k, nu) stays fixed throughout one run; the outer
updates reside in learning/. The scorer predicts future verified gain minus
resource costs; the independent policy emits preference logits. Neither can
write evidence, which must come from Acquire_Gamma or the fixed backend.

Implementation map: _prepare's menu is Eq. (5), completion seeds satisfy
Eqs. (14)--(15), whole-bundle refinement is Eq. (7), paid-state selection is
Eq. (8), _execute_plan's actual parent-child edits are Eq. (9), and acceptance
implements Eq. (10).

At one decision the observed state is S=(T,E,H,B,N,w), T=(C,A), with immutable
version nodes C, executed edit edges A, admitted records E, observed history H,
remaining expense B, remaining primitive attempts N, and worker count w.
S_t^- is before preparation, S_t is after actual preparation payment, and
S_tilde conservatively reserves its entire enforceable cap. These are distinct
conditioning states; replacing one with another changes value targets and the
recorded action probability.

Execution uses synchronous blocks: the selected work finishes before the next
policy choice. With multiple workers, adapters declaring isolated concurrent
execution run independent verification jobs in dependency waves. The coordinator
settles every launched operation, admits validated outcomes, and records both
worker durations and elapsed block time. Shared setups are paid once per block;
only the single elapsed clock enters the continuation's time penalty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from concurrent.futures import ThreadPoolExecutor
import copy
from itertools import combinations, zip_longest
import math
import random
import time
from typing import Callable, Sequence
from types import MappingProxyType

from .acceptance import accept
from .acquisition import ResultAcquirer
from .budget import BudgetLedger, CostCapViolation, cost_int
from .models import Candidate, digest
from .jobs import JobContext, JobExecutionResult, SetupExecutionResult, VerificationBackend
from .search import Action, SearchConfig, SearchEngine, SearchResult, SearchState, TraceTransition, ValueEstimate
from specrsi.planning.completion import CompletionProblem, VerificationJob, solve_completion
from specrsi.planning.policy import TraceDecision, choose


class ScorerQueryFailure(RuntimeError):
    """A failed model call with measured expense, safe to retain as missing Q."""
    def __init__(self, message: str, actual_cost: int):
        super().__init__(message)
        self.actual_cost = cost_int(actual_cost)


@dataclass(frozen=True)
class VATSConfig(SearchConfig):
    all_u: bool = True
    reserve_completion: bool = False
    max_attempts: int = 100
    max_draft_depth: int = 2
    max_draft_parents: int = 2
    max_draft_plans: int = 8
    preparation_cap: int | None = None
    scorer_query_cap: int = 1
    policy_call_cost: int = 1
    solver_call_cost: int = 1
    max_queries: int = 96
    max_extensions: int = 4
    max_menu_options: int = 32
    max_completion_targets: int = 4
    max_pair_targets: int = 4
    max_singletons: int = 4
    solver_backend: str = "scipy"
    solver_time_limit: float = 1.
    workers: int = 1
    scorer_snapshot: str = "frozen-scorer"
    policy_snapshot: str = "frozen-policy"
    menu_snapshot: str = "pdf-vats-v1"

    def __post_init__(self):
        super().__post_init__()
        for name in ("max_attempts", "max_draft_depth", "max_draft_parents", "max_draft_plans", "scorer_query_cap",
                     "policy_call_cost", "solver_call_cost", "max_queries", "max_extensions", "max_menu_options",
                     "max_completion_targets", "max_pair_targets", "max_singletons", "workers"):
            cost_int(getattr(self, name))
        if self.preparation_cap is not None:
            cost_int(self.preparation_cap)
        if self.margin != 0 or self.reserve_completion:
            raise ValueError("PDF V ATS uses root-eligible argmax without a margin or forced final completion")
        if self.all_u is not True:
            raise ValueError("PDF V ATS requires direct execution; cross-candidate legacy reuse is unsupported")
        if self.workers < 1 or self.max_draft_depth < 1 or self.max_draft_parents < 1 or self.max_draft_plans < 1:
            raise ValueError("workers and draft shape bounds must be positive")
        if not 0 <= self.exploration < 1:
            raise ValueError("PDF policy exploration must lie in [0, 1)")
        if self.max_menu_options < 2 + self.max_completion_targets + self.max_pair_targets + self.max_singletons:
            raise ValueError("menu cap cannot retain promised completion seeds, ACCEPT and a single edit")
        if self.solver_backend not in {"scipy", "exhaustive"} or not math.isfinite(self.solver_time_limit) or self.solver_time_limit <= 0:
            raise ValueError("invalid completion solver")
        if self.prep_cap < self.planning_cost + self.policy_call_cost:
            raise ValueError("preparation cap must cover fixed preparation and final policy call")

    @property
    def prep_cap(self) -> int:
        if self.preparation_cap is not None:
            return self.preparation_cap
        return (self.planning_cost + (self.max_completion_targets + self.max_pair_targets) * self.solver_call_cost
                + self.max_queries * self.scorer_query_cap + self.policy_call_cost)


@dataclass
class VATSState(SearchState):
    remaining_attempts: int = 0
    initial_attempts: int = 0
    workers: int = 1
    current_transition_index: int = 0
    hard_constraint: Callable[[Candidate], bool] | None = None
    job_catalog: CompletionProblem | None = None

    def features(self) -> tuple[float, ...]:
        # Eq. (3): F(S)=max_{c in C_full(S)} [J_Gamma(c)-J_Gamma(r)].
        # J_Gamma is Ledger.score, including its declared operating penalty
        # exactly once. Missing or inadmissible versions contribute no score;
        # the complete admissible root guarantees F(S)>=0.
        gain = max((self.ledger.score(c, self.scope) - self.ledger.score(self.incumbent, self.scope)
                    for c in self.candidates.values() if self.ledger.admissible(c, self.scope)
                    and (self.hard_constraint is None or self.hard_constraint(c))), default=0.)
        return (*super().features(), self.remaining_attempts / max(1, self.initial_attempts), float(self.workers), gain)


@dataclass(frozen=True)
class VATSPlan:
    """A bounded physical work descriptor from the current immutable tree.

    Eq. (5): Omega_t=D_t union {VERIFY(A):A in M_t} union {ACCEPT}.
    D_t contains DRAFT(P,h), with P=(P_a)_{a=1}^{|P|} tagging existing parents and
    h>=1 bounding dependent edit depth. M_t contains feasible nonempty job sets A
    on existing versions; these job sets differ from the version-tree edge set A.
    ``parent_ids`` is P, ``depth`` is h, and ``jobs`` identifies the verification
    bundle. ``work_cap`` bounds expense in the experiment's declared integer unit;
    ``attempt_cap`` bounds primitive attempts. Neither cap promises successful
    edits/checks or supplies evidence. ACCEPT has zero physical work and both caps
    zero. Plan identity canonicalizes physical work so duplicates share one menu
    entry and one cached prediction at a fixed proposal state.
    """
    kind: str
    parent_ids: tuple[str, ...] = ()
    depth: int = 1
    jobs: tuple[VerificationJob, ...] = ()
    work_cap: int = 0
    attempt_cap: int = 0
    origin: str = ""

    def __post_init__(self):
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(self, "jobs", tuple(sorted(self.jobs, key=lambda j: j.uid)))
        cost_int(self.work_cap)
        cost_int(self.attempt_cap)
        if type(self.depth) is not int:
            raise ValueError("draft depth must be an integer")
        if self.kind not in {"accept", "draft", "verify"}:
            raise ValueError("unknown V ATS plan")
        if (len(set(self.parent_ids)) != len(self.parent_ids)
                or len({j.uid for j in self.jobs}) != len(self.jobs)):
            raise ValueError("duplicate physical work in plan")
        if self.kind == "draft" and (not self.parent_ids or self.depth < 1 or self.jobs):
            raise ValueError("draft plans require existing parents and positive bounded depth")
        if self.kind == "verify" and (not self.jobs or self.parent_ids):
            raise ValueError("verification plans require jobs only")
        if self.kind == "accept" and (self.jobs or self.parent_ids or self.work_cap or self.attempt_cap):
            raise ValueError("ACCEPT has no physical work")

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(j.uid for j in self.jobs)

    @property
    def candidate_id(self) -> str:
        """A convenience for adapters; cross-branch plans use an empty identity."""
        ids = set(self.parent_ids) | {c for j in self.jobs for c, _ in j.keys}
        return next(iter(ids)) if len(ids) == 1 else ""

    @property
    def uid(self) -> str:
        if self.kind == "accept":
            return "ACCEPT"
        return self.kind + ":" + digest({"parents": self.parent_ids, "depth": self.depth if self.kind == "draft" else 0,
                                       "jobs": self.job_ids})

    def manifest(self) -> dict:
        return {"id": self.uid, **asdict(self)}

    def features(self, state: SearchState) -> tuple[float, ...]:
        keys = {key for job in self.jobs for key in job.keys}
        candidates = {c for c, _ in keys}
        completed = sum({t.uid for t in state.ledger.missing(state.candidates[c], state.scope)}
                        .issubset({t for candidate, t in keys if candidate == c}) for c in candidates)
        parent_missing = sum(len(state.ledger.missing(state.candidates[c], state.scope)) for c in self.parent_ids)
        # Only actual existing parents and requested jobs enter these features.
        return (float(self.kind == "accept"), float(self.kind == "draft"), float(self.kind == "verify"),
                float(len(self.parent_ids)), float(self.depth if self.kind == "draft" else 0), float(len(self.jobs)),
                float(self.work_cap), float(self.attempt_cap), float(len(candidates)), float(completed),
                float(len(keys)), float(parent_missing))


def _clone_state(state: VATSState) -> VATSState:
    return replace(state, ledger=state.ledger.clone(), budget=state.budget.clone(),
                   candidates=dict(state.candidates), forecasts=copy.deepcopy(state.forecasts),
                   history=copy.deepcopy(state.history), children_drafted=dict(state.children_drafted))


@dataclass(frozen=True)
class PaidStateSnapshot:
    """Saved post-preparation host state, with exactly the original legal menu.

    Runtime isolation is deliberately not faked by copying a Python executor
    reference. ``from_paid_state`` requires an independently restored executor
    and drafter; environment-copy costs belong to the caller's training ledger.
    """
    state: VATSState
    plans: tuple[VATSPlan, ...]
    problem: CompletionProblem
    config: VATSConfig
    remaining_actions: int
    target_cursor: int
    resume_tokens: tuple[tuple[tuple[str, str], str], ...]
    job_resume_tokens: tuple[tuple[str, str], ...] = ()
    backend_version: str | None = None

    @property
    def uid(self) -> str:
        return digest({"scope": self.state.scope.uid, "root": self.state.incumbent.uid,
                       "candidates": sorted(self.state.candidates), "history": self.state.history,
                       "ledger": self.state.ledger.to_json(), "budget": self.state.budget.available,
                       "attempts": self.state.remaining_attempts, "workers": self.state.workers,
                       "menu": [p.uid for p in self.plans], "menu_snapshot": self.config.menu_snapshot,
                       "backend_version": self.backend_version})


def heuristic_vats_score(state: VATSState, plan: VATSPlan) -> float:
    """Compute a deterministic continuation-priority feature combination.

    ACCEPT returns zero. For DRAFT, the value is
        0.3 / max(1, sum_{c in P}(children_drafted(c)+1)),
    which decreases as selected parents receive more expansion attempts.
    For VERIFY, the value is
        n_complete + 0.05*n_keys - 0.05*work_cap/max(1,total_budget),
    where n_complete counts candidates whose missing keys are covered by this
    plan and n_keys counts distinct requested candidate-item keys. The expense
    fraction is dimensionless. These features describe planned work and do not
    change the observed completeness mask or supply evaluation evidence.
    """
    if plan.kind == "accept":
        return 0.
    if plan.kind == "draft":
        return .3 / max(1, sum(state.children_drafted.get(c, 0) + 1 for c in plan.parent_ids))
    features = plan.features(state)
    # Caps may be tokens, primitive units or milliseconds. A fixed coefficient
    # on raw units would reject every native-scale verification in favor of 0.
    return features[9] + .05 * features[10] - .05 * plan.work_cap / max(1, state.budget.total)


def heuristic_vats_preference(state: VATSState, plan: VATSPlan) -> float:
    # Use the deterministic feature combination as the policy logit z_psi for
    # this callback. Selection applies Eq. (8) to these logits after preparation;
    # the independently supplied scorer callback still controls bundle refinement.
    return heuristic_vats_score(state, plan)


class VATSEngine(SearchEngine):
    def __init__(self, scope, incumbent, ledger, budget, executor, drafter, config: VATSConfig | None = None, *,
                 scorer: Callable | None = None, policy: Callable | None = None,
                 policy_menu: Callable[[VATSState, tuple[VATSPlan, ...]], Sequence[float]] | None = None,
                 hard_constraint=None, capture_paid_states: bool = False,
                 verification_backend: VerificationBackend | None = None):
        config = config or VATSConfig()
        if not isinstance(config, VATSConfig):
            raise TypeError("VATSEngine requires VATSConfig")
        if policy is not None and policy_menu is not None:
            raise ValueError("provide one independent policy interface")
        super().__init__(scope, incumbent, ledger, budget, executor, drafter, config, hard_constraint=hard_constraint)
        self.state = VATSState(scope, incumbent, {incumbent.uid: incumbent}, ledger, budget,
                               remaining_attempts=config.max_attempts, initial_attempts=config.max_attempts,
                               workers=config.workers, hard_constraint=hard_constraint)
        self.scorer = scorer or heuristic_vats_score
        self.policy = policy or heuristic_vats_preference
        self.policy_menu = policy_menu
        self.value_source = "learned-or-custom-vats-hooks" if scorer is not None or policy is not None or policy_menu is not None else "heuristic-fallback"
        self._resume_tokens: dict[tuple[str, str], str] = {}
        self._target_cursor = 0
        # Target reservoir randomness must not consume the actor's random stream.
        self._target_rng = random.Random(config.seed ^ 0x56415453)
        self.capture_paid_states = capture_paid_states
        self.paid_snapshots: list[PaidStateSnapshot] = []
        self._forced_start: tuple[VATSPlan, CompletionProblem] | None = None
        self._rounds_executed = 0
        self.verification_backend = verification_backend
        if verification_backend is not None and (not isinstance(verification_backend.version, str) or not verification_backend.version):
            raise ValueError("compound verification backend requires a fixed version identity")
        self._job_resume_tokens: dict[str, str] = {}

    def snapshot_paid_state(self, plans: tuple[VATSPlan, ...], problem: CompletionProblem) -> PaidStateSnapshot:
        """Copy an already paid origin; no outcomes or ungenerated edits are made."""
        if self.state.budget.pending:
            raise ValueError("paid-state snapshots cannot silently clone in-flight physical work")
        return PaidStateSnapshot(_clone_state(self.state), plans, problem, self.config,
                                 self.config.max_actions - self._rounds_executed, self._target_cursor,
                                 tuple(self._resume_tokens.items()), tuple(self._job_resume_tokens.items()),
                                 self.verification_backend.version if self.verification_backend else None)

    @classmethod
    def from_paid_state(cls, snapshot: PaidStateSnapshot, forced_plan: VATSPlan | str, *, executor, drafter,
                        seed: int, scorer=None, policy=None, policy_menu=None, hard_constraint=None,
                        capture_paid_states: bool = False,
                        verification_backend: VerificationBackend | None = None) -> "VATSEngine":
        """Force one recorded legal plan, then resume the frozen continuation law.

        The forced plan has no invented policy likelihood. Prior preparation is
        neither repeated nor charged to its suffix. Each call clones host state;
        callers provide isolated/restored executor and drafter instances and log
        any physical environment-copy charges separately from rollout returns.
        """
        uid = forced_plan.uid if isinstance(forced_plan, VATSPlan) else forced_plan
        plan = next((p for p in snapshot.plans if p.uid == uid), None)
        if plan is None:
            raise ValueError("forced plan is outside the saved paid-state support")
        backend_version = verification_backend.version if verification_backend else None
        if backend_version != snapshot.backend_version:
            raise ValueError("paid continuation must restore the same compound verification backend")
        state = _clone_state(snapshot.state)
        config = replace(snapshot.config, seed=seed, max_actions=snapshot.remaining_actions)
        if hard_constraint is None:
            hard_constraint = state.hard_constraint
        engine = cls(state.scope, state.incumbent, state.ledger, state.budget, executor, drafter, config,
                     scorer=scorer, policy=policy, policy_menu=policy_menu, hard_constraint=hard_constraint,
                     capture_paid_states=capture_paid_states, verification_backend=verification_backend)
        engine.state = state
        engine.acquirer.ledger = state.ledger
        engine._target_cursor = snapshot.target_cursor
        engine._resume_tokens = dict(snapshot.resume_tokens)
        engine._job_resume_tokens = dict(snapshot.job_resume_tokens)
        engine._forced_start = plan, snapshot.problem
        return engine

    def _catalog(self, state: VATSState) -> CompletionProblem:
        # Required missingness and target keys:
        # U_c(S)={i in I:m_ci=0}, K_c(S)={(c,i):i in U_c(S)}.
        # R(S)=Remaining(S) reads masks, candidate/job relations, prerequisites,
        # shared setups and source identities from existing observations only.
        # A job's K_j describes possible requested coverage, not acquired
        # outcomes. Neither catalog construction nor predicted job success can
        # set m_ci=1; acquisition must later return a Valid_Gamma record.
        if self.verification_backend is not None:
            problem = self.verification_backend.catalog(self._job_context(state), budget=state.budget.available,
                                                        attempts=state.remaining_attempts)
            if not isinstance(problem, CompletionProblem):
                raise TypeError("verification catalog must return a CompletionProblem")
            if problem.budget > state.budget.available or problem.attempts > state.remaining_attempts:
                raise ValueError("verification catalog cannot increase available resources")
            allowed_keys = {(c, t.uid) for c in state.candidates for t in state.scope.tasks}
            if any(not set(job.keys).issubset(allowed_keys) for job in problem.jobs):
                raise ValueError("verification catalog refers to an unknown candidate or required item")
            return problem
        jobs = []
        for candidate in state.candidates.values():
            for task in state.ledger.missing(candidate, state.scope):
                sources = self._sources(candidate, task)
                cap = self.config.caps(task).total(len(sources))
                jobs.append(VerificationJob(f"request:{candidate.uid}:{task.uid}", ((candidate.uid, task.uid),),
                                            cap, attempts=1 + len(sources)))
        return CompletionProblem(tuple(jobs), {"execution": self.config.setup_cost}, state.budget.available, state.remaining_attempts)

    @staticmethod
    def _job_context(state: VATSState) -> JobContext:
        valid = {key: record for key, record in state.ledger.records.items() if state.ledger.validate_key(key)}
        return JobContext(state.scope, state.candidates, valid, tuple(state.history))

    def _verify_plan(self, ids, problem, origin="") -> VATSPlan:
        return VATSPlan("verify", jobs=tuple(problem.by_id[i] for i in ids), work_cap=problem.cost(ids),
                        attempt_cap=problem.attempt_cost(ids), origin=origin)

    def _draft_plans(self, state: VATSState) -> list[VATSPlan]:
        cfg = self.config
        parents = [c.uid for c in state.candidates.values() if state.children_drafted.get(c.uid, 0) < cfg.max_children]
        cursor = self._target_cursor % max(1, len(parents))
        parents = parents[cursor:] + parents[:cursor]
        slots = cfg.max_candidates - len(state.candidates)
        plans = []
        # Register single edits before deeper or multi-branch plans, preserving a fallback.
        attempt_limit = min(slots, state.remaining_attempts)
        if cfg.draft_cap:
            attempt_limit = min(attempt_limit, state.budget.available // cfg.draft_cap)
        single = (((p,), 1) for p in parents)
        deep = (((p,), depth) for depth in range(2, min(cfg.max_draft_depth, attempt_limit) + 1) for p in parents)
        branches = ((tuple(ps), depth) for n in range(2, min(cfg.max_draft_parents, len(parents), attempt_limit) + 1)
                    for ps in combinations(parents, n) for depth in range(1, min(cfg.max_draft_depth, attempt_limit // n) + 1))
        # Interleave action families before truncating: many existing parents
        # must not crowd every deeper or multi-parent descriptor out of support.
        shapes = (shape for row in zip_longest(single, deep, branches) for shape in row if shape is not None)
        for selected, depth in shapes:
            attempts = len(selected) * depth
            cap = attempts * cfg.draft_cap
            if attempts <= min(slots, state.remaining_attempts) and cap <= state.budget.available:
                plans.append(VATSPlan("draft", selected, depth, work_cap=cap, attempt_cap=attempts, origin="draft-catalog"))
                if len(plans) >= cfg.max_draft_plans:
                    break
        return plans

    def _conservative_state(self) -> VATSState:
        state = self.state
        # Appendix F.3, Algorithm 2 line 1: S_tilde=S^-[B <- B^- - o_bar].
        # This local budget view reserves the entire preparation cap before any
        # solver/scorer call. All queries share this exact view, not shrinking caps.
        budget = BudgetLedger(state.budget.total)
        budget.spent = state.budget.spent
        budget.reserve("prior-commitments", state.budget.pending + self.config.prep_cap, "preparation-view")
        return replace(state, budget=budget, candidates=dict(state.candidates), history=list(state.history),
                       children_drafted=dict(state.children_drafted))

    def _prepare(self) -> tuple[VATSPlan, CompletionProblem, list[dict]]:
        """Build affordable work, refine its continuation value, then select.

        Eq. (5): Omega_t=D_t union {VERIFY(A):A in M_t} union {ACCEPT}.
        Eq. (6): Q^pi(S_t,omega)=E_pi[F(S_end)-F(S_t)
                                     -lambda*C_{t:end}-mu*L_{t:end}|S_t,omega].
        The continuation law pi includes future edit generation, policy selections,
        checks and branch changes with fixed generator settings. Q values the whole
        remaining continuation, not a candidate's present task score or the immediate
        check reward. C_{t:end} counts future expense, including later preparation and
        failed/unfinished work; L_{t:end} counts elapsed time on one search clock.
        lambda>=0 has score-units/expense-unit and mu>=0 has score-units/second, so each
        subtracted term has the same units as F. Cost-only evaluation sets mu=0.
        Q^pi(S_t,ACCEPT)=0 because no further work or paid preparation is performed.

        Eq. (7): Delta_phi_bar(j|A,S_tilde)=
            Q_phi_bar(S_tilde,VERIFY(A union D_j))
            - Q_phi_bar(S_tilde,VERIFY(A)),
        where D_j includes j and all its prerequisites. Both terms use the same
        conservative state S_tilde and frozen scorer phi_bar. Whole-plan differences
        preserve complementary checks, shared setups and competing candidate gains.
        The query counter q counts attempted uncached scorer calls; failed calls leave
        missing values, not zero predictions. An affordable original completion seed
        remains eligible even if no positive extension is found.

        After debiting actual preparation o_t, Eq. (8) uses the paid S_t:
            pi_psi(omega|S_t,R_t,Omega_t) =
              (1-epsilon)*exp(z_psi(S_t,R_t,omega)/T_pol)
                /sum_{v in Omega_t}exp(z_psi(S_t,R_t,v)/T_pol)
              +epsilon/|Omega_t|,
        with T_pol>0, 0<=epsilon<1 and z_psi(S_t,R_t,ACCEPT)=0.
        z_psi is a preference logit independent of Q_phi. Collection samples this
        mixture; deployment takes argmax z_psi. The exact canonical menu, paid state
        and chosen probability are retained for later same-support likelihood replay.
        """
        s, cfg = self.state, self.config
        proposal_state = self._conservative_state()
        preparation_job = self._job("vats-preparation")
        s.budget.reserve(preparation_job, cfg.prep_cap, "preparation")
        expense = cfg.planning_cost
        try:
            problem = self._catalog(proposal_state)
            # Both value refinement and the paid policy observe the same fixed
            # physical work graph. Reading it supplies no evaluation evidence.
            proposal_state.job_catalog = problem
            s.job_catalog = problem
        except Exception:
            # Remaining() is a fixed-price local read, even when its contract is
            # rejected. It is forbidden from launching remote outcome queries.
            s.budget.settle(preparation_job, expense)
            raise
        events = [{"event": "preparation_contract", "cap": cfg.prep_cap, "proposal_budget": proposal_state.budget.available,
                   "scorer_snapshot": cfg.scorer_snapshot, "policy_snapshot": cfg.policy_snapshot,
                   "proposal_snapshot": cfg.menu_snapshot}]
        seeds: dict[str, VATSPlan] = {}
        values: dict[str, float | None] = {}
        queries = 0

        def affordable(cost):
            return expense + cost + cfg.policy_call_cost <= cfg.prep_cap

        def query(plan):
            nonlocal queries, expense
            if plan.uid in values:
                return values[plan.uid]
            if queries >= cfg.max_queries or not affordable(cfg.scorer_query_cap):
                return None
            queries += 1
            error = None
            try:
                estimate = self.scorer(proposal_state, plan)
                value, cost = (None, cfg.scorer_query_cap) if estimate is None else self._estimate(estimate, cfg.scorer_query_cap)
            except ScorerQueryFailure as failure:
                value, cost, error = None, failure.actual_cost, str(failure)
            expense += cost
            if cost > cfg.scorer_query_cap:
                s.budget.breached = True
                raise CostCapViolation("scorer exceeded the bounded query cost")
            values[plan.uid] = value
            events.append({"event": "scorer_query", "plan_id": plan.uid, "value": value, "cost": cost, "error": error})
            return value

        try:
            # Completion targets are the *entire* contracted missing set. A
            # restricted backend catalog cannot silently redefine completion.
            by_candidate = {candidate.uid: {(candidate.uid, task.uid) for task in s.ledger.missing(candidate, s.scope)}
                            for candidate in s.candidates.values() if s.ledger.missing(candidate, s.scope)}
            ids = list(by_candidate)
            # Mix round-robin coverage with a separately seeded reservoir. Target
            # selection reads only current records, before acquiring new outcomes.
            start = self._target_cursor % max(1, len(ids))
            ordered = ids[start:] + ids[:start]
            self._target_cursor += cfg.max_completion_targets
            count = min(cfg.max_completion_targets, len(ordered))
            rotating = ordered[:(count + 1) // 2]
            reservoir = self._target_rng.sample(ordered[len(rotating):], count - len(rotating))
            targets = [("completion", by_candidate[c]) for c in rotating + reservoir]
            pairs = list(combinations(ordered, 2))
            targets += [("cross-branch-completion", by_candidate[a] | by_candidate[b])
                        for a, b in self._target_rng.sample(pairs, min(len(pairs), cfg.max_pair_targets))]
            for origin, target in targets:
                if not affordable(cfg.solver_call_cost):
                    break
                # Algorithm 2/F.3 charges every attempted MILP, including a
                # runtime solver failure. Such a failure supplies no seed; the
                # independently checked singleton/draft/ACCEPT support remains
                # usable. Invalid configuration errors still propagate.
                expense += cfg.solver_call_cost
                solve_error = None
                try:
                    solution = solve_completion(problem, target, backend=cfg.solver_backend,
                                                time_limit=cfg.solver_time_limit)
                except RuntimeError as failure:
                    solution, solve_error = None, str(failure)
                events.append({"event": "completion_solve", "target": sorted(target), "cost": cfg.solver_call_cost,
                               "status": solution.status if solution else "solver-error" if solve_error else "no-feasible-incumbent",
                               "optimal": solution.optimal if solution else False, "error": solve_error})
                if solution is not None:
                    plan = self._verify_plan(solution.job_ids, problem, origin)
                    seeds.setdefault(plan.uid, plan)
            singleton_count = 0
            for job in problem.jobs if cfg.max_singletons else ():
                # A prerequisite is operational work, not the promised fallback
                # check. Include it through a result job's dependency closure so
                # catalog ordering cannot spend every singleton slot on setup.
                if job.kind != "result":
                    continue
                ids = problem.closure((job.uid,))
                if problem.feasible(ids):
                    plan = self._verify_plan(ids, problem, "singleton")
                    seeds.setdefault(plan.uid, plan)
                    singleton_count += 1
                    if singleton_count >= cfg.max_singletons:
                        break
            refined: dict[str, VATSPlan] = {}
            for seed in seeds.values():
                # Eq. (7): Delta_phi(j | A, S_tilde) =
                # Q_phi(S_tilde, VERIFY(A union D_j)) - Q_phi(S_tilde, VERIFY(A)).
                # Both values use the same frozen scorer and conservative state.
                # Comparing whole plans preserves complementary-check effects;
                # summing isolated job values would repeat the base continuation.
                current = seed
                value = query(current)
                for _ in range(cfg.max_extensions):
                    if value is None or queries >= cfg.max_queries:
                        break
                    options = []
                    for job in problem.jobs:
                        ids = problem.closure((*current.job_ids, job.uid))
                        if ids == current.job_ids or not problem.feasible(ids):
                            continue
                        trial = self._verify_plan(ids, problem, "scorer-refinement")
                        trial_value = query(trial)
                        if trial_value is not None and trial_value > value:
                            options.append((trial_value, trial))
                    if not options:
                        break
                    # Q(A) is common to all extensions, so maximizing Q(U) is
                    # exactly maximizing Delta=Q(U)-Q(A). Only positive deltas
                    # enter options. Original seeds remain in ``seeds`` even if
                    # no positive extension exists (Appendix G.6.2).
                    value, current = min(options, key=lambda p: (-p[0], p[1].uid))
                    refined.setdefault(current.uid, current)
            drafts = self._draft_plans(proposal_state)
            menu = {"ACCEPT": VATSPlan("accept")}
            # Eq. (5): Omega_t = D_t union {VERIFY(A): A in M_t} union
            # {ACCEPT}. Required seeds and a single-edit fallback get reserved
            # slots before optional deeper drafts / refinements consume the cap.
            menu.update(seeds)
            if drafts:
                menu[drafts[0].uid] = drafts[0]
            if len(menu) > cfg.max_menu_options:
                raise ValueError("registered mandatory seeds exceed menu cap")
            for plan in [*drafts[1:], *refined.values()]:
                if len(menu) >= cfg.max_menu_options:
                    break
                menu.setdefault(plan.uid, plan)
            plans = tuple(menu.values())
            expense += cfg.policy_call_cost
            s.budget.settle(preparation_job, expense)
        except CostCapViolation:
            if preparation_job in s.budget.reserved:
                s.budget.settle(preparation_job, expense)
            raise
        except Exception:
            if preparation_job in s.budget.reserved:
                # A callback with unknown expense cannot be silently free.
                s.budget.settle_unknown(preparation_job)
            raise
        s.version += 1
        s.current_transition_index = len(self.transitions) + 1
        events.append({"event": "preparation_paid", "cost": expense, "policy_call_cost": cfg.policy_call_cost,
                       "query_count": queries, "menu": [p.manifest() for p in plans],
                       "scorer_values": values, "paid_budget": s.budget.available})
        if self.verification_backend is not None:
            events.append({"event": "job_catalog", "backend_version": self.verification_backend.version,
                           "jobs": [asdict(job) for job in problem.jobs], "setup_caps": dict(problem.setup_costs),
                           "dependencies": problem.dependencies, "conflicts": problem.conflicts})
        s.history.extend(events)
        # Eq. (6): Q^pi(S_t,omega)=E[F(S_end)-F(S_t)
        #                         -lambda*C_{t:end}-mu*L_{t:end} | S_t,omega].
        # S_t is AFTER this preparation payment: current preparation is sunk,
        # while later preparation/work/failures belong to the suffix target.
        # The contracted policy call is already debited; evaluate the exact state
        # whose likelihood is recorded. Its score branch never selects directly.
        raw = tuple(self.policy_menu(s, plans)) if self.policy_menu else tuple(self.policy(s, p) if p.kind != "accept" else 0. for p in plans)
        if len(raw) != len(plans):
            raise ValueError("policy must return one preference for each canonical plan")
        logits = tuple(0. if p.kind == "accept" else float(v) for p, v in zip(plans, raw))
        selected, probabilities = choose(tuple(p.uid for p in plans), logits, mode=cfg.mode, rng=self.rng,
                                         exploration=cfg.exploration, temperature=cfg.temperature)
        self.decisions.append(TraceDecision("plan", tuple(p.uid for p in plans), selected, probabilities, logits, s.version,
                              len(self.transitions) + 1, cfg.temperature, cfg.exploration if cfg.mode == "collect" else 0.,
                              cfg.menu_snapshot, state_features=s.features(), option_features=tuple(p.features(s) for p in plans),
                              planning_cost_paid=expense, stochastic=cfg.mode == "collect", value_source=self.value_source))
        if self.capture_paid_states:
            self.paid_snapshots.append(self.snapshot_paid_state(plans, problem))
        return menu[selected], problem, events

    def _execute_plan(self, plan: VATSPlan, problem: CompletionProblem) -> list[dict]:
        """Execute the selected bounded descriptor before another work choice.

        Eq. (9), for branch a=1,...,|P| and depth d=1,...,h:
            c_{a,0}=P_a,
            e_{a,d} ~ pi_theta(.|c_{a,d-1},H_{a,d}),
            c_{a,d}=Freeze(Apply(c_{a,d-1},e_{a,d})).
        H_{a,d} contains observations available before that edit; future evaluator
        outcomes are absent. Each successor is generated from an actual immutable
        parent, not from a predicted future artifact. An invalid edit is paid, stops
        that path, and leaves its valid prefix available to future plans.

        DRAFT modifies the version tree:
            C^+=C union Delta_C, A^+=A union Delta_A.
        VERIFY requests Jobs(omega), preserves that tree, and updates only evidence:
            T^+=T, E^+=E union {e in Delta_E:Valid_Gamma(e)=1}.
        Both actions update expense, attempts and observed history. Valid_Gamma checks
        identity, scope, fields and provenance, including records of hard failure.
        Only later ACCEPT may use the complete admissible candidates:
            Eq. (10): ACCEPT(S)=TieArgMax_{c in C_full(S)} J_Gamma(c).
        """
        s, cfg = self.state, self.config
        if plan.work_cap > s.budget.available or plan.attempt_cap > s.remaining_attempts:
            raise ValueError("selected work is no longer feasible")
        if plan.kind == "accept":
            return [{"event": "accept"}]
        events = []
        s.current_transition_index = len(self.transitions)
        if plan.kind == "draft":
            # Eq. (9): e_{a,d} ~ pi_theta(.|c_{a,d-1},H_{a,d});
            # c_{a,d}=Freeze(Apply(c_{a,d-1},e_{a,d})). The next path step
            # uses an actually executed immutable parent, even if its required
            # checks are incomplete. A failed edit consumes one attempt and its
            # expense; only that path stops, and its valid prefix survives.
            if (plan.depth > cfg.max_draft_depth or len(plan.parent_ids) > cfg.max_draft_parents
                    or plan.work_cap != len(plan.parent_ids) * plan.depth * cfg.draft_cap
                    or plan.attempt_cap != len(plan.parent_ids) * plan.depth
                    or plan.attempt_cap > cfg.max_candidates - len(s.candidates)
                    or any(p not in s.candidates or s.children_drafted.get(p, 0) >= cfg.max_children for p in plan.parent_ids)):
                raise ValueError("draft descriptor no longer matches its physical work bounds")
            for parent_id in plan.parent_ids:
                current = parent_id
                for _ in range(plan.depth):
                    s.remaining_attempts -= 1
                    first_new_decision = len(self.decisions)
                    subevents = self._draft(Action("draft", current, cfg.draft_cap))
                    for i in range(first_new_decision, len(self.decisions)):
                        # Deployment makes the work choice greedy; Eq. (9)
                        # still allows sampled edits. Preserve the actual law.
                        self.decisions[i] = replace(self.decisions[i], transition_index=len(self.transitions),
                                                    state_version=s.version)
                    events.extend(subevents)
                    frozen = [event["candidate_id"] for event in subevents if event["event"] == "freeze"]
                    if not frozen:
                        break  # failed edit retains its existing valid prefix
                    current = frozen[-1]
            return events
        if (not problem.feasible(plan.job_ids) or any(job != problem.by_id[job.uid] for job in plan.jobs)
                or plan.work_cap != problem.cost(plan.job_ids) or plan.attempt_cap != problem.attempt_cost(plan.job_ids)):
            raise ValueError("verification descriptor violates its registered dependency/conflict/resource contract")
        if cfg.workers > 1:
            return self._execute_parallel(plan, problem)
        if self.verification_backend is not None:
            return self._execute_compound(plan, problem)
        # Reserve every requested operation plus shared setup atomically. These
        # commitments cannot be consumed by another request in the same block.
        if any(len(job.keys) != 1 for job in plan.jobs):
            raise ValueError("default V ATS executor supports single-key jobs; compound jobs require an execution adapter")
        jobs = {j.uid: self._job("vats-request") for j in plan.jobs}
        groups = {j.group for j in plan.jobs}
        setup_jobs = {g: self._job("vats-setup") for g in groups}
        reservations = {jobs[j.uid]: (j.cap, "acquisition") for j in plan.jobs}
        reservations.update({setup_jobs[g]: (problem.setup_costs[g], "shared-setup") for g in groups})
        s.budget.reserve_many(reservations)
        try:
            for group in sorted(groups):
                s.budget.settle(setup_jobs[group], problem.setup_costs[group])
                events.append({"event": "shared_setup", "group": group, "cost": problem.setup_costs[group]})
            for job_id in problem.order(plan.job_ids):
                job = problem.by_id[job_id]
                candidate_id, task_id = job.keys[0]
                candidate = s.candidates[candidate_id]
                task = next(t for t in s.scope.tasks if t.uid == task_id)
                # Debit the enforceable attempt cap before the physical call.
                # A returned receipt lets us refund unused attempts; an exception
                # with no receipt cannot make failed/unfinished work free. This
                # matches the parallel execution path and Appendix F.4.
                s.remaining_attempts -= job.attempts
                result = self.acquirer.acquire(candidate, task, s.scope, self._sources(candidate, task), cfg.caps(task),
                                               reservation=jobs[job.uid], resume_token=self._resume_tokens.get((candidate_id, task_id)))
                actual_attempts = len(result.checks) + int(
                    not result.from_cache and (result.record is None or result.record.route == "U"))
                s.remaining_attempts += job.attempts - actual_attempts
                if actual_attempts > job.attempts:
                    s.budget.breached = True
                    raise CostCapViolation("verification job exceeded its primitive-attempt cap")
                if result.resume_token:
                    self._resume_tokens[candidate_id, task_id] = result.resume_token
                else:
                    self._resume_tokens.pop((candidate_id, task_id), None)
                events.append({"event": "acquisition", "candidate_id": candidate_id, "task_id": task_id,
                               "status": result.status, "route": result.record.route if result.record else None,
                               "cost": result.actual_cost, "from_cache": result.from_cache,
                               "resume_token": result.resume_token, "error": result.error,
                               "checks": [{"status": c.status, "actual_cost": c.actual_cost, "reason": c.reason,
                                           "source_id": c.certificate.source_id if c.certificate else None,
                                           "checker_id": s.ledger.checker.version} for c in result.checks]})
        finally:
            for job in reservations:
                if job in s.budget.reserved:
                    s.budget.release(job)
        return events

    def _execute_parallel(self, plan: VATSPlan, problem: CompletionProblem) -> list[dict]:
        """Synchronous dependency waves with coordinator-only host mutations.

        An adapter must explicitly guarantee concurrent calls isolate relevant
        execution state. A Python copy is only a host-ledger snapshot, never a
        substitute for that execution contract. Every launched call is joined
        and charged before propagating an error; only unstarted work is released.
        """
        s, cfg, backend = self.state, self.config, self.verification_backend
        adapter = backend if backend is not None else self.executor
        if getattr(adapter, "parallel_safe", False) is not True:
            raise ValueError("workers > 1 requires an adapter with parallel_safe=True and isolated concurrent execution")
        if backend is None and any(job.kind != "result" or len(job.keys) != 1 for job in plan.jobs):
            raise ValueError("default V ATS executor supports single-key result jobs; compound jobs require an execution adapter")
        jobs = {job.uid: self._job("parallel-request") for job in plan.jobs}
        groups = {job.group for job in plan.jobs}
        setup_jobs = {group: self._job("parallel-setup") for group in groups}
        reservations = {jobs[job.uid]: (job.cap, "compound-acquisition" if backend else "acquisition") for job in plan.jobs}
        reservations.update({setup_jobs[group]: (problem.setup_costs[group], "shared-setup") for group in groups})
        s.budget.reserve_many(reservations)
        started = time.monotonic()
        events, intervals, setup_intervals = [], [], []
        setups: dict[str, SetupExecutionResult] = {}
        results = {}
        statuses = {}
        task_by_id = {task.uid: task for task in s.scope.tasks}

        def invoke(call):
            begin = time.monotonic() - started
            try:
                value, error = call(), None
            except BaseException as exc:
                value, error = None, exc
            return value, error, begin, time.monotonic() - started

        def timing(begin, end):
            return {"started_seconds": begin, "completed_seconds": end, "worker_seconds": end - begin}

        try:
            # Setup remains serial and runs once per actual group. Artifacts are
            # immutable; a parallel-safe backend must safely share their use.
            for group in sorted(groups):
                reservation = setup_jobs[group]
                if backend is None:
                    s.budget.settle(reservation, problem.setup_costs[group])
                    setups[group] = SetupExecutionResult("complete", problem.setup_costs[group])
                    events.append({"event": "shared_setup", "group": group, "cost": problem.setup_costs[group]})
                    continue
                context = self._job_context(s)
                result, error, begin, end = invoke(lambda: backend.setup(group, context, cap=problem.setup_costs[group]))
                setup_intervals.append((begin, end))
                if error is not None or not isinstance(result, SetupExecutionResult):
                    s.budget.settle_unknown(reservation)
                    raise error if error is not None else TypeError("setup must return a metered SetupExecutionResult")
                s.budget.settle(reservation, result.actual_cost)
                if result.status == "complete" and (result.receipt.get("group") != group
                        or result.receipt.get("execution_id") != s.scope.execution_id
                        or not isinstance(result.receipt.get("run_id"), str) or not result.receipt["run_id"]):
                    raise ValueError("shared setup requires a matching execution receipt")
                setups[group] = result
                events.append({"event": "shared_setup", "group": group, "status": result.status,
                               "cost": result.actual_cost, "receipt": dict(result.receipt), "error": result.error,
                               **timing(begin, end)})

            dependencies = {uid: {b for a, b in problem.dependencies if a == uid} for uid in plan.job_ids}
            pending = list(problem.order(plan.job_ids))
            with ThreadPoolExecutor(max_workers=min(cfg.workers, len(pending)), thread_name_prefix="vats-verify") as pool:
                while pending:
                    # Propagate failed prerequisites without spending attempts or
                    # claiming a cancellation of work that never started.
                    for uid in tuple(pending):
                        job = problem.by_id[uid]
                        if (setups[job.group].status != "complete"
                                or any(dep in statuses and statuses[dep] != "complete" for dep in dependencies[uid])):
                            s.budget.release(jobs[uid])
                            statuses[uid] = "blocked"
                            pending.remove(uid)
                            events.append({"event": "blocked_job", "job_id": uid, "status": "not_started",
                                           "reason": "setup or prerequisite did not complete", "cost": 0, "attempts": 0})
                    if not pending:
                        break
                    wave, covered = [], set()
                    for uid in pending:
                        job = problem.by_id[uid]
                        if (len(wave) < cfg.workers and all(statuses.get(dep) == "complete" for dep in dependencies[uid])
                                and not covered.intersection(job.keys)):
                            wave.append(uid)
                            covered.update(job.keys)
                    if not wave:
                        raise RuntimeError("verification dependency graph has no runnable jobs")
                    context = self._job_context(s)
                    local_acquirers, calls, futures, completed_calls = {}, {}, {}, {}
                    # Construct each request before launch. Worker threads never
                    # receive the host's mutable ledger, budget or search state.
                    for uid in wave:
                        job = problem.by_id[uid]
                        if backend is not None:
                            prerequisites = MappingProxyType({dep: results[dep] for dep in dependencies[uid]})
                            token = self._job_resume_tokens.get(uid)
                            call = lambda job=job, prerequisites=prerequisites, token=token: backend.run(
                                job, context, setup=setups[job.group].artifacts,
                                prerequisites=prerequisites, resume_token=token)
                        else:
                            candidate_id, task_id = job.keys[0]
                            candidate, task = s.candidates[candidate_id], task_by_id[task_id]
                            local = ResultAcquirer(s.ledger.clone(), BudgetLedger(job.cap), self.executor)
                            local.budget.reserve(jobs[uid], job.cap, "acquisition")
                            local_acquirers[uid] = local
                            token = self._resume_tokens.get((candidate_id, task_id))
                            call = lambda local=local, candidate=candidate, task=task, token=token, uid=uid: local.acquire(
                                candidate, task, s.scope, (), cfg.caps(task), reservation=jobs[uid], resume_token=token)
                        calls[uid] = call
                    failures = []

                    def run_request(uid):
                        # Keep a receipt even if the pool enqueues a request but
                        # fails before returning its Future to the coordinator.
                        completed_calls[uid] = invoke(calls[uid])

                    for uid in wave:
                        try:
                            futures[uid] = pool.submit(run_request, uid)
                        except BaseException as exc:
                            failures.append(exc)
                            break
                    for future in futures.values():
                        while True:
                            try:
                                future.result()
                                break
                            except BaseException as exc:
                                failures.append(exc)
                                if future.done():
                                    break
                    if failures:
                        # Also drain requests whose Future was lost on dispatch.
                        # Interruptions propagate only after launched work is paid.
                        while True:
                            try:
                                pool.shutdown(wait=True)
                                break
                            except BaseException as exc:
                                failures.append(exc)
                    # Complete the entire wave before feedback changes host state.
                    wave = [uid for uid in wave if uid in completed_calls]
                    finished = dict(completed_calls)
                    for uid in wave:
                        job, reservation = problem.by_id[uid], jobs[uid]
                        result, error, begin, end = finished[uid]
                        intervals.append((begin, end))
                        if error is not None:
                            failures.append(error)
                        try:
                            if backend is None:
                                local = local_acquirers[uid]
                                # The ordinary acquirer already meters failures,
                                # including overruns and unmetered exceptions.
                                attempts = (job.attempts if result is None else len(result.checks)
                                    + int(not result.from_cache and (result.record is None or result.record.route == "U")))
                                s.remaining_attempts -= attempts
                                if local.budget.charges:
                                    kind = "actual" if local.budget.cost_valid else "conservative_bound"
                                    s.budget.settle(reservation, local.budget.spent, cost_kind=kind)
                                    if local.budget.breached:
                                        s.budget.breached = True
                                else:
                                    s.budget.settle_unknown(reservation)
                                if attempts > job.attempts:
                                    s.budget.breached = True
                                    raise CostCapViolation("verification job exceeded its primitive-attempt cap")
                            elif isinstance(result, JobExecutionResult):
                                s.remaining_attempts -= result.actual_attempts
                                s.budget.settle(reservation, result.actual_cost)
                                if result.actual_attempts > job.attempts:
                                    s.budget.breached = True
                                    raise CostCapViolation("compound job exceeded its primitive-attempt cap")
                            else:
                                s.remaining_attempts -= job.attempts
                                s.budget.settle_unknown(reservation)
                                if error is None:
                                    raise TypeError("compound job must return a metered JobExecutionResult")
                        except Exception as exc:
                            failures.append(exc)
                    if failures:
                        raise failures[0]

                    # Validate the entire wave against a private admission ledger
                    # before any host evidence is changed. Partial raw fields are
                    # never outcomes; interrupted compound calls may still carry
                    # separately complete, receipted outcomes.
                    staged = s.ledger.clone()
                    admissions, realized_by_job = [], {}
                    for uid in wave:
                        job, result = problem.by_id[uid], finished[uid][0]
                        if backend is not None:
                            for outcome in result.outcomes:
                                outcome.validate(job, context)
                            if job.kind == "prerequisite" and result.outcomes:
                                raise ValueError("operational prerequisites cannot supply task evidence")
                            outcomes = [(o.candidate_id, o.task_id, o.outcome, o.receipt) for o in result.outcomes]
                        else:
                            record = result.record
                            outcomes = [] if record is None else [(record.candidate_id, record.task_id, record.outcome, record.receipt)]
                        realized = set()
                        for candidate_id, task_id, outcome, receipt in outcomes:
                            candidate, task = s.candidates[candidate_id], task_by_id[task_id]
                            if staged.get(candidate, task, s.scope) is None:
                                realized.add((candidate_id, task_id))
                            staged.admit_direct(candidate, task, s.scope, outcome, receipt)
                            admissions.append((candidate, task, outcome, receipt))
                        realized_by_job[uid] = sorted(realized)
                    for candidate, task, outcome, receipt in admissions:
                        s.ledger.admit_direct(candidate, task, s.scope, outcome, receipt)
                    for uid in wave:
                        job = problem.by_id[uid]
                        result, _, begin, end = finished[uid]
                        statuses[uid], results[uid] = result.status, result
                        pending.remove(uid)
                        if backend is not None:
                            if result.resume_token:
                                self._job_resume_tokens[uid] = result.resume_token
                            else:
                                self._job_resume_tokens.pop(uid, None)
                            events.append({"event": "compound_acquisition", "job_id": uid, "status": result.status,
                                           "requested_keys": job.keys, "realized_keys": realized_by_job[uid], "route": "U",
                                           "cost": result.actual_cost, "attempts": result.actual_attempts,
                                           "artifacts": dict(result.artifacts), "resume_token": result.resume_token,
                                           "error": result.error, **timing(begin, end)})
                        else:
                            candidate_id, task_id = job.keys[0]
                            if result.resume_token:
                                self._resume_tokens[candidate_id, task_id] = result.resume_token
                            else:
                                self._resume_tokens.pop((candidate_id, task_id), None)
                            events.append({"event": "acquisition", "job_id": uid, "candidate_id": candidate_id, "task_id": task_id,
                                           "status": result.status, "route": result.record.route if result.record else None,
                                           "cost": result.actual_cost, "from_cache": result.from_cache,
                                           "resume_token": result.resume_token, "error": result.error,
                                           "checks": [], **timing(begin, end)})
            active = peak = 0
            for _, delta in sorted([(begin, 1) for begin, _ in intervals] + [(end, -1) for _, end in intervals]):
                active += delta
                peak = max(peak, active)
            events.append({"event": "verification_schedule", "scheduler": "synchronous_dependency_waves",
                           "workers": cfg.workers, "peak_active_workers": peak,
                           "elapsed_seconds": time.monotonic() - started,
                           "worker_seconds": math.fsum(end - begin for begin, end in intervals + setup_intervals),
                           "worker_time_unit": "host-call-seconds", "work_cap": plan.work_cap, "attempt_cap": plan.attempt_cap})
        finally:
            for reservation in reservations:
                if reservation in s.budget.reserved:
                    s.budget.release(reservation)
        return events

    def _execute_compound(self, plan: VATSPlan, problem: CompletionProblem) -> list[dict]:
        """Execute the declared graph; potential coverage never updates evidence."""
        s, backend = self.state, self.verification_backend
        jobs = {job.uid: self._job("compound-request") for job in plan.jobs}
        groups = {job.group for job in plan.jobs}
        setup_jobs = {group: self._job("compound-setup") for group in groups}
        reservations = {jobs[job.uid]: (job.cap, "compound-acquisition") for job in plan.jobs}
        reservations.update({setup_jobs[group]: (problem.setup_costs[group], "shared-setup") for group in groups})
        s.budget.reserve_many(reservations)
        events = []
        setups: dict[str, SetupExecutionResult] = {}
        results: dict[str, JobExecutionResult] = {}
        try:
            for group in sorted(groups):
                reservation = setup_jobs[group]
                try:
                    result = backend.setup(group, self._job_context(s), cap=problem.setup_costs[group])
                    if not isinstance(result, SetupExecutionResult):
                        raise TypeError("setup must return a metered SetupExecutionResult")
                    s.budget.settle(reservation, result.actual_cost)
                    if result.status == "complete":
                        if (result.receipt.get("group") != group
                                or result.receipt.get("execution_id") != s.scope.execution_id
                                or not isinstance(result.receipt.get("run_id"), str) or not result.receipt["run_id"]):
                            raise ValueError("shared setup requires a matching execution receipt")
                    setups[group] = result
                    events.append({"event": "shared_setup", "group": group, "status": result.status,
                                   "cost": result.actual_cost, "receipt": dict(result.receipt), "error": result.error})
                except Exception:
                    if reservation in s.budget.reserved:
                        s.budget.settle_unknown(reservation)
                    raise
            for job_id in problem.order(plan.job_ids):
                job, reservation = problem.by_id[job_id], jobs[job_id]
                dependencies = {b for a, b in problem.dependencies if a == job_id}
                if (setups[job.group].status != "complete"
                        or any(dep not in results or results[dep].status != "complete" for dep in dependencies)):
                    s.budget.release(reservation)
                    events.append({"event": "blocked_job", "job_id": job_id, "status": "not_started",
                                   "reason": "setup or prerequisite did not complete", "cost": 0, "attempts": 0})
                    continue
                try:
                    context = self._job_context(s)
                    # Appendix F.4: failed/unfinished physical attempts remain
                    # charged. Reserve attempts before invoking the backend;
                    # only a typed metered result can refund its unused portion.
                    s.remaining_attempts -= job.attempts
                    result = backend.run(job, context, setup=setups[job.group].artifacts,
                                         prerequisites=MappingProxyType({dep: results[dep] for dep in dependencies}),
                                         resume_token=self._job_resume_tokens.get(job_id))
                    if not isinstance(result, JobExecutionResult):
                        raise TypeError("compound job must return a metered JobExecutionResult")
                    # Settle the actual work even when it reveals a broken cap or
                    # invalid evidence; invalid results never become free calls.
                    s.remaining_attempts += job.attempts - result.actual_attempts
                    s.budget.settle(reservation, result.actual_cost)
                    if result.actual_attempts > job.attempts:
                        s.budget.breached = True
                        raise CostCapViolation("compound job exceeded its primitive-attempt cap")
                    for outcome in result.outcomes:
                        outcome.validate(job, context)
                    if job.kind == "prerequisite" and result.outcomes:
                        raise ValueError("operational prerequisites cannot supply task evidence")
                    realized = set()
                    for outcome in result.outcomes:
                        candidate = s.candidates[outcome.candidate_id]
                        task = next(t for t in s.scope.tasks if t.uid == outcome.task_id)
                        existing = s.ledger.get(candidate, task, s.scope)
                        s.ledger.admit_direct(candidate, task, s.scope, outcome.outcome, outcome.receipt)
                        if existing is None:
                            realized.add((candidate.uid, task.uid))
                    results[job_id] = result
                    if result.resume_token:
                        self._job_resume_tokens[job_id] = result.resume_token
                    else:
                        self._job_resume_tokens.pop(job_id, None)
                    events.append({"event": "compound_acquisition", "job_id": job_id, "status": result.status,
                                   "requested_keys": job.keys, "realized_keys": sorted(realized), "route": "U",
                                   "cost": result.actual_cost, "attempts": result.actual_attempts,
                                   "artifacts": dict(result.artifacts), "resume_token": result.resume_token,
                                   "error": result.error})
                except Exception:
                    if reservation in s.budget.reserved:
                        s.budget.settle_unknown(reservation)
                    raise
        finally:
            for reservation in reservations:
                if reservation in s.budget.reserved:
                    s.budget.release(reservation)
        return events

    def run(self) -> SearchResult:
        """Run one frozen search with a disjoint preparation/work clock.

        Eq. (16), at visited choice t:
            B_t=B_t^- - o_t,
            B_{t+1}^-=B_t-kappa_t,
            N_{t+1}=N_t-n_t,
            r_t^m=-lambda*o_t-mu*l_t^m,
            r_t^w=F(S_{t+1}^-)-F(S_t)-lambda*kappa_t-mu*l_t^w.
        The superscripts m and w identify menu preparation and physical work; they are
        not powers. o_t and kappa_t use the declared expense unit; n_t counts charged
        primitive attempts; l_t^m and l_t^w are measured elapsed seconds. Preparation
        acquires no outcomes, so F(S_t)=F(S_t^-). ACCEPT sets S_{t+1}^-=S_t and
        kappa_t=l_t^w=n_t=r_t^w=0. T is the number of visited paid choices, including
        a sampled ACCEPT. If no preparation fits, S_end=S_T^-; T=0 is allowed.

        Eq. (17), the selected plan's undiscounted suffix begins after its preparation:
            G_t=r_t^w + sum_{u=t+1}^{T-1}(r_u^m+r_u^w),
            C_{t:end}=kappa_t + sum_{u=t+1}^{T-1}(o_u+kappa_u),
            L_{t:end}=l_t^w + sum_{u=t+1}^{T-1}(l_u^m+l_u^w).
        Current preparation is sunk and excluded from G_t; every later preparation,
        including a terminal ACCEPT choice's preparation, is included. By contrast,
        whole-search totals C_g=sum_{t=0}^{T_g-1}(o_gt+kappa_gt) and
        L_g=sum_{t=0}^{T_g-1}(l_gt^m+l_gt^w) include the initial preparation. Worker
        durations from parallel checks are not added into the single-clock L_g.
        """
        if self._used:
            raise ValueError("VATSEngine instances are single-use")
        self._used = True
        s, cfg = self.state, self.config
        started = time.monotonic()
        initial_spent, initial_charges, initial_f = s.budget.spent, len(s.budget.charges), self._f()
        records = []
        for key in sorted(s.ledger.records):
            record = s.ledger.records[key].manifest()
            record.pop("signature")
            records.append(record)
        initial_context = digest({"incumbent": s.incumbent.uid, "scope": s.scope.uid, "initial_records": records,
                                  "budget": s.budget.total, "spent": s.budget.spent, "pending": s.budget.pending,
                                  "attempts": s.remaining_attempts, "workers": s.workers, "checker_version": s.ledger.checker.version,
                                  "history": s.history, "candidates": sorted(s.candidates), "algorithm": "vats-pdf-v1",
                                  "verification_backend": self.verification_backend.version if self.verification_backend else None})
        reason = "action_limit"
        remaining_rounds = cfg.max_actions
        if self._forced_start is not None:
            plan, problem = self._forced_start
            phase_started = time.monotonic()
            before, f_before, version = s.budget.spent, self._f(), s.version
            events = self._execute_plan(plan, problem)
            s.version += 1
            s.history.extend(events)
            self.transitions.append(TraceTransition(0, plan.kind, plan.candidate_id, version, s.version,
                                    s.budget.spent - before, f_before, self._f(), tuple(events), "forced-work",
                                    0. if plan.kind == "accept" else time.monotonic() - phase_started))
            self._rounds_executed += 1
            remaining_rounds -= 1
            if plan.kind == "accept":
                reason, remaining_rounds = "forced_accept", 0
        for _ in range(remaining_rounds):
            if s.remaining_attempts <= 0:
                reason = "attempt_limit"
                break
            if cfg.prep_cap > s.budget.available:
                reason = "no_affordable_preparation"
                break
            phase_started = time.monotonic()
            before, f_before, version = s.budget.spent, self._f(), s.version
            plan, problem, prep_events = self._prepare()
            # Eq. (16), preparation transition S_t^- -> S_t:
            # B_t=B_t^- - o_t; r_t^m=-lambda*o_t-mu*l_t^m.
            # _prepare has already settled o_t once. Record its spending delta
            # and wall-clock interval, never debit it again or acquire evidence.
            self.transitions.append(TraceTransition(len(self.transitions), "prepare", "", version, s.version,
                                    s.budget.spent - before, f_before, self._f(), tuple(prep_events), "preparation",
                                    time.monotonic() - phase_started))
            phase_started = time.monotonic()
            before, f_before, version = s.budget.spent, self._f(), s.version
            events = self._execute_plan(plan, problem)
            s.version += 1
            s.history.extend(events)
            # Eq. (16), work transition S_t -> S_{t+1}^-:
            # B_{t+1}^-=B_t-kappa_t; N_{t+1}=N_t-n_t;
            # r_t^w=F(S_{t+1}^-)-F(S_t)-lambda*kappa_t-mu*l_t^w.
            # Eq. (17) suffix G_t starts HERE: r_t^w +
            # sum_{u>t}(r_u^m+r_u^w), excluding the current sunk preparation.
            # ACCEPT contributes zero work cost/time/attempts/reward. Parallel
            # worker durations are diagnostic only; l_t^w is one elapsed clock.
            self.transitions.append(TraceTransition(len(self.transitions), plan.kind, plan.candidate_id, version, s.version,
                                    s.budget.spent - before, f_before, self._f(), tuple(events), "work",
                                    0. if plan.kind == "accept" else time.monotonic() - phase_started))
            self._rounds_executed += 1
            if not s.budget.invariant() or s.remaining_attempts < 0:
                raise CostCapViolation("V ATS resource invariant lost")
            if plan.kind == "accept":
                reason = "policy_accept"
                break
        outcome = accept(s.incumbent, s.candidates.values(), s.ledger, s.scope, 0., self.hard_constraint)
        if not s.budget.invariant() or s.remaining_attempts < 0:
            raise CostCapViolation("V ATS resource invariant lost at acceptance")
        total = s.budget.spent - initial_spent
        # Eq. (13), Eq. (16), Appendix G.1.2: disjoint transition charges give
        # C_g=sum_t(o_t+kappa_t), so the undiscounted reward telescopes to
        # R_g=F(S_end)-F(S_0^-)-lambda*C_g-mu*L_g. Failed work remains inside
        # these sums, even when it yields no complete admissible improvement.
        if total != sum(t.cost for t in self.transitions):
            raise RuntimeError("V ATS transition accounting omitted spending")
        return SearchResult(outcome, initial_context, self.search_id, cfg.seed, cfg.group_id, initial_f, self._f(), total,
                            cfg.cost_unit, tuple(self.transitions), tuple(self.decisions),
                            tuple(asdict(c) for c in s.budget.charges[initial_charges:]), reason, tuple(s.candidates),
                            s.budget.available, s.ledger.to_json(), self.value_source,
                            time.monotonic() - started, "vats-pdf-v1")
