"""Cost-minimizing completion covers, Eqs. (14)--(15).

Variable mapping: job.uid is j, job.keys is K_j, job.cap is c_bar_j,
job.attempts is n_bar_j, job.group is g(j), and setup_costs[g] is s_g.
``target`` contains an entire candidate's K_c or a union of complete K_c sets.
Dropping expensive missing items would change the required completion target.
``budget`` is B_plan=B^- - o_bar after reserving the enforceable preparation
cap o_bar from pre-menu remaining expense B^-. Only B_plan>=0 is considered.
c_bar_j and s_g use the same integer expense unit as B_plan; n_bar_j and N
count primitive attempts. Group g here indexes shared setups, independently of
the independent-search group index g used by the learning objective.

Eq. (14): min_{x,y} sum_j c_bar_j*x_j + sum_g s_g*y_g, subject to
  sum_{j:k in K_j} x_j >= 1                         for every required k;
  x_j <= x_i                                      for every prerequisite i;
  x_j + x_j' <= 1                                 for every conflict (j,j');
  x_j <= y_g(j), y_g <= sum_{j:g(j)=g} x_j;
  sum_j c_bar_j*x_j + sum_g s_g*y_g <= B_plan;
  sum_j n_bar_j*x_j <= N; x_j,y_g in {0,1}.
Eq. (15) additionally describes the independently checked returned seed:
  A_K in {A in X_Gamma(S_tilde): K subset K(A), D(A)=A,
                                C(A)<=B_plan, n(A)<=N},
where X_Gamma contains nonempty sets legal under fixed dependency, conflict and
resource rules, and
  K(A)=union_{j in A} K_j,
  D(A)=union_{j in A} D_j,
  n(A)=sum_{j in A} n_bar_j,
  C(A)=sum_{j in A} c_bar_j + sum_{g in {g(j):j in A}} s_g.
The second cost sum counts each activated setup once, even if its selected jobs
cover several candidates. Prerequisite closure D(A)=A ensures no required
operational job is omitted. Feasibility describes what can be attempted; it
does not assert that any acquired outcome will pass or even return complete.

Appendix G.1.3 supplies a further condition on X_Gamma: every legal nonempty
plan consumes at least one registered primitive attempt. Thus a returned plan
also satisfies n(A)>=1. This is a termination condition, not an extra term
printed in Eq. (14): individual n_bar_j may still be zero, for example for an
operational prerequisite paired with a positive-attempt contracted check.
Both solver backends and the independent Eq. (15) check enforce the condition;
otherwise a zero-attempt diagnostic could be offered indefinitely without
decreasing the remaining primitive-attempt allowance N.

These jobs request outcomes. A feasible cover is never evidence that they ran,
passed, or even returned complete fields. Unlike the legacy additive proposer,
keys can span several immutable candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from types import MappingProxyType
from typing import Iterable

from specrsi.core.budget import cost_int
from specrsi.core.models import FrozenMap


@dataclass(frozen=True)
class VerificationJob:
    uid: str
    keys: tuple[tuple[str, str], ...]
    cap: int
    attempts: int = 1
    group: str = "execution"
    kind: str = "result"

    def __post_init__(self):
        object.__setattr__(self, "keys", tuple(tuple(k) for k in self.keys))
        cost_int(self.cap)
        cost_int(self.attempts)
        if self.kind not in {"result", "prerequisite"}:
            raise ValueError("unknown verification job kind")
        if not self.uid or not self.group or any(len(k) != 2 or not all(isinstance(v, str) and v for v in k) for k in self.keys):
            raise ValueError("job requires immutable candidate-item keys")
        if self.kind == "result" and not self.keys:
            raise ValueError("result jobs require possible coverage")
        if self.kind == "prerequisite" and self.keys:
            # Appendix F.4: setup / prerequisite artifacts enable a real check;
            # they are not contracted task outcomes and cannot cover K_c.
            raise ValueError("operational prerequisites cannot request task evidence coverage")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("duplicate requested key")


@dataclass(frozen=True)
class CompletionProblem:
    jobs: tuple[VerificationJob, ...]
    setup_costs: FrozenMap
    budget: int
    attempts: int
    dependencies: tuple[tuple[str, str], ...] = ()
    conflicts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "jobs", tuple(self.jobs))
        object.__setattr__(self, "setup_costs", FrozenMap(self.setup_costs))
        object.__setattr__(self, "dependencies", tuple(tuple(x) for x in self.dependencies))
        object.__setattr__(self, "conflicts", tuple(tuple(x) for x in self.conflicts))
        cost_int(self.budget)
        cost_int(self.attempts)
        for value in self.setup_costs.values():
            cost_int(value)
        ids = {j.uid for j in self.jobs}
        if len(ids) != len(self.jobs) or any(j.group not in self.setup_costs for j in self.jobs):
            raise ValueError("jobs must have unique identities and known setup groups")
        if any(a == b or a not in ids or b not in ids for a, b in self.dependencies + self.conflicts):
            raise ValueError("invalid job constraint")
        self.order(ids)

    @cached_property
    def by_id(self):
        # CompletionProblem and VerificationJob are immutable. Reuse one read-only
        # index through feasibility/refinement, instead of rebuilding all n jobs
        # for each of the n terms in Eq. (14)'s cost/attempt sums.
        return MappingProxyType({j.uid: j for j in self.jobs})

    def closure(self, selected: Iterable[str]) -> tuple[str, ...]:
        ids = set(selected)
        if not ids.issubset(self.by_id):
            raise ValueError("unknown job")
        while True:
            expanded = ids | {b for a, b in self.dependencies if a in ids}
            if expanded == ids:
                return tuple(sorted(ids))
            ids = expanded

    def order(self, selected: Iterable[str]) -> tuple[str, ...]:
        ids = set(selected)
        if not ids.issubset(self.by_id):
            raise ValueError("unknown job")
        pending = {i: {b for a, b in self.dependencies if a == i and b in ids} for i in ids}
        result = []
        while pending:
            enabled = sorted(i for i, deps in pending.items() if not deps)
            if not enabled:
                raise ValueError("cyclic job dependencies")
            result.extend(enabled)
            for i in enabled:
                del pending[i]
            for deps in pending.values():
                deps.difference_update(enabled)
        return tuple(result)

    def cost(self, selected: Iterable[str]) -> int:
        # Appendix F.3: C(A)=sum_{j in A} c_bar_j +
        # sum_{g in {g(j):j in A}} s_g. Sets are intentional: a shared setup
        # is paid once per selected block, including across candidate branches.
        jobs = [self.by_id[i] for i in set(selected)]
        return sum(j.cap for j in jobs) + sum(self.setup_costs[g] for g in {j.group for j in jobs})

    def attempt_cost(self, selected: Iterable[str]) -> int:
        return sum(self.by_id[i].attempts for i in set(selected))

    def coverage(self, selected: Iterable[str]) -> set[tuple[str, str]]:
        return {key for i in selected for key in self.by_id[i].keys}

    def feasible(self, selected: Iterable[str], target: Iterable[tuple[str, str]] = ()) -> bool:
        # Eq. (15) is checked again outside the optimizer: closure, conflicts,
        # coverage, expense and attempts must all hold even for a timeout seed.
        # G.1.3 requires a positive total attempt charge for nonempty work; a
        # zero-attempt prerequisite is legal when its whole plan satisfies this.
        selected = tuple(selected)
        ids = set(selected)
        return (bool(ids) and len(ids) == len(selected) and ids.issubset(self.by_id)
                and self.cost(ids) <= self.budget and 1 <= self.attempt_cost(ids) <= self.attempts
                and all(a not in ids or b in ids for a, b in self.dependencies)
                and all(not {a, b}.issubset(ids) for a, b in self.conflicts)
                and set(target).issubset(self.coverage(ids)))


@dataclass(frozen=True)
class CompletionSolution:
    job_ids: tuple[str, ...]
    cost_cap: int
    optimal: bool
    status: str


def solve_completion(problem: CompletionProblem, target: Iterable[tuple[str, str]], *,
                     backend: str = "scipy", time_limit: float = 1.) -> CompletionSolution | None:
    """Return a checked cover, or no seed; never discard expensive coverage keys.

    Eq. (14) chooses binary job variables x_j and setup variables y_g to minimize
    sum_j c_bar_j*x_j + sum_g s_g*y_g. A candidate target K_c must include every
    still-missing contracted item; a cross-branch target is a union of full K_c
    sets. The output A_K is checked against every Eq. (15) condition, separately
    from the optimizer's termination status.

    The exhaustive backend enumerates finite small cases. SciPy/HiGHS solves the
    sparse integer program. A feasible timeout incumbent remains a usable seed
    after independent integrality, closure, conflict, coverage, setup activation
    and budget/attempt checks; it is not marked optimal. Empty targets need no
    work and return no seed. Infeasible targets likewise produce no seed.
    """
    import math
    target = frozenset(target)
    if backend not in {"scipy", "exhaustive"} or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("invalid completion solver settings")
    if not target or not target.issubset(problem.coverage(problem.by_id)):
        return None
    if backend == "exhaustive":
        if len(problem.jobs) > 22:
            raise ValueError("exhaustive completion solver is limited to 22 jobs")
        best = None
        ids = sorted(problem.by_id)
        for n in range(1, len(ids) + 1):
            for selected in combinations(ids, n):
                if problem.feasible(selected, target):
                    key = (problem.cost(selected), selected)
                    if best is None or key < best:
                        best = key
        return None if best is None else CompletionSolution(best[1], best[0], True, "exhaustive-optimum")

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    jobs = problem.jobs
    n = len(jobs)
    groups = sorted({j.group for j in jobs})
    indices = {j.uid: i for i, j in enumerate(jobs)}
    group_indices = {g: n + i for i, g in enumerate(groups)}
    objective = np.array([j.cap for j in jobs] + [problem.setup_costs[g] for g in groups], dtype=float)
    # Coverage and dependency constraints are sparse. Dense rows scale with
    # jobs times required keys even though a direct check covers just one key.
    # Keep full-scope completion practical without dropping expensive items.
    row_indices, column_indices, entries, upper = [], [], [], []

    def add(coefficients, bound=0.):
        for i, value in coefficients.items():
            if value:
                row_indices.append(len(upper))
                column_indices.append(i)
                entries.append(value)
        upper.append(bound)

    # Eq. (14), last row: the objective coefficients also enforce the hard
    # expense cap. The attempt row excludes setup variables, whose attempt
    # requirements must be represented as explicit prerequisite jobs.
    add(dict(enumerate(objective)), problem.budget)
    add({i: j.attempts for i, j in enumerate(jobs)}, problem.attempts)
    # G.1.3 restricts legal nonempty work to n(A)>=1. Put the condition in
    # the optimizer as well as feasible(): merely rejecting an all-zero cheap
    # incumbent afterward could hide a different, valid positive-attempt cover.
    add({i: -j.attempts for i, j in enumerate(jobs)}, -1.)
    for key in sorted(target):
        # Convert coverage >= 1 to -coverage <= -1 for LinearConstraint.
        add({i: -1. for i, j in enumerate(jobs) if key in j.keys}, -1.)
    for a, b in problem.dependencies:
        # x_a <= x_b; transitive prerequisites follow through these edges.
        add({indices[a]: 1., indices[b]: -1.})
    for a, b in problem.conflicts:
        add({indices[a]: 1., indices[b]: 1.}, 1.)
    for i, job in enumerate(jobs):
        # Both directions are necessary: active jobs imply setup activation,
        # and an active setup must have at least one actually selected job.
        add({i: 1., group_indices[job.group]: -1.})
    for group in groups:
        add({group_indices[group]: 1., **{i: -1. for i, j in enumerate(jobs) if j.group == group}})
    matrix = coo_matrix((entries, (row_indices, column_indices)),
                        shape=(len(upper), len(objective))).tocsc()
    result = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(matrix, -np.inf, np.array(upper)),
                  options={"time_limit": time_limit, "mip_rel_gap": 0.})
    if result.x is None:
        return None
    solution = np.asarray(result.x)
    # Independently validate every binary variable, including setup activation.
    # Merely being close to an integer would also accept values such as -1 or 2.
    if (solution.shape != objective.shape or not np.all(np.isfinite(solution))
            or np.any(solution < -1e-6) or np.any(solution > 1. + 1e-6)
            or np.max(abs(solution - np.rint(solution))) > 1e-6):
        return None
    selected = tuple(sorted(j.uid for i, j in enumerate(jobs) if solution[i] > .5))
    if not problem.feasible(selected, target):
        return None
    activated = {g for g, i in group_indices.items() if solution[i] > .5}
    if activated != {problem.by_id[i].group for i in selected}:
        return None
    return CompletionSolution(selected, problem.cost(selected), result.status == 0, str(result.message))
