"""Feasible additive proposals plus completion-aware whole-batch menu hooks."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Callable, Iterable, Mapping, Sequence

from specrsi.core.budget import cost_int
from specrsi.core.models import FrozenMap, digest


@dataclass(frozen=True)
class CheckOption:
    uid: str
    candidate_id: str
    task_ids: tuple[str, ...]
    cap: int
    group: str = "default"
    predicted_value: float = 0.
    kind: str = "acquire"
    source_ids: tuple[str, ...] = ()
    reserved_cap: int = 0
    reservation_id: str | None = None

    def __post_init__(self) -> None:
        cost_int(self.cap)
        cost_int(self.reserved_cap)
        if self.reserved_cap > self.cap:
            raise ValueError("transferred reservation exceeds operation cap")
        if self.reserved_cap and not self.reservation_id:
            raise ValueError("transferred cap must identify the exact existing physical-work reservation")
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if not self.uid or not self.candidate_id or not self.group or not math.isfinite(self.predicted_value):
            raise ValueError("invalid check option")
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("possible coverage must be nonempty and duplicate-free")
        if self.kind not in ("acquire", "direct", "prerequisite"):
            raise ValueError("unknown operation kind")


@dataclass(frozen=True)
class BatchProblem:
    options: tuple[CheckOption, ...]
    setup_costs: FrozenMap
    budget: int
    dependencies: tuple[tuple[str, str], ...] = ()  # operation -> prerequisite
    conflicts: tuple[tuple[str, str], ...] = ()
    state_version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "setup_costs", FrozenMap(self.setup_costs))
        object.__setattr__(self, "dependencies", tuple(tuple(x) for x in self.dependencies))
        object.__setattr__(self, "conflicts", tuple(tuple(x) for x in self.conflicts))
        cost_int(self.budget)
        ids = {o.uid for o in self.options}
        if len(ids) != len(self.options) or len({o.candidate_id for o in self.options}) > 1:
            raise ValueError("a batch requires unique options for one fixed candidate")
        for v in self.setup_costs.values():
            cost_int(v)
        if any(o.group not in self.setup_costs for o in self.options):
            raise ValueError("missing setup cap")
        for a, b in self.dependencies + self.conflicts:
            if a not in ids or b not in ids or a == b:
                raise ValueError("invalid constraint endpoints")
        self.dependency_order(tuple(ids))  # executable dependencies must form a DAG

    @property
    def by_id(self) -> dict[str, CheckOption]:
        return {o.uid: o for o in self.options}

    def cost(self, selected: Sequence[str]) -> int:
        options = [self.by_id[x] for x in selected]
        return sum(o.cap - o.reserved_cap for o in options) + sum(self.setup_costs[g] for g in {o.group for o in options})

    def feasible(self, selected: Sequence[str], *, budget: int | None = None) -> bool:
        ids = set(selected)
        if len(ids) != len(selected) or not ids.issubset(self.by_id):
            return False
        if self.cost(selected) > (self.budget if budget is None else budget):
            return False
        reserved = [self.by_id[i].reservation_id for i in selected if self.by_id[i].reserved_cap]
        if len(reserved) != len(set(reserved)):
            return False
        return (all(not (a in ids and b in ids) for a, b in self.conflicts)
                and all(a not in ids or b in ids for a, b in self.dependencies))

    def closure(self, selected: Iterable[str]) -> tuple[str, ...]:
        ids = set(selected)
        if not ids.issubset(self.by_id):
            raise ValueError("unknown option")
        changed = True
        while changed:
            changed = False
            for a, b in self.dependencies:
                if a in ids and b not in ids:
                    ids.add(b)
                    changed = True
        return tuple(sorted(ids))

    def dependency_order(self, selected: Sequence[str]) -> tuple[str, ...]:
        ids = set(selected)
        edges = {i: set() for i in ids}
        for a, b in self.dependencies:
            if a in ids and b in ids:
                edges[a].add(b)
        ordered = []
        while edges:
            enabled = sorted(i for i, prerequisites in edges.items() if not prerequisites)
            if not enabled:
                raise ValueError("cyclic check dependencies cannot execute")
            ordered.extend(enabled)
            for i in enabled:
                del edges[i]
            for prerequisites in edges.values():
                prerequisites.difference_update(enabled)
        return tuple(ordered)


@dataclass(frozen=True)
class Batch:
    option_ids: tuple[str, ...]
    origin: str = "proposal"

    def __post_init__(self) -> None:
        if len(set(self.option_ids)) != len(self.option_ids):
            raise ValueError("duplicate operation in batch")
        object.__setattr__(self, "option_ids", tuple(sorted(self.option_ids)))

    @property
    def uid(self) -> str:
        # Canonical physical work, regardless of which menu generator proposed it.
        return digest(self.option_ids)


@dataclass(frozen=True)
class SolveResult:
    batch: Batch
    surrogate_value: float
    optimal: bool
    status: str


def exhaustive_batch(problem: BatchProblem) -> SolveResult:
    if len(problem.options) > 24:
        raise ValueError("exhaustive proposal capped at 24 operations")
    best, best_value = (), 0.
    options = sorted(problem.options, key=lambda o: o.uid)
    for mask in product((False, True), repeat=len(options)):
        chosen = tuple(o.uid for o, included in zip(options, mask) if included)
        if problem.feasible(chosen):
            value = math.fsum(o.predicted_value for o, included in zip(options, mask) if included)
            if value > best_value + 1e-12:
                best, best_value = chosen, value
    return SolveResult(Batch(best), best_value, True, "exhaustive-surrogate-optimum")


def greedy_batch(problem: BatchProblem) -> SolveResult:
    chosen: tuple[str, ...] = ()
    order = sorted(problem.options, key=lambda o: (-o.predicted_value / max(1, o.cap + problem.setup_costs[o.group]), o.uid))
    for option in order:
        trial = problem.closure((*chosen, option.uid))
        gain = sum(problem.by_id[i].predicted_value for i in trial if i not in chosen)
        if gain > 0 and problem.feasible(trial):
            chosen = trial
    return SolveResult(Batch(chosen, "greedy-fallback"), sum(problem.by_id[i].predicted_value for i in chosen), False, "checked-greedy-fallback")


def solve_batch(problem: BatchProblem, *, time_limit: float = 1., backend: str = "auto") -> SolveResult:
    """Optimize the proposal surrogate, never claim optimal future search utility."""
    if backend not in ("auto", "exhaustive", "scipy", "greedy") or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("invalid planner settings")
    if not problem.options:
        return SolveResult(Batch(()), 0., True, "empty")
    if backend == "greedy":
        return greedy_batch(problem)
    if backend == "exhaustive" or (backend == "auto" and len(problem.options) <= 16):
        return exhaustive_batch(problem)
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        options = problem.options
        n = len(options)
        groups = sorted({o.group for o in options})
        idx = {o.uid: i for i, o in enumerate(options)}
        gi = {g: n + i for i, g in enumerate(groups)}
        rows, upper = [], []

        def add(coefficients: Mapping[int, float], cap: float = 0.) -> None:
            row = np.zeros(n + len(groups))
            for i, value in coefficients.items():
                row[i] = value
            rows.append(row)
            upper.append(cap)

        add({**{i: o.cap - o.reserved_cap for i, o in enumerate(options)}, **{gi[g]: problem.setup_costs[g] for g in groups}}, problem.budget)
        for i, o in enumerate(options):
            add({i: 1., gi[o.group]: -1.})
        for g in groups:
            add({gi[g]: 1., **{i: -1. for i, o in enumerate(options) if o.group == g}})
        for a, b in problem.dependencies:
            add({idx[a]: 1., idx[b]: -1.})
        for a, b in problem.conflicts:
            add({idx[a]: 1., idx[b]: 1.}, 1.)
        reservation_groups: dict[str, list[int]] = {}
        for i, option in enumerate(options):
            if option.reserved_cap:
                reservation_groups.setdefault(option.reservation_id, []).append(i)
        for indices in reservation_groups.values():
            if len(indices) > 1:
                add({i: 1. for i in indices}, 1.)
        objective = np.array([-o.predicted_value for o in options] + [0.] * len(groups))
        result = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(0., 1.),
                      constraints=LinearConstraint(np.array(rows), -np.inf, np.array(upper)),
                      options={"time_limit": time_limit, "mip_rel_gap": 0.})
        if result.x is None or not np.all(np.isfinite(result.x)) or np.max(abs(result.x - np.rint(result.x))) > 1e-6:
            fallback = greedy_batch(problem)
            return SolveResult(fallback.batch, fallback.surrogate_value, False, "solver-no-valid-incumbent:" + str(result.message))
        chosen = tuple(o.uid for i, o in enumerate(options) if result.x[i] > .5)
        if not problem.feasible(chosen):
            raise RuntimeError("solver incumbent failed independent feasibility check")
        return SolveResult(Batch(chosen), sum(problem.by_id[i].predicted_value for i in chosen), result.status == 0, str(result.message))
    except ImportError:
        fallback = greedy_batch(problem)
        return SolveResult(fallback.batch, fallback.surrogate_value, False, "scipy-unavailable:" + fallback.status)


@dataclass(frozen=True)
class Menu:
    batches: tuple[Batch, ...]
    proposal_snapshot: str
    state_version: int
    solver_status: str


class MenuBuilder:
    def __init__(self, *, backend: str = "auto", max_options: int = 8, time_limit: float = 1., snapshot: str = "finite-menu-v1"):
        if type(max_options) is not int or max_options < 2:
            raise ValueError("menu must retain room for completion and exploration")
        self.backend, self.max_options, self.time_limit, self.snapshot = backend, max_options, time_limit, snapshot

    def build(self, problem: BatchProblem, *, completion_ids: Sequence[str] = ()) -> Menu:
        solved = solve_batch(problem, backend=self.backend, time_limit=self.time_limit)
        proposals: list[Batch] = []
        # Protected options are inserted before truncating optional diversification.
        if completion_ids:
            completion = problem.closure(completion_ids)
            if problem.feasible(completion):
                proposals.append(Batch(completion, "direct-completion"))
        feasible_singletons = [problem.closure((o.uid,)) for o in problem.options]
        feasible_singletons = [ids for ids in feasible_singletons if problem.feasible(ids)]
        if feasible_singletons:
            proposals.append(Batch(min(feasible_singletons, key=lambda ids: (problem.cost(ids), ids)), "exploration"))
        if solved.batch.option_ids:
            proposals.append(solved.batch)
        for ids in feasible_singletons:
            proposals.append(Batch(ids, "diversified"))
        batches: dict[str, Batch] = {}
        for batch in proposals:
            if problem.feasible(batch.option_ids):
                batches.setdefault(batch.uid, batch)
            if len(batches) >= self.max_options:
                break
        return Menu(tuple(batches.values()), self.snapshot, problem.state_version, solved.status)


def select_batch(menu: Menu, problem: BatchProblem, value_fn: Callable[[Batch], float]) -> Batch | None:
    legal = [b for b in menu.batches if problem.feasible(b.option_ids)]
    if not legal:
        return None
    values = [(float(value_fn(batch)), batch) for batch in legal]
    if any(not math.isfinite(v) for v, _ in values):
        raise ValueError("whole-batch values must be finite")
    return min(values, key=lambda item: (-item[0], item[1].uid))[1]
