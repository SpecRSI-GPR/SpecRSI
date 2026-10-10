"""Atomic cap reservations and exactly-once actual-cost settlement.

Costs use integer units selected by the experiment (e.g. tokens or microseconds).
Executors must enforce caps; expected costs cannot enforce a hard budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Mapping


class BudgetExceeded(ValueError):
    pass


class CostCapViolation(RuntimeError):
    pass


def cost_int(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("cost must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class Charge:
    job: str
    cap: int
    actual: int
    category: str
    cost_kind: str = "actual"


class BudgetLedger:
    def __init__(self, total: int):
        self.total = cost_int(total)
        self.spent = 0
        self._reserved: dict[str, tuple[int, str]] = {}
        self._finished: set[str] = set()
        self.charges: list[Charge] = []
        self.breached = False
        self.cost_valid = True
        self._lock = RLock()

    @property
    def reserved(self) -> dict[str, int]:
        with self._lock:
            return {k: v[0] for k, v in self._reserved.items()}

    @property
    def pending(self) -> int:
        return sum(self.reserved.values())

    @property
    def available(self) -> int:
        with self._lock:
            return 0 if self.breached or not self.cost_valid else self.total - self.spent - self.pending

    def reserve(self, job: str, cap: int, category: str = "execution") -> None:
        self.reserve_many({job: (cap, category)})

    def reserve_many(self, jobs: Mapping[str, tuple[int, str]]) -> None:
        """All jobs are admitted together, or none are admitted."""
        with self._lock:
            for job, (cap, _) in jobs.items():
                cost_int(cap)
                if not job or job in self._reserved or job in self._finished:
                    raise ValueError("reservation identity already used")
            if self.breached or sum(v[0] for v in jobs.values()) > self.available:
                raise BudgetExceeded("hard cap would consume outstanding reservations")
            self._reserved.update(jobs)

    def transfer(self, source: str, target: str, cap: int, category: str = "execution") -> None:
        """Move already reserved physical work, without buying it twice.

        Any increase is a new reservation and must fit free resources. A decrease
        refunds the difference. Transfer cannot resurrect a settled job.
        """
        cost_int(cap)
        with self._lock:
            if source not in self._reserved or target in self._finished or (target != source and target in self._reserved):
                raise ValueError("invalid reservation transfer")
            old = self._reserved[source][0]
            if self.breached or cap - old > self.available:
                raise BudgetExceeded("transfer increment exceeds budget")
            del self._reserved[source]
            self._reserved[target] = cap, category

    def release(self, job: str) -> int:
        """Cancel only work that has not started; execution is settled instead."""
        with self._lock:
            if job not in self._reserved:
                raise ValueError("unknown reservation")
            cap, _ = self._reserved.pop(job)
            self._finished.add(job)
            return cap

    def reserve_work(self, jobs: Mapping[str, tuple[int, str]], transfers: Mapping[str, int]) -> None:
        """Atomically increase matching existing work caps and admit new work."""
        with self._lock:
            for job, cap in transfers.items():
                cost_int(cap)
                if job not in self._reserved or job in jobs:
                    raise ValueError("invalid physical-work reservation transfer")
            for job, (cap, _) in jobs.items():
                cost_int(cap)
                if not job or job in self._reserved or job in self._finished:
                    raise ValueError("reservation identity already used")
            increment = sum(cap - self._reserved[job][0] for job, cap in transfers.items())
            if self.breached or increment + sum(x[0] for x in jobs.values()) > self.available:
                raise BudgetExceeded("batch would spend another task's completion reserve")
            for job, cap in transfers.items():
                self._reserved[job] = cap, self._reserved[job][1]
            self._reserved.update(jobs)

    def settle(self, job: str, actual: int, *, cost_kind: str = "actual") -> Charge:
        cost_int(actual)
        if cost_kind not in ("actual", "conservative_bound"):
            raise ValueError("invalid cost provenance")
        with self._lock:
            if job not in self._reserved:
                raise ValueError("unknown or already settled job")
            cap, category = self._reserved.pop(job)
            self.spent += actual
            self._finished.add(job)
            charge = Charge(job, cap, actual, category, cost_kind)
            self.charges.append(charge)
            if cost_kind != "actual":
                self.cost_valid = False
            if actual > cap:
                # Do not hide measured spending or continue after a broken executor.
                self.breached = True
                raise CostCapViolation(f"{job}: actual cost {actual} exceeded enforced cap {cap}")
            return charge

    def settle_unknown(self, job: str) -> Charge:
        """Fail-closed accounting for a crashed, unmetered worker; never actual C."""
        with self._lock:
            if job not in self._reserved:
                raise ValueError("unknown reservation")
            return self.settle(job, self._reserved[job][0], cost_kind="conservative_bound")

    def charge(self, job: str, actual: int, category: str) -> Charge:
        self.reserve(job, actual, category)
        return self.settle(job, actual)

    def invariant(self) -> bool:
        return not self.breached and self.cost_valid and self.spent + self.pending <= self.total

    def clone(self) -> "BudgetLedger":
        """Isolate a paid-state continuation, preserving every existing charge/cap."""
        with self._lock:
            clone = BudgetLedger(self.total)
            clone.spent = self.spent
            clone._reserved = dict(self._reserved)
            clone._finished = set(self._finished)
            clone.charges = list(self.charges)
            clone.breached, clone.cost_valid = self.breached, self.cost_valid
            return clone
