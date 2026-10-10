"""Runnable finite SpecRSI integration fixture, explicitly using heuristic value heads."""
from __future__ import annotations

from .acquisition import AcquisitionCaps, ResultAcquirer
from .budget import BudgetLedger
from .checker import ConservativeChecker
from .execution import PureIntegerExecutor
from .ledger import Ledger
from .models import Candidate, EvaluationScope, Task, thaw
from .search import DraftResult, SearchConfig, SearchEngine


class IntegerRepairDrafter:
    def draft(self, parent, state, cap, rng):
        if cap < 1:
            return DraftResult(None, cap, status="interrupted", error="draft cap exhausted")
        routes = thaw(parent.artifacts["routes"])
        routes["write"] = ("add", ("x",), ("const", 1))
        child = Candidate(f"repair-{len(state.candidates)}", {"routes": routes}, parent.runtime,
                          "pure_integer", True, parent_id=parent.uid)
        forecasts = {t.uid: (.95 if t.inputs["route"] == "read" else .05) for t in state.scope.tasks}
        return DraftResult(child, 1, forecasts)


def run_demo(*, seed: int = 7, budget: int = 120, mode: str = "deploy", all_u: bool = False) -> dict:
    tasks = (Task("read-path", {"route": "read", "x": 3}, 3),
             Task("normal-write", {"route": "write", "x": 2}, 3),
             Task("duplicate-write-case", {"route": "write", "x": 5}, 6))
    scope = EvaluationScope("finite-retry-illustration", tasks, "pure-int-v1", "integer-equality-v1",
                            scored_fields=("answer", "primitives"))
    incumbent = Candidate("incumbent", {"routes": {"read": ("x",), "write": ("x",)}},
                          {"contract": "pure-int-v1"}, "pure_integer", True)
    ledger = Ledger(ConservativeChecker(1))
    executor = PureIntegerExecutor()
    # Paper starts from a fully evaluated incumbent; setup cost is separately reported.
    initial_budget = BudgetLedger(100)
    acquirer = ResultAcquirer(ledger, initial_budget, executor)
    for task in tasks:
        acquirer.acquire(incumbent, task, scope, (), AcquisitionCaps(10))
    config = SearchConfig(execution_cap=10, draft_cap=2, max_actions=8, max_candidates=3, max_children=1,
                          seed=seed, mode=mode, all_u=all_u, cost_unit="deterministic-primitive-and-controller-work")
    engine = SearchEngine(scope, incumbent, ledger, BudgetLedger(budget), executor, IntegerRepairDrafter(), config)
    result = engine.run()
    output = result.to_dict()
    output["initial_incumbent_cost"] = initial_budget.spent
    output["provenance_records_checked"] = ledger.audit_provenance()
    output["demo_only"] = True
    output["note"] = "Finite integer fixture with heuristic heads; not a paper benchmark or a learned checkpoint reproduction."
    return output


if __name__ == "__main__":
    import json
    print(json.dumps(run_demo(), indent=2, ensure_ascii=False))
