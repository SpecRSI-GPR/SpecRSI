from dataclasses import replace

import pytest

from specrsi.core import AcquisitionCaps, BudgetLedger, Candidate, ConservativeChecker, EvaluationScope, Ledger, PureIntegerExecutor, ResultAcquirer, Task
from specrsi.core.demo import IntegerRepairDrafter
from specrsi.core.search import DraftResult, ValueEstimate
from specrsi.core.vats import ScorerQueryFailure, VATSConfig, VATSEngine
from specrsi.planning.completion import CompletionProblem, VerificationJob, solve_completion


def make_vats(**kwargs):
    tasks = (Task("read", {"route": "read", "x": 2}, 2), Task("write", {"route": "write", "x": 2}, 3))
    scope = EvaluationScope("vats", tasks, "pure-int-v1", "integer-equality-v1", ("answer", "primitives"))
    incumbent = Candidate("base", {"routes": {"read": ("x",), "write": ("x",)}}, {"contract": "pure-int-v1"}, "pure_integer", True)
    ledger, executor = Ledger(ConservativeChecker(1)), PureIntegerExecutor()
    initial = ResultAcquirer(ledger, BudgetLedger(100), executor)
    for task in tasks:
        initial.acquire(incumbent, task, scope, (), AcquisitionCaps(10))
    config = kwargs.pop("config", VATSConfig(execution_cap=10, draft_cap=1, max_actions=4, max_candidates=3,
                    max_children=1, preparation_cap=12, max_queries=4, solver_backend="exhaustive"))
    return VATSEngine(scope, incumbent, ledger, BudgetLedger(kwargs.pop("budget", 200)), executor,
                      kwargs.pop("drafter", IntegerRepairDrafter()), config, **kwargs)


def test_pdf_direct_only_completion_and_preparation_cost_origins():
    engine = make_vats()
    result = engine.run()
    assert result.acceptance.improved
    assert {record.route for record in engine.state.ledger.records.values()} == {"U"}
    assert result.total_cost == sum(t.cost for t in result.transitions) == sum(c["actual"] for c in result.charges)
    assert all(t.category in {"preparation", "work"} for t in result.transitions)
    for decision in result.decisions:
        assert decision.kind == "plan"
        assert result.transitions[decision.transition_index - 1].action == "prepare"
        assert result.transitions[decision.transition_index].category == "work"
        assert decision.state_version == result.transitions[decision.transition_index].state_version_before
        assert "ACCEPT" in decision.support
        assert decision.values[decision.support.index("ACCEPT")] == 0.
    assert result.algorithm == "vats-pdf-v1"


def test_policy_is_independent_of_scorer_and_accept_keeps_partial_work_partial():
    def logits(state, plans):
        return [3. if p.kind == "draft" and len(state.candidates) == 1 else -10. for p in plans]

    engine = make_vats(scorer=lambda state, plan: 1e6, policy_menu=logits)
    result = engine.run()
    assert result.stop_reason == "policy_accept"
    assert [t.action for t in result.transitions] == ["prepare", "draft", "prepare", "accept"]
    assert result.selected == engine.state.incumbent
    assert len(engine.state.ledger.records) == 2
    assert engine.state.budget.pending == 0
    assert result.transitions[-1].cost == 0


@pytest.mark.parametrize("mode", ["collect", "deploy"])
@pytest.mark.parametrize("sampled_edit", [False, True])
def test_plan_mode_never_relabels_the_actual_drafter_sampling_law(mode, sampled_edit):
    from specrsi.planning.policy import TraceDecision

    class TracedDrafter:
        def draft(self, parent, state, cap, rng):
            support = ("edit:a", "edit:b")
            probabilities = (.25, .75) if sampled_edit else (1., 0.)
            selected = rng.choices(range(2), weights=probabilities)[0] if sampled_edit else 0
            child = replace(parent, name=parent.name + support[selected], parent_id=parent.uid)
            trace = TraceDecision("edit", support, support[selected], probabilities, (0., 0.),
                state.version, state.current_transition_index, 1., 0., stochastic=sampled_edit)
            return DraftResult(child, 1, edit_decision=trace)

    config = VATSConfig(execution_cap=10, draft_cap=1, max_actions=1, max_candidates=2, max_children=1,
                        preparation_cap=12, max_queries=0, mode=mode, exploration=0.)
    engine = make_vats(config=config, drafter=TracedDrafter(),
        policy_menu=lambda state, plans: [1000. if p.kind == "draft" else 0. for p in plans])
    result = engine.run()
    edit = next(d for d in result.decisions if d.kind == "edit")
    assert edit.stochastic is sampled_edit
    assert edit.probabilities == ((.25, .75) if sampled_edit else (1., 0.))
    assert result.transitions[edit.transition_index].action == "draft"
    assert edit.state_version == result.transitions[edit.transition_index].state_version_before


def test_negative_checks_do_not_remove_completion_seeds_or_cross_branch_union():
    seen = []

    def policy(state, plans):
        seen.append(plans)
        if len(state.candidates) < 3:
            return [10. if p.kind == "draft" and p.depth == 1 else -1. for p in plans]
        return [-1. for p in plans]

    engine = make_vats(scorer=lambda state, plan: -100., policy_menu=policy)
    engine.run()
    menu = seen[-1]
    assert any(p.origin == "completion" and len(p.jobs) == 2 for p in menu)
    assert any(p.origin == "cross-branch-completion" and len(p.jobs) == 4 for p in menu)
    assert any(p.origin == "singleton" for p in menu)


def test_same_conservative_state_for_refinement_and_paid_policy_input():
    scorer_budgets, policy_budgets = [], []

    def scorer(state, plan):
        scorer_budgets.append((state.version, state.budget.available))
        return ValueEstimate(float(len(plan.jobs)), 0)

    def policy(state, plans):
        policy_budgets.append((state.version, state.budget.available, state.budget.spent))
        return [2. if p.kind == "draft" and len(state.candidates) == 1 else -1. for p in plans]

    engine = make_vats(scorer=scorer, policy_menu=policy)
    result = engine.run()
    for version in {v for v, _ in scorer_budgets}:
        assert len({b for v, b in scorer_budgets if v == version}) == 1
    for decision, (_, available, spent) in zip(result.decisions, policy_budgets):
        assert decision.state_features[0] == pytest.approx(available / engine.state.budget.total)
        assert spent >= decision.planning_cost_paid


def test_bounded_depth_drafts_expand_actual_incomplete_parent_and_charge_failure():
    parents = []

    class PrefixDrafter:
        def draft(self, parent, state, cap, rng):
            parents.append(parent.uid)
            if len(parents) == 2:
                return DraftResult(None, 1, status="error", error="invalid second edit")
            return IntegerRepairDrafter().draft(parent, state, cap, rng)

    config = VATSConfig(execution_cap=10, draft_cap=1, max_actions=1, max_candidates=4,
                        max_children=1, max_draft_depth=3, max_attempts=3, preparation_cap=2,
                        max_queries=0, solver_call_cost=0, solver_backend="exhaustive")
    engine = make_vats(config=config, drafter=PrefixDrafter(), policy=lambda state, p: float(p.depth if p.kind == "draft" else -1))
    result = engine.run()
    child = next(c for c in engine.state.candidates.values() if c.uid != engine.state.incumbent.uid)
    assert parents == [engine.state.incumbent.uid, child.uid]
    assert engine.state.remaining_attempts == 1
    assert result.transitions[-1].cost == 2
    assert len(engine.state.candidates) == 2
    assert not result.acceptance.improved


def test_attempt_cap_and_preparation_cap_prevent_unaffordable_work():
    cfg = VATSConfig(execution_cap=10, draft_cap=1, max_attempts=1, preparation_cap=2,
                      max_queries=0, solver_call_cost=0, solver_backend="exhaustive")
    engine = make_vats(config=cfg)
    result = engine.run()
    assert result.stop_reason == "attempt_limit"
    assert len(engine.state.candidates) == 2
    assert result.selected == engine.state.incumbent
    engine = make_vats(config=cfg, budget=1)
    assert engine.run().total_cost == 0


def test_completion_milp_minimizes_cover_cost_with_dependencies_and_shared_setup():
    jobs = (VerificationJob("a", (("c1", "one"),), 3, group="shared"),
            VerificationJob("b", (("c2", "two"),), 3, group="shared"),
            VerificationJob("joint", (("c1", "one"), ("c2", "two")), 9, group="single"),
            VerificationJob("prerequisite", (("c1", "aux"),), 1, group="shared"))
    problem = CompletionProblem(jobs, {"shared": 1, "single": 0}, 10, 3,
                                 dependencies=(("a", "prerequisite"),), conflicts=(("a", "joint"),))
    target = {("c1", "one"), ("c2", "two")}
    exact = solve_completion(problem, target, backend="exhaustive")
    assert exact.job_ids == ("a", "b", "prerequisite")
    assert exact.cost_cap == 8
    scipy = pytest.importorskip("scipy.optimize")
    solved = solve_completion(problem, target)
    assert solved.cost_cap == exact.cost_cap and solved.optimal
    assert solve_completion(replace(problem, budget=7), target) is None
    limited = solve_completion(replace(problem, attempts=1), target)
    assert limited.job_ids == ("joint",)


def test_pdf_configuration_rejects_hidden_margin_and_insufficient_seed_slots():
    with pytest.raises(ValueError, match="without a margin"):
        VATSConfig(margin=.01)
    with pytest.raises(ValueError, match="promised completion"):
        VATSConfig(max_menu_options=3)
    with pytest.raises(ValueError, match="direct execution"):
        VATSConfig(all_u=False)


def test_paid_state_forks_isolate_results_force_legal_work_without_repaying_preparation():
    engine = make_vats(capture_paid_states=True)
    engine.run()
    saved = next(snapshot for snapshot in engine.paid_snapshots if any(p.kind == "verify" for p in snapshot.plans))
    complete = next(p for p in saved.plans if p.origin == "completion")
    before = (saved.state.budget.spent, len(saved.state.ledger.records), saved.state.remaining_attempts)
    accept_branch = VATSEngine.from_paid_state(saved, "ACCEPT", executor=PureIntegerExecutor(),
                       drafter=IntegerRepairDrafter(), seed=80)
    complete_branch = VATSEngine.from_paid_state(saved, complete, executor=PureIntegerExecutor(),
                       drafter=IntegerRepairDrafter(), seed=81, policy=lambda state, p: -10.)
    stopped = accept_branch.run()
    improved = complete_branch.run()
    assert stopped.total_cost == 0 and not stopped.decisions
    assert stopped.stop_reason == "forced_accept"
    assert improved.acceptance.improved
    assert improved.transitions[0].action == "verify"
    assert improved.transitions[0].category == "forced-work"
    assert improved.total_cost == sum(c["actual"] for c in improved.charges)
    assert (saved.state.budget.spent, len(saved.state.ledger.records), saved.state.remaining_attempts) == before
    assert stopped.initial_context_id == improved.initial_context_id
    assert accept_branch.state.ledger is not complete_branch.state.ledger
    with pytest.raises(ValueError, match="outside"):
        VATSEngine.from_paid_state(saved, "nonexistent", executor=PureIntegerExecutor(), drafter=IntegerRepairDrafter(), seed=82)


def test_metered_failed_scorer_query_retains_seeds_without_inventing_zero():
    def failed(state, plan):
        raise ScorerQueryFailure("bounded timeout", 1)

    engine = make_vats(scorer=failed)
    result = engine.run()
    queries = [event for t in result.transitions for event in t.events if event["event"] == "scorer_query"]
    assert queries and all(q["value"] is None and q["cost"] == 1 for q in queries)
    assert result.acceptance.improved
    assert engine.state.budget.invariant()


def test_many_parents_do_not_exclude_deep_or_multibranch_drafts():
    engine = make_vats(config=VATSConfig(max_candidates=30, max_children=2, max_attempts=100, draft_cap=1))
    for i in range(12):
        child = replace(engine.state.incumbent, name=f"existing-{i}", parent_id=engine.state.incumbent.uid)
        engine.state.candidates[child.uid] = child
        engine.state.ledger.register(child, engine.state.scope)
    plans = engine._draft_plans(engine.state)
    assert any(p.depth > 1 for p in plans)
    assert any(len(p.parent_ids) > 1 for p in plans)
    assert any(p.depth == 1 and len(p.parent_ids) == 1 for p in plans)


def test_interrupted_results_remain_missing_and_resume_is_passed_to_same_frozen_item():
    from specrsi.core.execution import ExecutionResult

    class ResumeExecutor:
        def __init__(self):
            self.calls = []

        def run(self, candidate, task, scope, cap, resume_token=None):
            self.calls.append((candidate.uid, task.uid, resume_token))
            if resume_token is None:
                return ExecutionResult("interrupted", 1, resume_token=f"resume:{candidate.uid}:{task.uid}")
            assert resume_token == f"resume:{candidate.uid}:{task.uid}"
            return PureIntegerExecutor().run(candidate, task, scope, cap)

    cfg = VATSConfig(execution_cap=10, draft_cap=1, max_actions=3, max_candidates=2, max_children=1,
                      preparation_cap=5, max_queries=0, solver_backend="exhaustive")
    engine = make_vats(config=cfg)
    executor = ResumeExecutor()
    engine.executor = engine.acquirer.executor = executor
    result = engine.run()
    assert result.acceptance.improved
    assert len(executor.calls) == 4
    acquisitions = [t for t in result.transitions if t.action == "verify"]
    assert acquisitions[0].f_after == 0
    assert all(e.get("route") is None for e in acquisitions[0].events if e["event"] == "acquisition")
    assert engine.state.remaining_attempts == cfg.max_attempts - 5


def test_scorer_cap_violation_records_the_actual_overrun_and_stops():
    from specrsi.core import CostCapViolation

    engine = make_vats(scorer=lambda state, plan: ValueEstimate(1., 1000))
    with pytest.raises(CostCapViolation):
        engine.run()
    assert engine.state.budget.breached
    assert engine.state.budget.charges[-1].actual >= 1000
    assert engine.state.budget.charges[-1].cost_kind == "actual"


def test_bootstrap_verification_preference_does_not_depend_on_expense_unit():
    from specrsi.core.vats import VATSPlan, heuristic_vats_preference

    engine = make_vats()
    child = replace(engine.state.incumbent, name="child", parent_id=engine.state.incumbent.uid)
    engine.state.candidates[child.uid] = child
    engine.state.ledger.register(child, engine.state.scope)
    jobs = tuple(VerificationJob(task.name, ((child.uid, task.uid),), 10) for task in engine.state.scope.tasks)
    plan = VATSPlan("verify", jobs=jobs, work_cap=20, attempt_cap=2)
    original = heuristic_vats_preference(engine.state, plan)
    engine.state.budget = BudgetLedger(engine.state.budget.total * 10000)
    scaled = replace(plan, work_cap=plan.work_cap * 10000,
                     jobs=tuple(replace(job, cap=job.cap * 10000) for job in jobs))
    assert heuristic_vats_preference(engine.state, scaled) == pytest.approx(original)
    assert original > heuristic_vats_preference(engine.state, VATSPlan("accept"))


class CompoundIntegerBackend:
    """Actually run the pure integer evaluator; no predicted coverage is evidence."""
    version = "compound-integer-test-v1"

    def __init__(self, *, fail_setup=False, fail_prerequisite=False, partial=False,
                 missing_catalog_key=False, bad_receipt=False, exceed_attempts=False):
        self.fail_setup, self.fail_prerequisite = fail_setup, fail_prerequisite
        self.partial, self.missing_catalog_key = partial, missing_catalog_key
        self.bad_receipt, self.exceed_attempts = bad_receipt, exceed_attempts
        self.setup_calls = []
        self.job_calls = []
        self.executions = []

    def catalog(self, context, *, budget, attempts):
        jobs, dependencies = [], []
        for candidate in context.candidates.values():
            keys = tuple((candidate.uid, task.uid) for task in context.scope.tasks
                         if (candidate.uid, task.uid, context.scope.uid) not in context.records)
            if not keys:
                continue
            if self.missing_catalog_key:
                keys = keys[:1]
            prerequisite = VerificationJob(f"prepare:{candidate.uid}", (), 1, kind="prerequisite")
            compound = VerificationJob(f"evaluate:{candidate.uid}", keys, 10, attempts=len(keys))
            jobs.extend((prerequisite, compound))
            dependencies.append((compound.uid, prerequisite.uid))
        return CompletionProblem(tuple(jobs), {"execution": 3}, budget, attempts, tuple(dependencies))

    def setup(self, group, context, *, cap):
        from specrsi.core.jobs import SetupExecutionResult
        self.setup_calls.append(group)
        if self.fail_setup:
            return SetupExecutionResult("error", 1, error="setup failed")
        return SetupExecutionResult("complete", 2, {"setup_ready": True},
                    {"group": group, "execution_id": context.scope.execution_id, "run_id": f"setup-{len(self.setup_calls)}"})

    def run(self, job, context, *, setup, prerequisites, resume_token=None):
        from specrsi.core.jobs import ExecutedJobOutcome, JobExecutionResult
        self.job_calls.append((job.uid, resume_token))
        assert setup["setup_ready"]
        if job.kind == "prerequisite":
            return JobExecutionResult("error" if self.fail_prerequisite else "complete", 1, 1,
                                      artifacts={"ready": not self.fail_prerequisite})
        assert prerequisites and all(result.artifacts["ready"] for result in prerequisites.values())
        keys = job.keys[:1] if self.partial and resume_token is None else job.keys
        outcomes, cost = [], 0
        for candidate_id, task_id in keys:
            task = next(t for t in context.scope.tasks if t.uid == task_id)
            execution = PureIntegerExecutor().run(context.candidates[candidate_id], task, context.scope, job.cap - cost)
            self.executions.append((candidate_id, task_id))
            cost += execution.actual_cost
            receipt = {"job_id": job.uid, "candidate_id": candidate_id, "task_id": task_id,
                       "scope_id": context.scope.uid, "execution_id": context.scope.execution_id,
                       "run_id": f"physical-{len(self.executions)}", **dict(execution.receipt)}
            if self.bad_receipt:
                receipt["scope_id"] = "wrong-scope"
            outcomes.append(ExecutedJobOutcome(candidate_id, task_id, execution.outcome, receipt))
        interrupted = len(keys) < len(job.keys)
        return JobExecutionResult("interrupted" if interrupted else "complete", cost,
                                  job.attempts + 1 if self.exceed_attempts else len(keys), tuple(outcomes),
                                  resume_token="resume-compound" if interrupted else None)


def make_compound(backend, **kwargs):
    config = VATSConfig(execution_cap=10, draft_cap=1, max_actions=kwargs.pop("max_actions", 2), max_candidates=2,
                        max_children=1, preparation_cap=5, max_queries=0, solver_backend="exhaustive")
    return make_vats(config=config, verification_backend=backend, **kwargs)


def test_compound_jobs_execute_real_coverage_after_prerequisite_with_one_shared_setup():
    backend = CompoundIntegerBackend()
    engine = make_compound(backend)
    result = engine.run()
    assert result.acceptance.improved
    assert backend.setup_calls == ["execution"]
    assert [uid.split(":")[0] for uid, _ in backend.job_calls] == ["prepare", "evaluate"]
    assert len(backend.executions) == 2
    assert result.transitions[-1].cost == 7  # shared setup 2 + prerequisite 1 + actual evaluator 4
    assert engine.state.remaining_attempts == engine.config.max_attempts - 4
    assert engine.state.ledger.audit_provenance() == 4
    assert engine.state.budget.pending == 0


@pytest.mark.parametrize("failure", ["fail_setup", "fail_prerequisite"])
def test_failed_operational_dependencies_block_descendants_without_fake_coverage(failure):
    backend = CompoundIntegerBackend(**{failure: True})
    engine = make_compound(backend)
    result = engine.run()
    assert not result.acceptance.improved
    assert not backend.executions
    assert len(engine.state.ledger.records) == 2
    blocked = [event for event in result.transitions[-1].events if event["event"] == "blocked_job"]
    assert blocked and all(event["cost"] == 0 and event["attempts"] == 0 for event in blocked)
    assert result.transitions[-1].cost == (1 if failure == "fail_setup" else 3)
    assert engine.state.budget.pending == 0


def test_compound_partial_results_admit_only_receipted_subset_and_resume_missing_keys():
    backend = CompoundIntegerBackend(partial=True)
    engine = make_compound(backend, max_actions=3)
    result = engine.run()
    verifications = [t for t in result.transitions if t.action == "verify"]
    assert verifications[0].f_after == 0 and verifications[1].f_after > 0
    assert len(backend.executions) == len(set(backend.executions)) == 2
    assert [token for uid, token in backend.job_calls if uid.startswith("evaluate:")] == [None, "resume-compound"]
    assert result.acceptance.improved


def test_incomplete_catalog_cannot_shrink_the_required_completion_target():
    engine = make_compound(CompoundIntegerBackend(missing_catalog_key=True), capture_paid_states=True)
    result = engine.run()
    menu = engine.paid_snapshots[-1].plans
    assert any(plan.kind == "verify" for plan in menu)
    assert not any(plan.origin == "completion" for plan in menu)
    assert not result.acceptance.improved


def test_singleton_fallback_cannot_be_crowded_out_by_operational_prerequisites():
    cfg = VATSConfig(execution_cap=10, draft_cap=1, max_actions=1, max_candidates=2,
                     max_children=0, preparation_cap=2, max_queries=0, max_completion_targets=0,
                     max_pair_targets=0, max_singletons=1, solver_backend="exhaustive")
    engine = make_vats(config=cfg, verification_backend=CompoundIntegerBackend(),
                       policy=lambda state, plan: -1., capture_paid_states=True)
    child = replace(engine.state.incumbent, name="missing-checks", parent_id=engine.state.incumbent.uid)
    engine.state.candidates[child.uid] = child
    engine.state.ledger.register(child, engine.state.scope)
    engine.run()
    plan = next(plan for plan in engine.paid_snapshots[0].plans if plan.kind == "verify")
    assert plan.origin == "singleton"
    assert {job.kind for job in plan.jobs} == {"prerequisite", "result"}
    assert {key for job in plan.jobs for key in job.keys} == {(child.uid, task.uid) for task in engine.state.scope.tasks}


def test_compound_receipts_are_checked_after_charging_and_before_any_evidence_is_admitted():
    engine = make_compound(CompoundIntegerBackend(bad_receipt=True))
    with pytest.raises(ValueError, match="receipt"):
        engine.run()
    assert len(engine.state.ledger.records) == 2
    assert engine.state.budget.charges[-1].actual == 4
    assert engine.state.budget.pending == 0


def test_compound_attempt_overrun_is_charged_and_stops_before_admission():
    from specrsi.core import CostCapViolation
    engine = make_compound(CompoundIntegerBackend(exceed_attempts=True))
    with pytest.raises(CostCapViolation, match="attempt"):
        engine.run()
    assert engine.state.budget.breached
    assert len(engine.state.ledger.records) == 2
    assert engine.state.budget.charges[-1].actual == 4


def test_compound_paid_forks_require_a_matching_isolated_backend():
    engine = make_compound(CompoundIntegerBackend(), capture_paid_states=True)
    engine.run()
    saved = engine.paid_snapshots[-1]
    plan = next(p for p in saved.plans if p.origin == "completion")
    with pytest.raises(ValueError, match="same compound"):
        VATSEngine.from_paid_state(saved, plan, executor=PureIntegerExecutor(), drafter=IntegerRepairDrafter(), seed=1)
    restored = VATSEngine.from_paid_state(saved, plan, executor=PureIntegerExecutor(), drafter=IntegerRepairDrafter(), seed=1,
                                          verification_backend=CompoundIntegerBackend())
    assert restored.run().acceptance.improved


def test_timing_separates_objective_phases_from_overall_host_time_and_accept_remains_zero():
    engine = make_vats(policy=lambda state, plan: -1.)
    result = engine.run()
    assert result.transitions[-1].action == "accept"
    assert result.transitions[-1].elapsed_seconds == 0
    timing = result.to_dict()["timing"]
    assert timing["objective_elapsed_seconds"] == sum(t.elapsed_seconds for t in result.transitions)
    assert timing["overall_elapsed_seconds"] == pytest.approx(
        timing["objective_elapsed_seconds"] + timing["bookkeeping_elapsed_seconds"])
    assert timing["bookkeeping_elapsed_seconds"] >= 0


def test_multibranch_depth_plan_uses_each_real_parent_and_failure_only_ends_that_path():
    parents = []

    class BranchDrafter:
        def draft(self, parent, state, cap, rng):
            parents.append(parent)
            if len(parents) == 2:
                return DraftResult(None, 1, status="error", error="first path cannot continue")
            return IntegerRepairDrafter().draft(parent, state, cap, rng)

    cfg = VATSConfig(execution_cap=10, draft_cap=1, max_actions=1, max_candidates=8, max_children=1,
                      max_draft_depth=2, max_draft_parents=2, preparation_cap=5, max_queries=0,
                      solver_backend="exhaustive")
    engine = make_vats(config=cfg, drafter=BranchDrafter(),
                       policy=lambda state, p: 10. if p.kind == "draft" and len(p.parent_ids) == 2 and p.depth == 2 else -1.)
    sibling = replace(engine.state.incumbent, name="existing-partial-branch", parent_id=engine.state.incumbent.uid)
    engine.state.candidates[sibling.uid] = sibling
    engine.state.ledger.register(sibling, engine.state.scope)
    result = engine.run()
    assert len(parents) == 4
    assert {parents[0].uid, parents[2].uid} == {engine.state.incumbent.uid, sibling.uid}
    assert parents[1].parent_id == parents[0].uid
    assert parents[3].parent_id == parents[2].uid
    assert len(engine.state.candidates) == 5
    assert engine.state.remaining_attempts == cfg.max_attempts - 4
    assert result.transitions[-1].cost == 4
    assert result.selected == engine.state.incumbent


def test_conflicting_compound_jobs_are_rejected_before_any_backend_call():
    backend = CompoundIntegerBackend()
    engine = make_compound(backend)
    child = replace(engine.state.incumbent, name="unverified", parent_id=engine.state.incumbent.uid)
    engine.state.candidates[child.uid] = child
    engine.state.ledger.register(child, engine.state.scope)
    problem = engine._catalog(engine.state)
    source = next(job for job in problem.jobs if job.kind == "result")
    duplicate = replace(source, uid="alternative")
    problem = replace(problem, jobs=(*problem.jobs, duplicate), conflicts=((source.uid, duplicate.uid),))
    selected = problem.closure((source.uid, duplicate.uid))
    plan = engine._verify_plan(selected, problem)
    with pytest.raises(ValueError, match="dependency/conflict"):
        engine._execute_plan(plan, problem)
    assert not backend.setup_calls and not backend.job_calls
    assert engine.state.budget.spent == 0


def test_solver_runtime_failure_is_paid_and_preserves_executable_fallbacks(monkeypatch):
    """Algorithm 2 buys a solver attempt, not a promise of a successful seed."""
    calls = []

    def failing_solver(problem, target, **kwargs):
        calls.append(frozenset(target))
        raise RuntimeError("bounded solver backend failure")

    monkeypatch.setattr("specrsi.core.vats.solve_completion", failing_solver)
    config = VATSConfig(execution_cap=10, draft_cap=1, max_actions=3, max_candidates=2,
                        max_children=1, preparation_cap=5, max_queries=0,
                        solver_call_cost=1, solver_backend="exhaustive")
    engine = make_vats(config=config, capture_paid_states=True)
    result = engine.run()
    solve_events = [event for transition in result.transitions for event in transition.events
                    if event["event"] == "completion_solve"]
    assert len(solve_events) == len(calls) > 0
    assert all(event["status"] == "solver-error" and event["cost"] == 1
               and not event["optimal"] for event in solve_events)
    assert all("ACCEPT" in decision.support for decision in result.decisions)
    assert any(plan.origin == "singleton" for state in engine.paid_snapshots for plan in state.plans)
    # Paid singleton fallbacks can still complete both real checks. A missing
    # MILP result never creates evidence or prevents using the surviving support.
    assert result.acceptance.improved
    assert result.total_cost == sum(charge["actual"] for charge in result.charges)
    assert engine.state.budget.cost_valid and engine.state.budget.pending == 0


def test_serial_unknown_execution_failure_consumes_its_attempt_cap():
    """F.4: an exception cannot erase a physically launched verification attempt."""
    class RaisingExecutor:
        def run(self, *args, **kwargs):
            raise RuntimeError("physical evaluator interrupted without a receipt")

    engine = make_vats()
    child = replace(engine.state.incumbent, name="failed-check", parent_id=engine.state.incumbent.uid)
    engine.state.candidates[child.uid] = child
    engine.state.ledger.register(child, engine.state.scope)
    engine.executor = engine.acquirer.executor = RaisingExecutor()
    problem = engine._catalog(engine.state)
    plan = engine._verify_plan((problem.jobs[0].uid,), problem)
    before = engine.state.remaining_attempts
    with pytest.raises(RuntimeError, match="without a receipt"):
        engine._execute_plan(plan, problem)
    assert engine.state.remaining_attempts == before - plan.attempt_cap
    assert len(engine.state.ledger.records) == 2
    assert engine.state.budget.pending == 0
    assert engine.state.budget.charges[-1].cost_kind == "conservative_bound"


def test_serial_compound_exception_charges_launched_attempts_only():
    class RaisingBackend(CompoundIntegerBackend):
        def run(self, job, *args, **kwargs):
            self.job_calls.append((job.uid, None))
            raise RuntimeError("prerequisite interrupted without a receipt")

    backend = RaisingBackend()
    engine = make_compound(backend)
    with pytest.raises(RuntimeError, match="prerequisite interrupted"):
        engine.run()
    # One draft and one launched prerequisite; the dependent task checks never
    # started, so their released reservations consume neither attempts nor cost.
    assert len(backend.job_calls) == 1
    assert engine.state.remaining_attempts == engine.config.max_attempts - 2
    assert len(engine.state.ledger.records) == 2
    assert engine.state.budget.pending == 0
    assert engine.state.budget.charges[-1].cost_kind == "conservative_bound"
