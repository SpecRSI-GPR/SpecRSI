from dataclasses import replace
import random

import pytest

from specrsi.planning import (Batch, BatchProblem, CheckOption, MenuBuilder, exhaustive_batch,
                          menu_distribution, menu_ratio, select_batch, solve_batch)


def problem(budget=10):
    options = (CheckOption("normal", "c", ("a",), 3, "runtime", .1),
               CheckOption("duplicate", "c", ("b",), 3, "runtime", .1),
               CheckOption("proxy", "c", ("a",), 3, "runtime", .8))
    return BatchProblem(options, {"runtime": 2}, budget, conflicts=(("normal", "proxy"),))


def test_shared_setup_dependencies_conflicts_and_complementary_coverage():
    p = problem(8)
    assert p.cost(("normal", "duplicate")) == 8
    assert p.feasible(("normal", "duplicate"))
    assert not p.feasible(("normal", "proxy"))
    assert not p.feasible(("normal", "normal"))
    dependent = replace(p, dependencies=(("duplicate", "normal"),))
    assert not dependent.feasible(("duplicate",))
    assert dependent.dependency_order(("duplicate", "normal")) == ("normal", "duplicate")
    with pytest.raises(ValueError, match="cyclic"):
        replace(dependent, dependencies=(("duplicate", "normal"), ("normal", "duplicate")))
    # Shared coverage alone is not grounds for a conflict: conjunctive checks may
    # be necessary together, and only the evaluator later admits an actual result.
    overlapping = BatchProblem((CheckOption("x", "c", ("t",), 1), CheckOption("y", "c", ("t",), 1)), {"default": 0}, 2)
    assert overlapping.feasible(("x", "y"))


def test_solver_matches_independent_exhaustive_for_random_small_problems():
    pytest.importorskip("scipy.optimize")
    rng = random.Random(18)
    for _ in range(12):
        options = tuple(CheckOption(str(i), "c", (str(i),), rng.randrange(1, 6), str(i % 2), rng.uniform(-1, 2)) for i in range(7))
        p = BatchProblem(options, {"0": 2, "1": 1}, rng.randrange(3, 18),
                         dependencies=(("2", "0"),), conflicts=(("1", "3"),))
        exact = exhaustive_batch(p)
        solved = solve_batch(p, backend="scipy", time_limit=5.)
        assert p.feasible(solved.batch.option_ids)
        assert solved.surrogate_value == pytest.approx(exact.surrogate_value)
        assert solved.optimal


def test_menu_preserves_affordable_completion_and_exploration_then_whole_batch_head():
    p = problem(8)
    menu = MenuBuilder(max_options=4).build(p, completion_ids=("normal", "duplicate"))
    assert any(set(b.option_ids) == {"normal", "duplicate"} for b in menu.batches)
    assert any(b.origin == "exploration" for b in menu.batches)
    chosen = select_batch(menu, p, lambda b: 10. if set(b.option_ids) == {"normal", "duplicate"} else 0.)
    assert set(chosen.option_ids) == {"normal", "duplicate"}
    # Its additive surrogate is lower: the separate whole-batch head can reverse it.
    assert exhaustive_batch(p).surrogate_value > .2


def test_canonical_menu_deduplicates_identical_physical_work():
    p = BatchProblem((CheckOption("only", "c", ("task",), 1),), {"default": 0}, 1)
    menu = MenuBuilder().build(p, completion_ids=("only",))
    assert len(menu.batches) == 1
    assert Batch(("only",), "completion").uid == Batch(("only",), "solver").uid


def test_transferred_cap_does_not_spend_other_finalist_reservations():
    p = BatchProblem((CheckOption("a", "c", ("a",), 13, reserved_cap=10, reservation_id="finalist-a"),
                      CheckOption("b", "c", ("b",), 10, reserved_cap=10, reservation_id="finalist-b")), {"default": 0}, 2)
    assert p.cost(("a",)) == 3
    assert not p.feasible(("a",))
    assert p.feasible(("b",))


def test_one_reservation_cannot_authorize_two_physical_jobs():
    p = BatchProblem((CheckOption("a", "c", ("a",), 10, reserved_cap=10, reservation_id="same-work"),
                      CheckOption("b", "c", ("b",), 10, reserved_cap=10, reservation_id="same-work")), {"default": 0}, 0)
    assert p.feasible(("a",))
    assert not p.feasible(("a", "b"))


def test_stale_menu_is_revalidated_against_current_budget():
    p = problem(8)
    menu = MenuBuilder().build(p, completion_ids=("normal", "duplicate"))
    assert select_batch(menu, replace(p, budget=1), lambda b: 100.) is None


def test_same_recorded_support_is_required_for_likelihood_ratio():
    probs = menu_distribution([10000., -10000.], exploration=.2)
    assert sum(probs) == pytest.approx(1.) and all(x > 0 for x in probs)
    assert menu_ratio(("a", "b"), ("a", "b"), "a", probs, probs) == 1.
    with pytest.raises(ValueError, match="support changed"):
        menu_ratio(("b", "a"), ("a", "b"), "a", probs, probs)
    with pytest.raises(ValueError):
        menu_ratio(("a", "b"), ("a", "b"), "a", (1., 1.), probs)
