"""Check the PDF's solver contract independently of the optimizer's claims."""
import random
from types import SimpleNamespace

import numpy as np
import pytest

from specrsi.planning.completion import CompletionProblem, VerificationJob, solve_completion


@pytest.mark.parametrize("vector", [
    [2., 0., 1.], [1., -1., 1.], [1., 0., 2.], [1., 0., .5],
    [1., 0., np.nan], [1., 0., 1., 0.], [[1., 0., 1.]],
])
def test_solver_incumbent_requires_binary_values_and_exact_dimension(monkeypatch, vector):
    scipy = pytest.importorskip("scipy.optimize")
    monkeypatch.setattr(scipy, "milp", lambda *args, **kwargs:
                        SimpleNamespace(x=np.asarray(vector), status=0, message="claimed optimal"))
    problem = CompletionProblem((VerificationJob("required", (("c", "one"),), 1),
                                 VerificationJob("optional", (("c", "two"),), 1)),
                                {"execution": 0}, 10, 10)
    assert solve_completion(problem, {("c", "one")}) is None


def test_feasible_timeout_incumbent_is_preserved_without_optimality_claim(monkeypatch):
    scipy = pytest.importorskip("scipy.optimize")
    monkeypatch.setattr(scipy, "milp", lambda *args, **kwargs:
                        SimpleNamespace(x=np.array([1., 1.]), status=1, message="time limit"))
    problem = CompletionProblem((VerificationJob("required", (("c", "one"),), 2),),
                                {"execution": 1}, 3, 1)
    solution = solve_completion(problem, {("c", "one")})
    assert solution.job_ids == ("required",)
    assert solution.cost_cap == 3
    assert not solution.optimal


def test_sparse_milp_matches_exhaustive_covers_with_shared_setup_and_constraints():
    pytest.importorskip("scipy.optimize")
    rng = random.Random(2417)
    keys = [("first", "one"), ("first", "two"), ("second", "one")]
    for _ in range(30):
        jobs = tuple(VerificationJob(str(i), tuple(rng.sample(keys, rng.randint(1, 3))),
                                     rng.randint(0, 6), attempts=rng.randint(1, 3), group=str(i % 2))
                     for i in range(6))
        dependencies = tuple((str(i), str(i - 1)) for i in range(1, 6) if rng.random() < .3)
        conflicts = tuple((str(i), str(i - 2)) for i in range(2, 6) if rng.random() < .2)
        problem = CompletionProblem(jobs, {"0": 2, "1": 3}, rng.randint(3, 18), rng.randint(1, 12),
                                    dependencies, conflicts)
        target = rng.sample(keys, rng.randint(1, 3))
        exact = solve_completion(problem, target, backend="exhaustive")
        actual = solve_completion(problem, target)
        assert (actual is None) == (exact is None)
        if actual is not None:
            assert actual.optimal
            assert actual.cost_cap == exact.cost_cap
            assert problem.feasible(actual.job_ids, target)


def test_large_direct_cover_keeps_all_required_keys_with_sparse_solver_input(monkeypatch):
    scipy = pytest.importorskip("scipy.optimize")
    from scipy.sparse import issparse
    original = scipy.milp
    seen = []

    def checked(*args, **kwargs):
        matrix = kwargs["constraints"].A
        assert issparse(matrix)
        seen.append((matrix.shape, matrix.nnz))
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy, "milp", checked)
    keys = tuple(("candidate", str(i)) for i in range(500))
    jobs = tuple(VerificationJob(str(i), (key,), 1) for i, key in enumerate(keys))
    problem = CompletionProblem(jobs, {"execution": 7}, 507, 500)
    solution = solve_completion(problem, keys)
    assert solution is not None and solution.optimal
    assert len(solution.job_ids) == 500 and solution.cost_cap == 507
    assert problem.coverage(solution.job_ids) == set(keys)
    assert seen[0][1] < 10 * len(jobs)


def test_operational_prerequisite_cannot_promise_task_evidence():
    # Otherwise Eq. (14) could "cover" a required key with a job the executor
    # forbids from admitting outcomes, producing a misleading completion seed.
    with pytest.raises(ValueError, match="prerequisites cannot request task evidence"):
        VerificationJob("setup", (("candidate", "item"),), 1, kind="prerequisite")


def test_cached_job_index_cannot_mutate_the_frozen_feasibility_contract():
    problem = CompletionProblem((VerificationJob("check", (("candidate", "item"),), 2),),
                                {"execution": 1}, 3, 1)
    assert problem.by_id is problem.by_id
    with pytest.raises(TypeError):
        problem.by_id["check"] = VerificationJob("check", (("candidate", "item"),), 0)
    assert problem.cost(("check",)) == 3
    assert problem.feasible(("check",), (("candidate", "item"),))
