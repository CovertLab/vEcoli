"""
Unit tests for the cvxpy solver fallback in
`ecoli.processes.metabolism_redux.solve_with_fallback`.

These use a trivial standalone cvxpy LP (not the full MetabolismRedux
network flow model) so they run fast and don't require a whole-cell
simulation. They verify that when the primary solver (standing in for
GLOP) raises `cvxpy.error.SolverError`, the solve transparently falls
back to another solver and still returns an optimal solution, without
altering the problem's objective or constraints. See
`ecoli/processes/metabolism_redux.py` for the production usage inside
`NetworkFlowModel.solve`.
"""

import cvxpy as cp
import pytest
from cvxpy.error import SolverError

from ecoli.processes.metabolism_redux import solve_with_fallback


def test_solve_with_fallback_recovers_from_primary_solver_error(monkeypatch):
    """A SolverError from the primary solver should trigger a retry with a
    fallback solver, and the fallback should still find the correct
    optimum (i.e. the problem itself is untouched)."""
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.abs(x - 3)), [x >= 0, x <= 10])

    real_solve = cp.Problem.solve
    calls = []

    def fake_solve(self, solver=None, **kwargs):
        calls.append(solver)
        if solver == "FAKE_PRIMARY":
            raise SolverError("simulated GLOP failure")
        return real_solve(self, solver=solver, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)

    solve_with_fallback(
        problem,
        primary_solver="FAKE_PRIMARY",
        fallback_solvers=(cp.CLARABEL,),
    )

    assert problem.status == "optimal"
    assert x.value is not None
    assert abs(x.value - 3) < 1e-4
    # Primary was tried first and failed; fallback was then used.
    assert calls == ["FAKE_PRIMARY", cp.CLARABEL]


def test_solve_with_fallback_tries_multiple_fallbacks_in_order(monkeypatch):
    """If the first fallback also errors, the next fallback in the list
    should be attempted before giving up."""
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.abs(x - 3)), [x >= 0, x <= 10])

    real_solve = cp.Problem.solve
    calls = []

    def fake_solve(self, solver=None, **kwargs):
        calls.append(solver)
        if solver in ("FAKE_PRIMARY", "FAKE_FALLBACK_1"):
            raise SolverError(f"simulated failure for {solver}")
        return real_solve(self, solver=solver, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)

    solve_with_fallback(
        problem,
        primary_solver="FAKE_PRIMARY",
        fallback_solvers=("FAKE_FALLBACK_1", cp.CLARABEL),
    )

    assert problem.status == "optimal"
    assert calls == ["FAKE_PRIMARY", "FAKE_FALLBACK_1", cp.CLARABEL]


def test_solve_with_fallback_raises_with_context_when_all_solvers_fail(monkeypatch):
    """If every solver (primary + all fallbacks) fails, the original error
    should be re-raised (wrapped) with context about what was tried,
    instead of silently swallowing the failure."""
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.abs(x)), [x >= 0])

    def always_fail(self, solver=None, **kwargs):
        raise SolverError(f"simulated failure for {solver}")

    monkeypatch.setattr(cp.Problem, "solve", always_fail)

    with pytest.raises(ValueError, match="did not converge"):
        solve_with_fallback(
            problem,
            primary_solver="FAKE_PRIMARY",
            fallback_solvers=("FAKE_FALLBACK",),
        )
