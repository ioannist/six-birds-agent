import numpy as np

from sbt_agency.kernel import FiniteKernel
from sbt_agency.viability import (
    ledger_feasible_actions,
    post_support_from_kernel,
    viability_kernel,
)


def _make_kernel_no_income():
    P = np.zeros((2, 3, 3))
    for a in range(2):
        for r in range(3):
            nxt = max(0, r - 1)
            P[a, r, nxt] = 1.0
    return FiniteKernel(P)


def _make_kernel_income():
    P = np.zeros((2, 3, 3))
    for r in range(3):
        P[0, r, max(0, r - 1)] = 1.0
        P[1, r, r] = 1.0
    return FiniteKernel(P)


def test_ledger_viability_income_vs_no_income():
    states = [0, 1, 2]
    actions = [0, 1]

    def safe(s: int) -> bool:
        return s >= 1

    feasible_actions = ledger_feasible_actions(
        actions, ledger=lambda s: float(s), cost=lambda _a: 1.0
    )

    assert feasible_actions(0) == []
    assert feasible_actions(1) == [0, 1]

    kernel_no_income = _make_kernel_no_income()
    post_support_no_income = post_support_from_kernel(kernel_no_income)
    K_no_income = viability_kernel(
        states, actions, feasible_actions, post_support_no_income, safe
    )
    assert K_no_income == set()

    kernel_income = _make_kernel_income()
    post_support_income = post_support_from_kernel(kernel_income)
    K_income = viability_kernel(states, actions, feasible_actions, post_support_income, safe)
    assert K_income == {1, 2}

