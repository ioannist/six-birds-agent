"""Finite-state viability kernel utilities."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Sequence
import math
from typing import Callable

import numpy as np

from sbt_agency.kernel import FiniteKernel


def ledger_feasible_actions(
    actions: Sequence[Hashable],
    ledger: Callable[[Hashable], float],
    cost: Callable[[Hashable], float],
    eps: float = 0.0,
) -> Callable[[Hashable], list]:
    """Return ledger-gated actions; positive eps explicitly relaxes the gate."""
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative")

    def feasible_actions(s: Hashable) -> list:
        available = ledger(s)
        if not math.isfinite(available) or available < 0:
            raise ValueError("ledger value must be finite and non-negative")
        allowed: list = []
        for a in actions:
            c = float(cost(a))
            if not math.isfinite(c) or c < 0:
                raise ValueError("action cost must be finite and non-negative")
            if c <= available + eps:
                allowed.append(a)
        return allowed

    return feasible_actions


def post_support_from_kernel(
    kernel: FiniteKernel, atol: float = 0.0
) -> Callable[[int, int], set[int]]:
    """Positive support. Positive atol instead models a truncated support law."""
    kernel.validate()
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    support: list[list[set[int]]] = []
    for a in range(kernel.n_actions):
        action_support: list[set[int]] = []
        for s in range(kernel.n_states):
            succ = set(np.where(kernel.P[a, s] > atol)[0])
            if not succ:
                raise ValueError("support truncation removed all successors")
            action_support.append(succ)
        support.append(action_support)

    def post_support(s: int, a: int) -> set[int]:
        return support[a][s]

    return post_support


def viability_kernel(
    states: Sequence[Hashable],
    actions: Sequence[Hashable],
    feasible_actions: Callable[[Hashable], Iterable[Hashable]],
    post_support: Callable[[Hashable, Hashable], set],
    safe: Callable[[Hashable], bool],
) -> set:
    """Greatest safe controlled-invariant set for fixed callbacks.

    Callbacks must depend only on their arguments, not on the iteration index.
    Successor sets must be nonempty, as for a stochastic kernel.
    """
    for K in _viability_iterates(states, actions, feasible_actions, post_support, safe):
        pass
    return K


def viability_kernel_history(
    states: Sequence[Hashable],
    actions: Sequence[Hashable],
    feasible_actions: Callable[[Hashable], Iterable[Hashable]],
    post_support: Callable[[Hashable, Hashable], set],
    safe: Callable[[Hashable], bool],
) -> list[set]:
    """Return the descending sequence of kernel iterates, including the fixed point."""
    return list(_viability_iterates(states, actions, feasible_actions, post_support, safe))


def _viability_iterates(
    states: Sequence[Hashable],
    actions: Sequence[Hashable],
    feasible_actions: Callable[[Hashable], Iterable[Hashable]],
    post_support: Callable[[Hashable, Hashable], set],
    safe: Callable[[Hashable], bool],
) -> Iterator[set]:
    K = {s for s in states if safe(s)}
    yield K
    action_set = set(actions)

    while True:
        next_K = set()
        for s in K:
            ok = False
            for a in feasible_actions(s):
                if a not in action_set:
                    raise ValueError("feasible action is outside the action alphabet")
                succ = post_support(s, a)
                if not succ:
                    raise ValueError("post_support must be nonempty for stochastic viability")
                if succ.issubset(K):
                    ok = True
                    break
            if ok:
                next_K.add(s)
        yield next_K
        if next_K == K:
            return
        K = next_K
