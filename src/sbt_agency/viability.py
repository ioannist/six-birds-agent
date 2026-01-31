"""Finite-state viability kernel utilities."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
import math
from typing import Callable

import numpy as np

from sbt_agency.kernel import FiniteKernel


def ledger_feasible_actions(
    actions: Sequence[Hashable],
    ledger: Callable[[Hashable], float],
    cost: Callable[[Hashable], float],
    eps: float = 1e-12,
) -> Callable[[Hashable], list]:
    """Return a feasible-actions function gated by a ledger value."""
    if eps < 0:
        raise ValueError("eps must be non-negative")

    def feasible_actions(s: Hashable) -> list:
        available = ledger(s)
        if not math.isfinite(available):
            raise ValueError("ledger value must be finite")
        allowed: list = []
        for a in actions:
            c = float(cost(a))
            if c < 0:
                raise ValueError("action cost must be non-negative")
            if c <= available + eps:
                allowed.append(a)
        return allowed

    return feasible_actions


def post_support_from_kernel(
    kernel: FiniteKernel, atol: float = 0.0
) -> Callable[[int, int], set[int]]:
    """Build a post_support function from a FiniteKernel transition tensor."""
    if atol < 0:
        raise ValueError("atol must be non-negative")
    support: list[list[set[int]]] = []
    for a in range(kernel.n_actions):
        action_support: list[set[int]] = []
        for s in range(kernel.n_states):
            succ = set(np.where(kernel.P[a, s] > atol)[0])
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
    """Compute the viability kernel as the greatest fixed point."""
    K = {s for s in states if safe(s)}

    while True:
        next_K = set()
        for s in K:
            ok = False
            for a in feasible_actions(s):
                succ = post_support(s, a)
                if succ.issubset(K):
                    ok = True
                    break
            if ok:
                next_K.add(s)
        if next_K == K:
            return next_K
        K = next_K


def viability_kernel_history(
    states: Sequence[Hashable],
    actions: Sequence[Hashable],
    feasible_actions: Callable[[Hashable], Iterable[Hashable]],
    post_support: Callable[[Hashable, Hashable], set],
    safe: Callable[[Hashable], bool],
) -> list[set]:
    """Return the descending sequence of kernel iterates, including the fixed point."""
    K = {s for s in states if safe(s)}
    history = [set(K)]

    while True:
        next_K = set()
        for s in K:
            ok = False
            for a in feasible_actions(s):
                succ = post_support(s, a)
                if succ.issubset(K):
                    ok = True
                    break
            if ok:
                next_K.add(s)
        history.append(set(next_K))
        if next_K == K:
            return history
        K = next_K
