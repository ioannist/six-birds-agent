"""Empirical packaging endomap and idempotence defect utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np

from sbt_agency.kernel import FiniteKernel


def _validate_policy_output(
    out: int | np.ndarray, n_actions: int, atol: float = 1e-12
) -> np.ndarray:
    if isinstance(out, (int, np.integer)):
        a = int(out)
        if a < 0 or a >= n_actions:
            raise ValueError("policy action out of range")
        probs = np.zeros(n_actions, dtype=float)
        probs[a] = 1.0
        return probs

    probs = np.asarray(out, dtype=float)
    if probs.ndim != 1 or probs.shape[0] != n_actions:
        raise ValueError("policy distribution has wrong shape")
    if not np.all(np.isfinite(probs)):
        raise ValueError("policy distribution must be finite")
    if np.any(probs < -atol):
        raise ValueError("policy distribution has negative entries")
    if not np.isclose(probs.sum(), 1.0, atol=atol, rtol=0.0):
        raise ValueError("policy distribution must sum to 1 within tolerance")
    return probs


def empirical_endomap(
    kernel: FiniteKernel,
    proj: Callable[[int], int],
    tau: int,
    policy: Callable[[int], int | np.ndarray],
    *,
    macro_labels: Sequence[int] | None = None,
) -> dict[int, int]:
    """Compute an empirical endomap on macro labels under a stationary policy."""
    if tau < 0:
        raise ValueError("tau must be non-negative")

    n_states = kernel.n_states
    n_actions = kernel.n_actions

    labels = []
    state_labels = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        x = proj(s)
        if not isinstance(x, (int, np.integer)):
            raise ValueError("proj(s) must return an integer")
        x = int(x)
        if x < 0:
            raise ValueError("proj(s) must be non-negative")
        state_labels[s] = x
        labels.append(x)

    if macro_labels is None:
        label_list = sorted(set(labels))
    else:
        label_list = sorted(int(x) for x in macro_labels)

    label_to_states: dict[int, list[int]] = {x: [] for x in label_list}
    for s, x in enumerate(state_labels.tolist()):
        if x in label_to_states:
            label_to_states[x].append(s)

    for x, states in label_to_states.items():
        if not states:
            raise ValueError(f"macro label {x} has no supporting states")

    T_pi = np.zeros((n_states, n_states), dtype=float)
    for s in range(n_states):
        probs = _validate_policy_output(policy(s), n_actions)
        T_pi[s] = probs @ kernel.P[:, s, :]
    if not np.allclose(T_pi.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("rows of T_pi must sum to 1 within tolerance")

    E: dict[int, int] = {}
    for x in label_list:
        states = label_to_states[x]
        d = np.zeros(n_states, dtype=float)
        d[states] = 1.0 / len(states)
        for _ in range(tau):
            d = d @ T_pi

        macro_dist = np.array([d[label_to_states[l]].sum() for l in label_list], dtype=float)
        idx = int(np.argmax(macro_dist))
        E[x] = label_list[idx]

    return E


def idempotence_defect(E: Mapping[int, int]) -> float:
    """Fraction of labels where E(E(x)) != E(x)."""
    keys = list(E.keys())
    if not keys:
        return 0.0
    count = 0
    for x in keys:
        y = E[x]
        if y not in E:
            raise KeyError(f"E[{x}]={y} not in domain")
        if E[y] != y:
            count += 1
    return count / len(keys)

