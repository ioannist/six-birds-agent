"""Simulation utilities for finite-state kernels."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from sbt_agency.kernel import FiniteKernel


def sample_next_state(kernel: FiniteKernel, s: int, a: int, rng: np.random.Generator) -> int:
    """Sample next state from the kernel."""
    if s < 0 or s >= kernel.n_states:
        raise IndexError("state out of range")
    if a < 0 or a >= kernel.n_actions:
        raise IndexError("action out of range")
    probs = kernel.P[a, s]
    return int(rng.choice(kernel.n_states, p=probs))


def rollout(
    kernel: FiniteKernel,
    s0: int,
    n_steps: int,
    pi: Callable[[tuple, int], int],
    *,
    state_tuples: Sequence[tuple],
    action_names: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Roll out a trajectory and return per-step records."""
    if rng is None:
        rng = np.random.default_rng(0)
    if s0 < 0 or s0 >= kernel.n_states:
        raise ValueError("s0 out of range")

    traj: list[dict] = []
    s = int(s0)
    for t in range(n_steps):
        state_tuple = state_tuples[s]
        a = int(pi(state_tuple, t))
        s_next = sample_next_state(kernel, s, a, rng)
        state_next = state_tuples[s_next]
        record = {
            "t": t,
            "s": s,
            "state": state_tuple,
            "a": a,
            "s_next": s_next,
            "state_next": state_next,
        }
        if action_names is not None:
            record["a_name"] = action_names[a]
        traj.append(record)
        s = s_next
    return traj

