"""Channel-matrix builders for action sequences and state projections."""

from __future__ import annotations

from itertools import product
from typing import Callable, Sequence

import numpy as np

from sbt_agency.kernel import FiniteKernel


def enumerate_action_seqs(actions: Sequence[int], H: int) -> list[tuple[int, ...]]:
    """Return all length-H action sequences over the provided actions list."""
    if H < 0:
        raise ValueError("H must be non-negative")
    return list(product(actions, repeat=H))


def _validate_dist(dist: np.ndarray, n_states: int, atol: float = 1e-12) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    if dist.ndim != 1 or dist.shape[0] != n_states:
        raise ValueError("s0 distribution must be a 1D array of shape (n_states,)")
    if np.any(dist < -atol):
        raise ValueError("s0 distribution has negative entries")
    if not np.isclose(dist.sum(), 1.0, atol=atol, rtol=0.0):
        raise ValueError("s0 distribution must sum to 1 within tolerance")
    return dist


def build_channel_matrix(
    kernel: FiniteKernel,
    s0: int | np.ndarray,
    action_seqs: Sequence[Sequence[int]],
    proj: Callable[[int], int],
) -> np.ndarray:
    """Build a channel matrix over projected outputs for action sequences."""
    n_states = kernel.n_states

    out_idx = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        y = int(proj(s))
        if y < 0:
            raise ValueError("proj must return non-negative output indices")
        out_idx[s] = y
    n_outputs = int(out_idx.max()) + 1

    if isinstance(s0, (int, np.integer)):
        s0_int = int(s0)
        if s0_int < 0 or s0_int >= n_states:
            raise IndexError("s0 state out of range")
        dist0 = kernel.delta(s0_int)
    else:
        dist0 = _validate_dist(np.asarray(s0, dtype=float), n_states)

    W = np.zeros((len(action_seqs), n_outputs), dtype=float)
    for i, seq in enumerate(action_seqs):
        dist = kernel.rollout_dist(dist0, seq)
        W[i] = np.bincount(out_idx, weights=dist, minlength=n_outputs)

    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Rows of W must sum to 1 within tolerance")

    return W

