"""Channel capacity utilities via Blahut-Arimoto."""

from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple

import numpy as np


def _validate_channel_matrix(W: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array with shape (n_inputs, n_outputs)")
    if not np.all(np.isfinite(W)):
        raise ValueError("W must contain only finite values")
    if np.any(W < 0.0):
        raise ValueError("W must have nonnegative entries")
    row_sums = W.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol, rtol=0.0):
        raise ValueError("Rows of W must sum to 1 within tolerance")
    return W


def blahut_arimoto(
    W: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000
) -> Tuple[float, np.ndarray]:
    """Compute channel capacity in nats and the optimal input distribution."""
    W = _validate_channel_matrix(W)
    n_inputs = W.shape[0]

    p = np.full(n_inputs, 1.0 / n_inputs, dtype=float)
    prev_C = -math.inf

    for _ in range(max_iter):
        q = p @ W
        # D_i = sum_y W[i,y] * log(W[i,y] / q_y), skipping zero terms
        D = np.zeros(n_inputs, dtype=float)
        for i in range(n_inputs):
            w = W[i]
            mask = w > 0.0
            if np.any(mask):
                q_mask = q[mask]
                w_mask = w[mask]
                safe = q_mask > 0.0
                if np.any(safe):
                    D[i] = np.sum(w_mask[safe] * (np.log(w_mask[safe]) - np.log(q_mask[safe])))
                else:
                    D[i] = 0.0
            else:
                D[i] = 0.0

        shift = D.max() if n_inputs > 0 else 0.0
        exp_D = np.exp(D - shift)
        p_new = exp_D / exp_D.sum()

        C = float(np.dot(p_new, D))
        if abs(C - prev_C) < tol:
            p = p_new
            break
        p = p_new
        prev_C = C

    return float(C), p


def capacity_bits(W: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000) -> float:
    """Compute channel capacity in bits."""
    C_nats, _ = blahut_arimoto(W, tol=tol, max_iter=max_iter)
    return C_nats / math.log(2.0)


def feasible_capacity_bits(
    W: np.ndarray,
    seqs: Sequence[Sequence[int]],
    cost_fn: Callable[[int], float],
    budget: float,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> float:
    """Compute capacity over sequences whose total cost is within budget."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array with shape (n_inputs, n_outputs)")
    if len(seqs) != W.shape[0]:
        raise ValueError("seqs length must match number of rows in W")

    feasible_idx = []
    for i, seq in enumerate(seqs):
        total_cost = 0.0
        for action in seq:
            total_cost += float(cost_fn(int(action)))
        if total_cost <= budget + 1e-12:
            feasible_idx.append(i)

    if not feasible_idx:
        return 0.0

    Wf = W[feasible_idx, :]
    return capacity_bits(Wf, tol=tol, max_iter=max_iter)
