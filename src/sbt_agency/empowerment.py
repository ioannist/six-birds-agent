"""Channel capacity utilities via Blahut-Arimoto."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


def _validate_channel_matrix(W: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array with shape (n_inputs, n_outputs)")
    if 0 in W.shape:
        raise ValueError("a channel must have nonempty input and output alphabets")
    if not np.all(np.isfinite(W)):
        raise ValueError("W must contain only finite values")
    if np.any(W < 0.0):
        raise ValueError("W must have nonnegative entries")
    row_sums = W.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol, rtol=0.0):
        raise ValueError("Rows of W must sum to 1 within tolerance")
    # Explicitly interpret tolerated row-sum roundoff as a normalized law.
    return W / row_sums[:, None]


@dataclass(frozen=True)
class CapacityResult:
    """Numerical capacity bracket in nats, without interval-arithmetic guarantees."""

    lower_nats: float
    upper_nats: float
    input_distribution: np.ndarray
    iterations: int


def capacity_bounds(
    W: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000
) -> CapacityResult:
    """Run Blahut--Arimoto until max_i D(W_i || pW) - I(p;W) <= tol.

    The lower bound is the mutual information of the returned distribution;
    the upper bound follows from I(r;W) = sum_i r_i D(W_i||q) - D(rW||q).
    All bounds use floating arithmetic. Failure to converge raises rather than
    silently presenting an unconverged iterate as the capacity.
    """
    W = _validate_channel_matrix(W)
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    # Duplicate channel rows add no distinguishable outputs. Removing them
    # also makes convergence insensitive to repeated action-sequence labels.
    rows, inverse, counts = np.unique(W, axis=0, return_inverse=True, return_counts=True)
    log_p = np.full(len(rows), -math.log(len(rows)))
    log_W = np.full_like(rows, -math.inf)
    np.log(rows, out=log_W, where=rows > 0)
    w_log_w = np.zeros_like(rows)
    np.multiply(rows, log_W, out=w_log_w, where=rows > 0)

    for iteration in range(1, max_iter + 1):
        # Log-space mixture preserves positive support even for tiny p_i.
        log_q = np.logaddexp.reduce(log_p[:, None] + log_W, axis=0)
        w_log_q = np.zeros_like(rows)
        np.multiply(rows, log_q, out=w_log_q, where=rows > 0)
        D = (w_log_w - w_log_q).sum(axis=1)
        p = np.exp(log_p)
        lower = max(0.0, float(p @ D))
        upper = max(lower, float(D.max()))
        if upper - lower <= tol:
            original_p = p[inverse] / counts[inverse]
            return CapacityResult(lower, upper, original_p, iteration)
        # The multiplicative p_i factor is essential.
        log_p = log_p + D
        log_p -= np.logaddexp.reduce(log_p)

    raise RuntimeError(
        f"Blahut-Arimoto did not converge in {max_iter} iterations: "
        f"capacity gap {upper - lower:.6g} nats exceeds tol={tol}"
    )


def blahut_arimoto(
    W: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000
) -> Tuple[float, np.ndarray]:
    """Return a capacity approximation in nats and its realizing input law."""
    result = capacity_bounds(W, tol=tol, max_iter=max_iter)
    return result.lower_nats, result.input_distribution


def capacity_bits(W: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000) -> float:
    """Approximate capacity in bits with numerical gap <= tol / log(2)."""
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
    """Initial-budget open-loop capacity; not a safety-preserving policy channel.

    Return zero by convention when the feasible input alphabet is empty.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array with shape (n_inputs, n_outputs)")
    if len(seqs) != W.shape[0]:
        raise ValueError("seqs length must match number of rows in W")
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")

    feasible_idx = []
    for i, seq in enumerate(seqs):
        total_cost = 0.0
        for action in seq:
            cost = float(cost_fn(int(action)))
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("action cost must be finite and non-negative")
            total_cost += cost
        if total_cost <= budget:
            feasible_idx.append(i)

    if not feasible_idx:
        return 0.0

    Wf = W[feasible_idx, :]
    return capacity_bits(Wf, tol=tol, max_iter=max_iter)
