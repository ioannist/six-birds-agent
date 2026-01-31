"""Finite-state transition kernel utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FiniteKernel:
    """Dense finite-state transition kernel.

    P[a, s, s2] = Pr(S_{t+1}=s2 | S_t=s, A_t=a)
    """

    P: np.ndarray

    def __post_init__(self) -> None:
        self.P = np.asarray(self.P, dtype=float)
        if self.P.ndim != 3:
            raise ValueError("P must be a 3D array with shape (n_actions, n_states, n_states)")
        if self.P.shape[1] != self.P.shape[2]:
            raise ValueError("P must have shape (n_actions, n_states, n_states)")
        self.n_actions = int(self.P.shape[0])
        self.n_states = int(self.P.shape[1])

    def validate(self, atol: float = 1e-12) -> None:
        """Validate nonnegativity and row-stochasticity."""
        if self.P.ndim != 3:
            raise ValueError("P must be a 3D array with shape (n_actions, n_states, n_states)")
        if self.P.shape[1] != self.P.shape[2]:
            raise ValueError("P must have shape (n_actions, n_states, n_states)")
        if np.any(self.P < -atol):
            raise ValueError("P contains negative probabilities")
        row_sums = self.P.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=atol, rtol=0.0):
            raise ValueError("Rows of P must sum to 1 within tolerance")

    def to_dense(self) -> np.ndarray:
        """Return a copy of the dense transition tensor."""
        return self.P.copy()

    def delta(self, s: int) -> np.ndarray:
        """Return a one-hot distribution over states."""
        if s < 0 or s >= self.n_states:
            raise IndexError("State out of range")
        dist = np.zeros(self.n_states, dtype=float)
        dist[s] = 1.0
        return dist

    def step_dist(self, dist_s: np.ndarray, action: int) -> np.ndarray:
        """Advance a state distribution by one step under an action."""
        if action < 0 or action >= self.n_actions:
            raise IndexError("Action out of range")
        dist = np.asarray(dist_s, dtype=float)
        if dist.ndim != 1 or dist.shape[0] != self.n_states:
            raise ValueError("dist_s must be a 1D array of shape (n_states,)")
        return dist @ self.P[action]

    def rollout_dist(self, dist_s: np.ndarray, action_seq: Sequence[int]) -> np.ndarray:
        """Roll out a sequence of actions on a state distribution."""
        dist = np.asarray(dist_s, dtype=float)
        if dist.ndim != 1 or dist.shape[0] != self.n_states:
            raise ValueError("dist_s must be a 1D array of shape (n_states,)")
        for action in action_seq:
            dist = self.step_dist(dist, int(action))
        return dist

