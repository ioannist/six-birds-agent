import numpy as np

from sbt_agency.channel import build_channel_matrix, enumerate_action_seqs
from sbt_agency.empowerment import feasible_capacity_bits
from sbt_agency.kernel import FiniteKernel


def _make_kernel():
    P = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    return FiniteKernel(P)


def test_feasible_capacity_monotonic():
    kernel = _make_kernel()
    seqs = enumerate_action_seqs([0, 1], 2)
    W = build_channel_matrix(kernel, s0=0, action_seqs=seqs, proj=lambda s: s)

    def cost_fn(a: int) -> float:
        return 0.0 if a == 0 else 1.0

    c0 = feasible_capacity_bits(W, seqs, cost_fn, budget=0.0)
    c1 = feasible_capacity_bits(W, seqs, cost_fn, budget=1.0)
    c2 = feasible_capacity_bits(W, seqs, cost_fn, budget=2.0)

    assert c0 <= c1 + 1e-12
    assert c1 <= c2 + 1e-12
    assert abs(c0 - 0.0) < 1e-9
    assert abs(c1 - 1.0) < 1e-6
    assert abs(c2 - 1.0) < 1e-6


def test_feasible_capacity_empty_returns_zero():
    kernel = _make_kernel()
    seqs = enumerate_action_seqs([0, 1], 2)
    W = build_channel_matrix(kernel, s0=0, action_seqs=seqs, proj=lambda s: s)

    def cost_fn(a: int) -> float:
        return 1.0

    c0 = feasible_capacity_bits(W, seqs, cost_fn, budget=0.0)
    assert c0 == 0.0

