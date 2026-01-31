import numpy as np

from sbt_agency.channel import build_channel_matrix, enumerate_action_seqs
from sbt_agency.kernel import FiniteKernel


def _make_kernel():
    # Action 0: identity; Action 1: swap
    P = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    return FiniteKernel(P)


def test_enumerate_action_seqs_order():
    seqs = enumerate_action_seqs([0, 1], 2)
    assert seqs == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_channel_matrix_deterministic():
    kernel = _make_kernel()
    seqs = enumerate_action_seqs([0, 1], 2)

    W = build_channel_matrix(kernel, s0=0, action_seqs=seqs, proj=lambda s: s)
    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    assert W.shape == (4, 2)
    assert np.allclose(W, expected, atol=0.0, rtol=0.0)


def test_channel_matrix_collapsed_projection():
    kernel = _make_kernel()
    seqs = enumerate_action_seqs([0, 1], 2)

    W = build_channel_matrix(kernel, s0=0, action_seqs=seqs, proj=lambda s: 0)
    assert W.shape == (4, 1)
    assert np.allclose(W, np.ones((4, 1)), atol=0.0, rtol=0.0)


def test_channel_matrix_probabilistic_start():
    kernel = _make_kernel()
    seqs = enumerate_action_seqs([0, 1], 2)
    dist0 = np.array([0.5, 0.5])

    W = build_channel_matrix(kernel, s0=dist0, action_seqs=seqs, proj=lambda s: s)
    expected = np.array([0.5, 0.5])
    assert np.allclose(W, np.tile(expected, (4, 1)), atol=0.0, rtol=0.0)

