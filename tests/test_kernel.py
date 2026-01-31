import numpy as np
import pytest

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


def test_validate_passes():
    kernel = _make_kernel()
    kernel.validate()


def test_validate_negative_fails():
    P = np.array(
        [
            [[1.0, -0.1], [0.0, 1.1]],
        ]
    )
    kernel = FiniteKernel(P)
    with pytest.raises(ValueError):
        kernel.validate()


def test_validate_row_sum_fails():
    P = np.array(
        [
            [[0.9, 0.0], [0.0, 1.0]],
        ]
    )
    kernel = FiniteKernel(P)
    with pytest.raises(ValueError):
        kernel.validate()


def test_rollout_composition():
    kernel = _make_kernel()
    dist = np.array([0.25, 0.75])
    actions = [1, 0, 1]

    rolled = kernel.rollout_dist(dist, actions)
    stepped = kernel.step_dist(kernel.step_dist(kernel.step_dist(dist, 1), 0), 1)
    assert np.allclose(rolled, stepped, atol=0.0, rtol=0.0)

