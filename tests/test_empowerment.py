import math

import numpy as np
import pytest

from sbt_agency.empowerment import blahut_arimoto, capacity_bits, capacity_bounds


def _mutual_information(W, p):
    q = p @ W
    terms = np.zeros_like(W)
    mask = W > 0
    terms[mask] = W[mask] * np.log((W / q)[mask])
    return float(p @ terms.sum(axis=1))


def test_capacity_is_realized_by_returned_distribution():
    # A redundant noisy input must get zero weight at the optimum. The old
    # update returned a positive mass and mislabeled a stale divergence as MI.
    W = np.array([[1., 0.], [0., 1.], [0.5, 0.5]])
    result = capacity_bounds(W)
    assert abs(result.lower_nats - math.log(2)) < 1e-12
    assert result.upper_nats - result.lower_nats <= 1e-12
    assert abs(_mutual_information(W, result.input_distribution) - result.lower_nats) < 1e-14
    assert result.input_distribution[2] < 1e-11


def test_asymmetric_z_channel_capacity():
    W = np.array([[1., 0.], [0.5, 0.5]])
    C, p = blahut_arimoto(W)
    assert abs(C / math.log(2) - math.log2(1.25)) < 1e-11
    assert np.allclose(p, [0.6, 0.4], atol=1e-6, rtol=0)
    assert abs(C - _mutual_information(W, p)) < 1e-14


def test_duplicate_inputs_do_not_change_capacity():
    W = np.array([[0.95, 0.05], [0.4, 0.6]])
    repeated = W[[0, 0, 0, 1]]
    C, p = blahut_arimoto(repeated)
    assert abs(C - blahut_arimoto(W)[0]) < 1e-12
    assert abs(C - _mutual_information(repeated, p)) < 1e-14


def test_capacity_does_not_silently_accept_nonconvergence():
    with pytest.raises(RuntimeError, match="did not converge"):
        capacity_bits(np.array([[1., 0.], [0.5, 0.5]]), max_iter=1)


@pytest.mark.parametrize("W", [np.empty((0, 2)), np.empty((2, 0))])
def test_capacity_rejects_empty_alphabets(W):
    with pytest.raises(ValueError, match="nonempty"):
        capacity_bits(W)


@pytest.mark.parametrize("tol,max_iter", [(0, 10), (float("nan"), 10), (1e-12, 0)])
def test_capacity_rejects_invalid_solver_controls(tol, max_iter):
    with pytest.raises(ValueError):
        capacity_bits(np.eye(2), tol=tol, max_iter=max_iter)


def test_deterministic_capacity_log3():
    W = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    C_bits = capacity_bits(W)
    assert abs(C_bits - math.log2(3.0)) < 1e-6

    C_nats, p_opt = blahut_arimoto(W)
    assert abs(p_opt.sum() - 1.0) < 1e-12
    assert np.all(p_opt >= -1e-15)
    assert math.isfinite(C_nats)


def test_bsc_capacity_matches_formula():
    p = 0.1
    W = np.array([[1.0 - p, p], [p, 1.0 - p]])
    expected = 1.0 + p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)
    C_bits = capacity_bits(W)
    assert abs(C_bits - expected) < 1e-6

    _, p_opt = blahut_arimoto(W)
    assert np.allclose(p_opt, np.array([0.5, 0.5]), atol=1e-6)


def test_validation_rejects_invalid():
    W_neg = np.array([[1.0, -0.1], [0.1, 0.9]])
    with pytest.raises(ValueError):
        blahut_arimoto(W_neg)

    W_bad = np.array([[0.9, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        blahut_arimoto(W_bad)
