import math

import numpy as np
import pytest

from sbt_agency.empowerment import blahut_arimoto, capacity_bits


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

