import numpy as np

from sbt_agency.kernel import FiniteKernel
from sbt_agency.packaging import empirical_endomap, idempotence_defect


def _make_kernel():
    P = np.zeros((1, 4, 4))
    P[0, 0, 2] = 1.0
    P[0, 1, 2] = 1.0
    P[0, 2, 0] = 1.0
    P[0, 3, 0] = 1.0
    return FiniteKernel(P)


def _proj(s: int) -> int:
    return 0 if s in (0, 1) else 1


def _policy(_s: int):
    return 0


def test_empirical_endomap_tau1_defect1():
    kernel = _make_kernel()
    E = empirical_endomap(kernel, _proj, tau=1, policy=_policy)
    assert set(E.keys()) == {0, 1}
    assert set(E.values()) == {0, 1}
    assert E[0] == 1
    assert E[1] == 0
    assert idempotence_defect(E) == 1.0


def test_empirical_endomap_tau2_defect0():
    kernel = _make_kernel()
    E = empirical_endomap(kernel, _proj, tau=2, policy=_policy)
    assert set(E.keys()) == {0, 1}
    assert set(E.values()) == {0, 1}
    assert E[0] == 0
    assert E[1] == 1
    assert idempotence_defect(E) == 0.0

