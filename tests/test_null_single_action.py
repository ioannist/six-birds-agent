import numpy as np

from sbt_agency.channel import build_channel_matrix, enumerate_action_seqs
from sbt_agency.empowerment import capacity_bits
from sbt_agency.kernel import FiniteKernel


def test_single_action_capacity_zero():
    P = np.zeros((1, 3, 3))
    P[0, 0, 1] = 1.0
    P[0, 1, 2] = 1.0
    P[0, 2, 0] = 1.0
    kernel = FiniteKernel(P)

    s0 = 0
    proj = lambda s: s
    for H in (1, 2, 3):
        seqs = enumerate_action_seqs([0], H)
        W = build_channel_matrix(kernel, s0, seqs, proj)
        cap = capacity_bits(W)
        assert cap <= 1e-9

