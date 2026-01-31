import numpy as np

from sbt_agency.kernel import FiniteKernel
from sbt_agency.sim import rollout


def test_rollout_deterministic_kernel():
    P = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    kernel = FiniteKernel(P)
    state_tuples = [(0,), (1,)]

    def pi(_state, _t):
        return 1

    traj = rollout(kernel, s0=0, n_steps=3, pi=pi, state_tuples=state_tuples)
    assert len(traj) == 3
    assert traj[-1]["s_next"] == 1

