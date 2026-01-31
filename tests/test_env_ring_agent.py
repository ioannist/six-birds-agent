import numpy as np

from sbt_agency.env_ring_agent import RingAgentConfig, build_kernel


def test_build_kernel_validates():
    config = RingAgentConfig()
    kernel, _projections, _metadata = build_kernel(config)
    kernel.validate()


def test_identity_invariant():
    config = RingAgentConfig(identity_on=True, g_size=2)
    kernel, _projections, metadata = build_kernel(config)
    tuples = metadata["state_tuples"]

    for a in range(kernel.n_actions):
        for s in range(kernel.n_states):
            g = tuples[s][4]
            succ = np.where(kernel.P[a, s] > 0.0)[0]
            for s2 in succ:
                g2 = tuples[s2][4]
                assert g2 == g


def test_perfect_repair_sets_u_zero():
    config = RingAgentConfig(enable_repair=True, p_repair=1.0)
    kernel, _projections, metadata = build_kernel(config)
    tuples = metadata["state_tuples"]
    action_names = metadata["action_names"]
    repair_idx = action_names.index("REPAIR")

    for s in range(kernel.n_states):
        y, u, phi, r, g, theta = tuples[s]
        if u != 1:
            continue
        if r < config.cost_repair:
            continue
        succ = np.where(kernel.P[repair_idx, s] > 0.0)[0]
        for s2 in succ:
            u2 = tuples[s2][1]
            assert u2 == 0

