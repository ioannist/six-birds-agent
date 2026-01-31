import math

from sbt_agency.env_ring_agent import RingAgentConfig
from sbt_agency.metrics import compute_ring_metrics


def test_metrics_smoke():
    config = RingAgentConfig(
        L=6,
        m_phase=2,
        R_max=1,
        g_size=1,
        theta_max=0,
        enable_protocol=True,
        enable_repair=True,
        enable_learn=False,
        identity_on=True,
        p_flip=0.1,
        p_slip=0.1,
        p_repair=1.0,
        cost_left=1,
        cost_right=1,
        cost_repair=0,
        cost_learn=0,
    )
    metrics = compute_ring_metrics(config, empowerment_max_states=8, seed=0)

    assert 0.0 <= metrics["idempotence_defect"] <= 1.0
    assert 0.0 <= metrics["empowerment_median_on_K"] <= math.log2(config.L) + 1e-6
    assert 0 <= metrics["kernel_size_viable"] <= metrics["n_states"]

