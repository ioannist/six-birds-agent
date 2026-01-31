from dataclasses import asdict

import numpy as np

from sbt_agency.channel import build_channel_matrix, enumerate_action_seqs
from sbt_agency.empowerment import capacity_bits
from sbt_agency.env_ring_agent import build_kernel
from sbt_agency.exp_configs import (
    ablations_suite,
    cfg_packaging_ring_off,
    cfg_packaging_ring_on,
)
from sbt_agency.metrics import compute_ring_metrics
from sbt_agency.packaging import empirical_endomap, idempotence_defect
from sbt_agency.kernel import FiniteKernel


def test_invariant_packaging_repair_tau2():
    cfg_off = cfg_packaging_ring_off()
    cfg_on = cfg_packaging_ring_on()

    kernel_off, projections_off, metadata_off = build_kernel(cfg_off)
    kernel_on, projections_on, metadata_on = build_kernel(cfg_on)

    proj_macro_off = projections_off["proj_macro"]
    proj_macro_on = projections_on["proj_macro"]

    right_idx_off = metadata_off["action_names"].index("RIGHT")
    right_idx_on = metadata_on["action_names"].index("RIGHT")
    repair_idx_on = metadata_on["action_names"].index("REPAIR")

    def policy_off(_s_idx: int) -> int:
        return right_idx_off

    def policy_on(s_idx: int) -> int:
        u = metadata_on["state_tuples"][s_idx][1]
        if u == 1:
            return repair_idx_on
        return right_idx_on

    E_off = empirical_endomap(kernel_off, proj_macro_off, tau=2, policy=policy_off)
    E_on = empirical_endomap(kernel_on, proj_macro_on, tau=2, policy=policy_on)
    defect_off = idempotence_defect(E_off)
    defect_on = idempotence_defect(E_on)

    assert defect_off >= 0.9
    assert defect_on <= 0.1


def test_invariant_null_single_action_capacity():
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


def test_invariant_schedule_trap_gap():
    P_right = np.zeros((2, 4, 4))
    for a in range(2):
        for x in (0, 1):
            for s_ext in (0, 1):
                idx = 2 * x + s_ext
                x_next = s_ext
                s_next = 1 - s_ext
                idx_next = 2 * x_next + s_next
                P_right[a, idx, idx_next] = 1.0
    kernel_right = FiniteKernel(P_right)
    d0 = np.array([0.5, 0.5, 0.0, 0.0])
    proj_x = lambda idx: idx // 2
    seqs = enumerate_action_seqs([0, 1], 1)
    W_right = build_channel_matrix(kernel_right, d0, seqs, proj_x)
    right_emp = capacity_bits(W_right)

    P_wrong = np.zeros((2, 2, 2))
    for x in (0, 1):
        P_wrong[0, x, 0] = 1.0
        P_wrong[1, x, 1] = 1.0
    kernel_wrong = FiniteKernel(P_wrong)
    s0 = 0
    proj = lambda s: s
    W_wrong = build_channel_matrix(kernel_wrong, s0, seqs, proj)
    wrong_emp = capacity_bits(W_wrong)

    assert wrong_emp > 0.5
    assert right_emp <= 1e-9


def test_invariant_protocol_increases_empowerment():
    suite = ablations_suite()
    cfg_full = suite["full"]
    cfg_np = suite["no_protocol"]

    m_full = compute_ring_metrics(
        cfg_full,
        safe_r_min=1,
        empowerment_H=2,
        empowerment_max_states=32,
        packaging_tau=0,
        seed=0,
    )
    m_np = compute_ring_metrics(
        cfg_np,
        safe_r_min=1,
        empowerment_H=2,
        empowerment_max_states=32,
        packaging_tau=0,
        seed=0,
    )

    emp_full = m_full["empowerment_median_on_K"]
    emp_np = m_np["empowerment_median_on_K"]
    assert emp_full >= emp_np + 0.2


def test_invariant_no_repair_collapses_viability():
    cfg_nr = ablations_suite()["no_repair"]
    m_nr = compute_ring_metrics(
        cfg_nr,
        safe_r_min=1,
        empowerment_H=2,
        empowerment_max_states=8,
        packaging_tau=0,
        seed=0,
    )
    assert m_nr["kernel_size_viable"] == 0

