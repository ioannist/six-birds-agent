"""Canonical experiment configs and sweep axes."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Dict, List, Tuple

import numpy as np

from sbt_agency.env_ring_agent import RingAgentConfig
from sbt_agency.repro import stable_hash


def cfg_packaging_ring_off() -> RingAgentConfig:
    return RingAgentConfig(
        L=8,
        m_phase=2,
        R_max=1,
        g_size=1,
        theta_max=0,
        enable_protocol=True,
        enable_repair=False,
        enable_learn=False,
        identity_on=True,
        p_flip=0.1,
        p_slip=0.2,
        p_repair=1.0,
        slip_improve_per_theta=0.0,
        gain_positions=(0,),
        gain_amount=0,
        maint_cost=0,
        cost_left=0,
        cost_right=0,
        cost_repair=1,
        cost_learn=0,
    )


def cfg_packaging_ring_on() -> RingAgentConfig:
    return replace(cfg_packaging_ring_off(), enable_repair=True)


def ablations_suite() -> Dict[str, RingAgentConfig]:
    base = {
        "L": 8,
        "m_phase": 2,
        "R_max": 2,
        "g_size": 1,
        "p_flip": 0.1,
        "p_slip": 0.2,
        "p_repair": 1.0,
        "gain_positions": (0,),
        "gain_amount": 0,
        "maint_cost": 0,
        "cost_left": 1,
        "cost_right": 1,
        "cost_repair": 0,
        "cost_learn": 0,
    }

    suite: Dict[str, RingAgentConfig] = {}
    suite["full"] = RingAgentConfig(
        **base,
        enable_protocol=True,
        enable_repair=True,
        enable_learn=False,
        theta_max=0,
        identity_on=True,
    )
    suite["no_protocol"] = RingAgentConfig(
        **base,
        enable_protocol=False,
        enable_repair=True,
        enable_learn=False,
        theta_max=0,
        identity_on=True,
    )
    suite["no_repair"] = RingAgentConfig(
        **base,
        enable_protocol=True,
        enable_repair=False,
        enable_learn=False,
        theta_max=0,
        identity_on=True,
    )
    suite["constraints_off"] = RingAgentConfig(
        **{**base, "cost_left": 0, "cost_right": 0, "cost_repair": 0, "cost_learn": 0},
        enable_protocol=True,
        enable_repair=False,
        enable_learn=False,
        theta_max=0,
        identity_on=True,
    )
    suite["learn_on"] = RingAgentConfig(
        **base,
        enable_protocol=True,
        enable_repair=True,
        enable_learn=True,
        theta_max=1,
        slip_improve_per_theta=0.2,
        identity_on=True,
    )
    suite["high_noise"] = RingAgentConfig(
        **{**base, "p_flip": 0.4, "p_slip": 0.4},
        enable_protocol=True,
        enable_repair=True,
        enable_learn=False,
        theta_max=0,
        identity_on=True,
    )
    suite["repair_imperfect"] = RingAgentConfig(
        **{**base, "p_repair": 0.5},
        enable_protocol=True,
        enable_repair=True,
        enable_learn=False,
        theta_max=0,
        identity_on=True,
    )

    return suite


def cfg_sweep_noise_maintenance_base() -> RingAgentConfig:
    return RingAgentConfig(
        L=8,
        m_phase=1,
        R_max=7,
        g_size=1,
        theta_max=0,
        enable_protocol=False,
        enable_repair=True,
        enable_learn=False,
        identity_on=True,
        p_slip=0.0,
        p_repair=1.0,
        slip_improve_per_theta=0.0,
        gain_positions=(0,),
        gain_amount=4,
        maint_cost=0,
        cost_left=1,
        cost_right=1,
        cost_learn=0,
    )


def sweep_noise_maintenance_axes() -> Tuple[List[float], List[int]]:
    p_flip_values = list(np.linspace(0.0, 0.7, 8))
    repair_cost_values = list(range(8))
    return p_flip_values, repair_cost_values


def sweep_noise_maintenance_run_id() -> str:
    base_cfg = cfg_sweep_noise_maintenance_base()
    p_flip_values, repair_cost_values = sweep_noise_maintenance_axes()
    return stable_hash(
        {
            "base_config": asdict(base_cfg),
            "p_flip_values": list(p_flip_values),
            "repair_cost_values": list(repair_cost_values),
            "safe": "r>=1 and u==0",
            "H": 2,
            "N": 16,
        }
    )


def cfg_learning_theta() -> RingAgentConfig:
    return RingAgentConfig(
        L=8,
        m_phase=1,
        R_max=2,
        g_size=1,
        theta_max=2,
        enable_protocol=False,
        enable_repair=False,
        enable_learn=True,
        identity_on=True,
        p_flip=0.0,
        p_slip=0.4,
        p_repair=1.0,
        slip_improve_per_theta=0.15,
        gain_positions=(0,),
        gain_amount=0,
        maint_cost=0,
        cost_left=0,
        cost_right=0,
        cost_repair=0,
        cost_learn=0,
    )
