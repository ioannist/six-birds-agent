"""Ring-world agent environment with protocol and ledger gating."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from sbt_agency.kernel import FiniteKernel


@dataclass(frozen=True)
class RingAgentConfig:
    L: int = 8
    m_phase: int = 2
    R_max: int = 3
    g_size: int = 2
    theta_max: int = 1
    enable_protocol: bool = True
    enable_repair: bool = True
    enable_learn: bool = False
    identity_on: bool = True
    p_flip: float = 0.1
    p_slip: float = 0.0
    p_repair: float = 1.0
    slip_improve_per_theta: float = 0.0
    gain_positions: tuple[int, ...] = (0,)
    gain_amount: int = 1
    maint_cost: int = 0
    cost_left: int = 1
    cost_right: int = 1
    cost_repair: int = 1
    cost_learn: int = 1


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _movement_displacement(config: RingAgentConfig, action: str, u: int, phi: int) -> int:
    if action not in {"LEFT", "RIGHT"}:
        return 0
    step = 1
    if config.enable_protocol:
        step = 1 if (phi % 2 == 0) else 2
    disp = step if action == "LEFT" else -step
    if u == 1:
        disp = -disp
    return disp


def build_kernel(
    config: RingAgentConfig,
) -> tuple[FiniteKernel, dict, dict]:
    """Build the ring-agent transition kernel and projections."""
    action_names = ["LEFT", "RIGHT"]
    if config.enable_repair:
        action_names.append("REPAIR")
    if config.enable_learn:
        action_names.append("LEARN")

    cost_by_action = {
        "LEFT": config.cost_left,
        "RIGHT": config.cost_right,
        "REPAIR": config.cost_repair,
        "LEARN": config.cost_learn,
    }

    state_tuples: list[tuple[int, int, int, int, int, int]] = []
    for y in range(config.L):
        for u in range(2):
            for phi in range(config.m_phase):
                for r in range(config.R_max + 1):
                    for g in range(config.g_size):
                        for theta in range(config.theta_max + 1):
                            state_tuples.append((y, u, phi, r, g, theta))

    tuple_to_state = {t: i for i, t in enumerate(state_tuples)}
    n_states = len(state_tuples)
    n_actions = len(action_names)

    P = np.zeros((n_actions, n_states, n_states), dtype=float)

    for s_idx, (y, u, phi, r, g, theta) in enumerate(state_tuples):
        phi_next = (phi + 1) % config.m_phase
        slip_eff = _clip01(config.p_slip - theta * config.slip_improve_per_theta)

        for a_idx, action in enumerate(action_names):
            cost = cost_by_action[action]
            executed = r >= cost

            movement = action in {"LEFT", "RIGHT"} and executed
            if movement:
                disp = _movement_displacement(config, action, u, phi)
                y_move = (y + disp) % config.L
                slip_outcomes = []
                if 1.0 - slip_eff > 0.0:
                    slip_outcomes.append((y_move, 1.0 - slip_eff))
                if slip_eff > 0.0:
                    slip_outcomes.append((y, slip_eff))
            else:
                slip_outcomes = [(y, 1.0)]

            if config.p_flip <= 0.0:
                u_noise_outcomes = [(u, 1.0)]
            elif config.p_flip >= 1.0:
                u_noise_outcomes = [(1 - u, 1.0)]
            else:
                u_noise_outcomes = [(u, 1.0 - config.p_flip), (1 - u, config.p_flip)]

            for y_post, p_slip in slip_outcomes:
                for u_noise, p_flip in u_noise_outcomes:
                    if action == "REPAIR" and executed:
                        if config.p_repair >= 1.0:
                            repair_outcomes = [(0, 1.0)]
                        elif config.p_repair <= 0.0:
                            repair_outcomes = [(u_noise, 1.0)]
                        else:
                            repair_outcomes = [
                                (0, config.p_repair),
                                (u_noise, 1.0 - config.p_repair),
                            ]
                    else:
                        repair_outcomes = [(u_noise, 1.0)]

                    for u_post, p_repair in repair_outcomes:
                        theta_post = theta
                        if action == "LEARN" and executed:
                            theta_post = min(theta + 1, config.theta_max)

                        g_post = g

                        r_post = r
                        if y_post in config.gain_positions:
                            r_post += config.gain_amount
                        r_post -= config.maint_cost
                        if executed:
                            r_post -= cost
                        r_post = max(0, min(config.R_max, r_post))

                        next_tuple = (y_post, u_post, phi_next, r_post, g_post, theta_post)
                        s2_idx = tuple_to_state[next_tuple]
                        P[a_idx, s_idx, s2_idx] += p_slip * p_flip * p_repair

    kernel = FiniteKernel(P)

    def proj_y(s_idx: int) -> int:
        return state_tuples[s_idx][0]

    def proj_macro(s_idx: int) -> int:
        y_val, _u, phi_val, r_val, _g, _theta = state_tuples[s_idx]
        return (phi_val * (config.R_max + 1) + r_val) * config.L + y_val

    projections = {
        "proj_y": proj_y,
        "proj_macro": proj_macro,
    }

    metadata = {
        "config": asdict(config),
        "action_names": action_names,
        "state_tuples": state_tuples,
        "tuple_to_state": tuple_to_state,
        "dims": {
            "L": config.L,
            "m_phase": config.m_phase,
            "R_max": config.R_max,
            "g_size": config.g_size,
            "theta_max": config.theta_max,
        },
    }

    return kernel, projections, metadata

