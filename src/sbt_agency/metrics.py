"""Metrics extraction for ring-agent environments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import math

import numpy as np

from sbt_agency.channel import build_channel_matrix, enumerate_action_seqs
from sbt_agency.empowerment import feasible_capacity_bits
from sbt_agency.env_ring_agent import RingAgentConfig, build_kernel
from sbt_agency.packaging import empirical_endomap, idempotence_defect as _idempotence_defect
from sbt_agency.repro import set_global_seed, stable_hash
from sbt_agency.viability import ledger_feasible_actions, post_support_from_kernel, viability_kernel


def _cost_by_action_name(config: RingAgentConfig, name: str) -> int:
    if name == "LEFT":
        return config.cost_left
    if name == "RIGHT":
        return config.cost_right
    if name == "REPAIR":
        return config.cost_repair
    if name == "LEARN":
        return config.cost_learn
    return 0


def compute_empowerment_medians_by_theta(
    config: RingAgentConfig,
    *,
    safe_r_min: int = 1,
    empowerment_H: int = 2,
    restrict_u: int | None = 0,
    restrict_phi: int | None = 0,
    action_subset: tuple[str, ...] = ("LEFT", "RIGHT"),
) -> dict[int, float]:
    """Compute median feasible-empowerment by theta group."""
    kernel, projections, metadata = build_kernel(config)
    action_names = metadata["action_names"]
    state_tuples = metadata["state_tuples"]

    def ledger(s: int) -> int:
        return int(state_tuples[s][3])

    def cost_fn(a_idx: int) -> float:
        name = action_names[int(a_idx)]
        return float(_cost_by_action_name(config, name))

    states = range(kernel.n_states)
    actions = range(kernel.n_actions)
    feasible_actions = ledger_feasible_actions(actions, ledger, cost_fn)
    post_support = post_support_from_kernel(kernel)

    def safe(s: int) -> bool:
        return ledger(s) >= safe_r_min

    K = viability_kernel(states, actions, feasible_actions, post_support, safe)

    action_indices = [action_names.index(name) for name in action_subset if name in action_names]
    if not action_indices:
        raise ValueError("action_subset yields no valid action indices")

    seqs = enumerate_action_seqs(action_indices, empowerment_H)
    proj_y = projections["proj_y"]

    theta_max = config.theta_max
    medians: dict[int, float] = {}
    for theta in range(theta_max + 1):
        theta_states = []
        for s in K:
            y, u, phi, r, g, th = state_tuples[s]
            if th != theta:
                continue
            if restrict_u is not None and u != restrict_u:
                continue
            if restrict_phi is not None and phi != restrict_phi:
                continue
            theta_states.append(s)

        if not theta_states:
            medians[theta] = 0.0
            continue

        caps = []
        for s in theta_states:
            budget = ledger(s)
            W = build_channel_matrix(kernel, s, seqs, proj_y)
            caps.append(feasible_capacity_bits(W, seqs, cost_fn, budget))
        medians[theta] = float(np.median(caps)) if caps else 0.0

    return medians


def compute_ring_metrics(
    config: RingAgentConfig,
    *,
    safe_r_min: int = 1,
    empowerment_H: int = 2,
    empowerment_max_states: int = 32,
    packaging_tau: int = 2,
    seed: int = 0,
) -> dict[str, float | int | str | dict | list]:
    """Compute viability, empowerment, and packaging metrics for a ring config."""
    set_global_seed(seed)
    kernel, projections, metadata = build_kernel(config)

    config_dict = asdict(config)
    config_hash = stable_hash(config_dict)

    action_names = metadata["action_names"]
    state_tuples = metadata["state_tuples"]

    def ledger(s: int) -> int:
        return int(state_tuples[s][3])

    def cost_fn(a_idx: int) -> float:
        name = action_names[int(a_idx)]
        return float(_cost_by_action_name(config, name))

    states = range(kernel.n_states)
    actions = range(kernel.n_actions)
    feasible_actions = ledger_feasible_actions(actions, ledger, cost_fn)
    post_support = post_support_from_kernel(kernel)

    def safe(s: int) -> bool:
        return ledger(s) >= safe_r_min

    K = viability_kernel(states, actions, feasible_actions, post_support, safe)
    kernel_size_viable = len(K)

    if not K:
        empowerment_median_on_K = 0.0
    else:
        rng = np.random.default_rng(seed)
        K_list = sorted(K)
        sample_n = min(len(K_list), empowerment_max_states)
        if len(K_list) > sample_n:
            sample_states = rng.choice(K_list, size=sample_n, replace=False)
        else:
            sample_states = np.array(K_list, dtype=int)
        seqs = enumerate_action_seqs(list(actions), empowerment_H)
        proj_y = projections["proj_y"]
        caps = []
        for s in sample_states.tolist():
            budget = ledger(int(s))
            W = build_channel_matrix(kernel, int(s), seqs, proj_y)
            caps.append(feasible_capacity_bits(W, seqs, cost_fn, budget))
        empowerment_median_on_K = float(np.median(caps)) if caps else 0.0

    proj_macro = projections["proj_macro"]
    right_idx = action_names.index("RIGHT") if "RIGHT" in action_names else 0
    repair_idx = action_names.index("REPAIR") if "REPAIR" in action_names else None

    def policy(s_idx: int) -> int:
        y, u, phi, r, g, theta = state_tuples[s_idx]
        if repair_idx is not None and u == 1 and r >= _cost_by_action_name(config, "REPAIR"):
            return repair_idx
        return right_idx

    E = empirical_endomap(kernel, proj_macro, packaging_tau, policy)
    defect = float(_idempotence_defect(E))

    return {
        "config_hash": config_hash,
        "n_states": kernel.n_states,
        "n_actions": kernel.n_actions,
        "kernel_size_viable": kernel_size_viable,
        "empowerment_median_on_K": empowerment_median_on_K,
        "idempotence_defect": defect,
        "config": config_dict,
        "action_names": list(action_names),
    }
