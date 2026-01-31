#!/usr/bin/env python3
"""Sweep noise vs maintenance cost for ring agent metrics."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import platform

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sbt_agency.channel import enumerate_action_seqs
from sbt_agency.empowerment import feasible_capacity_bits
from sbt_agency.env_ring_agent import RingAgentConfig, build_kernel
from sbt_agency.exp_configs import (
    cfg_sweep_noise_maintenance_base,
    sweep_noise_maintenance_axes,
    sweep_noise_maintenance_run_id,
)
from sbt_agency.viability import ledger_feasible_actions, post_support_from_kernel, viability_kernel


def _cost_by_action_name(cfg: RingAgentConfig, name: str) -> int:
    if name == "LEFT":
        return cfg.cost_left
    if name == "RIGHT":
        return cfg.cost_right
    if name == "REPAIR":
        return cfg.cost_repair
    if name == "LEARN":
        return cfg.cost_learn
    return 0


def _compute_empowerment_median(
    kernel,
    metadata,
    projections,
    K,
    cost_fn,
    max_states: int = 16,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> float:
    if not K:
        return 0.0
    state_tuples = metadata["state_tuples"]
    L = metadata["dims"]["L"]
    n_actions = kernel.n_actions
    seqs = enumerate_action_seqs(list(range(n_actions)), 2)

    y_of_state = np.array([t[0] for t in state_tuples], dtype=int)
    P = kernel.to_dense()

    YDIST = np.zeros((n_actions, kernel.n_states, L), dtype=float)
    for a in range(n_actions):
        for s in range(kernel.n_states):
            YDIST[a, s] = np.bincount(y_of_state, weights=P[a, s], minlength=L)

    K_sorted = sorted(K)
    sample_states = K_sorted[: min(max_states, len(K_sorted))]

    caps = []
    for s in sample_states:
        budget = int(state_tuples[s][3])
        W = np.zeros((len(seqs), L), dtype=float)
        for i, (a0, a1) in enumerate(seqs):
            p1 = P[a0, s]
            W[i] = p1 @ YDIST[a1]
        caps.append(
            feasible_capacity_bits(W, seqs, cost_fn, budget, tol=tol, max_iter=max_iter)
        )

    return float(np.median(caps)) if caps else 0.0


def main() -> int:
    p_flip_values, repair_cost_values = sweep_noise_maintenance_axes()
    base_cfg = cfg_sweep_noise_maintenance_base()
    run_id = sweep_noise_maintenance_run_id()

    K_size = np.zeros((len(p_flip_values), len(repair_cost_values)), dtype=float)
    emp_median = np.zeros_like(K_size)

    for i, p_flip in enumerate(p_flip_values):
        for j, cost_repair in enumerate(repair_cost_values):
            cfg_dict = asdict(base_cfg)
            cfg_dict["p_flip"] = float(p_flip)
            cfg_dict["cost_repair"] = int(cost_repair)
            cfg = RingAgentConfig(**cfg_dict)
            kernel, projections, metadata = build_kernel(cfg)
            state_tuples = metadata["state_tuples"]

            def ledger(s: int) -> int:
                return int(state_tuples[s][3])

            def ubit(s: int) -> int:
                return int(state_tuples[s][1])

            def safe(s: int) -> bool:
                return ledger(s) >= 1 and ubit(s) == 0

            action_names = metadata["action_names"]

            def cost_fn(a_idx: int) -> float:
                name = action_names[int(a_idx)]
                return float(_cost_by_action_name(cfg, name))

            feasible_actions = ledger_feasible_actions(range(kernel.n_actions), ledger, cost_fn)
            post_support = post_support_from_kernel(kernel, atol=0.0)
            K = viability_kernel(range(kernel.n_states), range(kernel.n_actions), feasible_actions, post_support, safe)
            K_size[i, j] = float(len(K))
            emp_median[i, j] = _compute_empowerment_median(kernel, metadata, projections, K, cost_fn)

    out_dir = Path("results") / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"noise_maintenance_{run_id}.npz"
    meta = {
        "base_config": asdict(base_cfg),
        "safe": "r>=1 and u==0",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    np.savez(
        npz_path,
        p_flip_values=p_flip_values,
        repair_cost_values=np.array(repair_cost_values),
        K_size=K_size,
        emp_median=emp_median,
        meta_json=json.dumps(meta),
    )

    k_png = out_dir / f"noise_maintenance_K_{run_id}.png"
    e_png = out_dir / f"noise_maintenance_E_{run_id}.png"

    def _plot(data, path, title, cbar_label):
        plt.figure(figsize=(6, 4))
        plt.imshow(data, origin="lower", aspect="auto")
        plt.title(title)
        plt.xlabel("repair_cost")
        plt.ylabel("p_flip")
        plt.xticks(ticks=range(len(repair_cost_values)), labels=repair_cost_values)
        plt.yticks(ticks=range(len(p_flip_values)), labels=[f"{v:.2f}" for v in p_flip_values])
        plt.colorbar(label=cbar_label)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    _plot(K_size, k_png, "Viability kernel size", "|K|")
    _plot(emp_median, e_png, "Empowerment median on K", "bits")

    print(f"run_id: {run_id}")
    print(f"K_size_min: {K_size.min()} K_size_max: {K_size.max()}")
    print(f"emp_median_min: {emp_median.min()} emp_median_max: {emp_median.max()}")
    print(f"npz: {npz_path}")
    print(f"K_png: {k_png}")
    print(f"E_png: {e_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
