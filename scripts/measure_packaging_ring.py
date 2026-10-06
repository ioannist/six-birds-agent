#!/usr/bin/env python3
"""Measure packaging defect for ring agent with/without repair."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import platform

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib
import matplotlib.pyplot as plt

from sbt_agency.env_ring_agent import build_kernel
from sbt_agency.exp_configs import cfg_packaging_ring_off, cfg_packaging_ring_on
from sbt_agency.packaging import empirical_endomap, idempotence_defect
from sbt_agency.repro import stable_hash
from sbt_agency.viability import ledger_feasible_actions, post_support_from_kernel, viability_kernel


def _policy_right(action_idx_right: int):
    def policy(_s_idx: int) -> int:
        return action_idx_right

    return policy


def _policy_funded_preventive_repair(action_idx_repair, action_idx_right, state_tuples, cost_repair):
    """Pay for preventive repair whenever affordable; otherwise move RIGHT.

    The matched substrate earns one unit per step, so funded states can keep
    paying indefinitely. No rejected command is used as an idle action.
    """
    def policy(s_idx: int) -> int:
        r = state_tuples[s_idx][3]
        return action_idx_repair if r >= cost_repair else action_idx_right
    return policy


def _compute_defects(kernel, proj_macro, policy, tau_list):
    defects = []
    for tau in tau_list:
        E = empirical_endomap(kernel, proj_macro, tau=tau, policy=policy)
        defects.append(idempotence_defect(E))
    return defects


def _coherent_viability(cfg, kernel, metadata):
    states = metadata["state_tuples"]
    cost_by_name = {"LEFT": cfg.cost_left, "RIGHT": cfg.cost_right, "REPAIR": cfg.cost_repair}
    cost = lambda a: cost_by_name[metadata["action_names"][a]]
    feasible = ledger_feasible_actions(range(kernel.n_actions), lambda s: states[s][3], cost)
    return viability_kernel(range(kernel.n_states), range(kernel.n_actions), feasible,
                            post_support_from_kernel(kernel), lambda s: states[s][3] >= 1 and states[s][1] == 0)


def main() -> int:
    cfg_off = cfg_packaging_ring_off()
    cfg_on = cfg_packaging_ring_on()

    kernel_off, projections_off, metadata_off = build_kernel(cfg_off)
    kernel_on, projections_on, metadata_on = build_kernel(cfg_on)

    proj_macro_off = projections_off["proj_macro"]
    proj_macro_on = projections_on["proj_macro"]

    action_names_off = metadata_off["action_names"]
    action_names_on = metadata_on["action_names"]

    right_idx_off = action_names_off.index("RIGHT")
    right_idx_on = action_names_on.index("RIGHT")
    repair_idx_on = action_names_on.index("REPAIR") if "REPAIR" in action_names_on else None

    policy_off = _policy_right(right_idx_off)
    policy_on = _policy_funded_preventive_repair(
        repair_idx_on, right_idx_on, metadata_on["state_tuples"], cfg_on.cost_repair
    )

    tau_list = list(range(1, 11))
    defects_off = _compute_defects(kernel_off, proj_macro_off, policy_off, tau_list)
    defects_on = _compute_defects(kernel_on, proj_macro_on, policy_on, tau_list)
    coherent_K_off = _coherent_viability(cfg_off, kernel_off, metadata_off)
    coherent_K_on = _coherent_viability(cfg_on, kernel_on, metadata_on)

    config_hash_off = stable_hash(asdict(cfg_off))
    config_hash_on = stable_hash(asdict(cfg_on))

    out_dir = Path("results") / "packaging"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"packaging_ring_{config_hash_off}_{config_hash_on}"
    json_path = out_dir / f"{stem}.json"
    png_path = out_dir / f"{stem}.png"

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_off": asdict(cfg_off),
        "config_on": asdict(cfg_on),
        "config_hash_off": config_hash_off,
        "config_hash_on": config_hash_on,
        "tau_list": tau_list,
        "defect_off": defects_off,
        "defect_on": defects_on,
        "policy_on_semantics": "paid preventive REPAIR whenever affordable; otherwise RIGHT",
        "funding": "one unit income each step at every site; REPAIR costs one unit",
        "coherent_safety": "r>=1 and u==0",
        "coherent_K_off": sorted(coherent_K_off),
        "coherent_K_on": sorted(coherent_K_on),
        "scope": "modal idempotence under specified policies; it does not alone imply stochastic closure or repair necessity",
        "lens": "proj_macro = (y,r,phi) encoded as int",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(6, 4))
    plt.plot(tau_list, defects_off, marker="o", label="repair OFF")
    plt.plot(tau_list, defects_on, marker="o", label="funded preventive repair ON")
    plt.xlabel("tau")
    plt.ylabel("idempotence defect")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    idx_tau2 = tau_list.index(2)
    lines = [
        f"hash_off: {config_hash_off}",
        f"hash_on: {config_hash_on}",
        f"defect_tau2_off: {defects_off[idx_tau2]}",
        f"defect_tau2_on: {defects_on[idx_tau2]}",
        f"json: {json_path}",
        f"png: {png_path}",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
