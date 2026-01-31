#!/usr/bin/env python3
"""Run baseline rollouts and save a qualitative trace."""

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

from sbt_agency.env_ring_agent import RingAgentConfig, build_kernel
from sbt_agency.policies import (
    cost_map_from_config,
    make_maintenance_first,
    make_move_right_if_possible,
    make_random_feasible,
)
from sbt_agency.repro import stable_hash
from sbt_agency.sim import rollout


def main() -> int:
    config = RingAgentConfig()
    kernel, _projections, metadata = build_kernel(config)

    config_dict = asdict(config)
    config_hash = stable_hash(config_dict)

    seed = 0
    rng = np.random.default_rng(seed)

    action_names = metadata["action_names"]
    cost_by_name = cost_map_from_config(config, action_names)

    state_tuples = metadata["state_tuples"]
    tuple_to_state = metadata["tuple_to_state"]
    init_tuple = (0, 1, 0, config.R_max, 0, 0)
    s0 = tuple_to_state[init_tuple]

    n_steps = 30
    policies = {
        "random_feasible": make_random_feasible(action_names, cost_by_name, rng),
        "maintenance_first": make_maintenance_first(action_names, cost_by_name),
        "move_right": make_move_right_if_possible(action_names, cost_by_name, rng),
    }

    runs = {}
    for name, pi in policies.items():
        runs[name] = rollout(
            kernel,
            s0,
            n_steps,
            pi,
            state_tuples=state_tuples,
            action_names=action_names,
            rng=rng,
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "config": config_dict,
        "seed": seed,
        "n_steps": n_steps,
        "action_names": action_names,
        "initial_state_tuple": init_tuple,
        "runs": runs,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    out_dir = Path("results") / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trace_{config_hash}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"Saved trace: {out_path}",
        f"config_hash: {config_hash}",
    ]
    for name, traj in runs.items():
        last = traj[-1]["state_next"]
        y, u, phi, r, _g, _theta = last
        lines.append(f"{name}: y={y} r={r} u={u} phi={phi}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
