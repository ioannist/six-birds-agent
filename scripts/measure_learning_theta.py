#!/usr/bin/env python3
"""Measure empowerment vs theta for learning config."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sbt_agency.exp_configs import cfg_learning_theta
from sbt_agency.metrics import compute_empowerment_medians_by_theta
from sbt_agency.repro import stable_hash


def main() -> int:
    cfg = cfg_learning_theta()
    medians = compute_empowerment_medians_by_theta(
        cfg,
        safe_r_min=1,
        empowerment_H=2,
        restrict_u=0,
        restrict_phi=0,
        action_subset=("LEFT", "RIGHT"),
    )

    m0 = medians.get(0, 0.0)
    m1 = medians.get(1, 0.0)
    m2 = medians.get(2, 0.0)

    config_hash = stable_hash(asdict(cfg))

    out_dir = Path("results") / "learning"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"learning_theta_{config_hash}.json"
    png_path = out_dir / f"learning_theta_{config_hash}.png"

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "config_hash": config_hash,
        "config": asdict(cfg),
        "safe_r_min": 1,
        "empowerment_H": 2,
        "restrict_u": 0,
        "restrict_phi": 0,
        "action_subset": ["LEFT", "RIGHT"],
        "medians_by_theta": {"0": m0, "1": m1, "2": m2},
        "theta_values": [0, 1, 2],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(5, 3.5))
    plt.plot([0, 1, 2], [m0, m1, m2], marker="o")
    plt.xlabel("theta")
    plt.ylabel("median empowerment (bits)")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print(f"config_hash={config_hash}")
    print(f"medians: theta0={m0} theta1={m1} theta2={m2}")
    print(f"json: {json_path}")
    print(f"png: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
