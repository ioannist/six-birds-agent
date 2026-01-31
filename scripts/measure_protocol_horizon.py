#!/usr/bin/env python3
"""Measure empowerment vs horizon with protocol on/off."""

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

from sbt_agency.exp_configs import ablations_suite
from sbt_agency.metrics import compute_ring_metrics
from sbt_agency.repro import stable_hash


def main() -> int:
    suite = ablations_suite()
    cfg_on = suite["full"]
    cfg_off = suite["no_protocol"]

    cfg_on_dict = asdict(cfg_on)
    cfg_off_dict = asdict(cfg_off)
    cfg_on_comp = dict(cfg_on_dict)
    cfg_off_comp = dict(cfg_off_dict)
    cfg_on_comp.pop("enable_protocol", None)
    cfg_off_comp.pop("enable_protocol", None)
    assert cfg_on_comp == cfg_off_comp

    hash_on = stable_hash(cfg_on_dict)
    hash_off = stable_hash(cfg_off_dict)

    H_list = [1, 2, 3, 4, 5]
    emp_on = []
    emp_off = []
    for H in H_list:
        m_on = compute_ring_metrics(
            cfg_on,
            safe_r_min=1,
            empowerment_H=H,
            empowerment_max_states=32,
            packaging_tau=0,
            seed=0,
        )
        m_off = compute_ring_metrics(
            cfg_off,
            safe_r_min=1,
            empowerment_H=H,
            empowerment_max_states=32,
            packaging_tau=0,
            seed=0,
        )
        emp_on.append(m_on["empowerment_median_on_K"])
        emp_off.append(m_off["empowerment_median_on_K"])

    k_on = compute_ring_metrics(
        cfg_on,
        safe_r_min=1,
        empowerment_H=2,
        empowerment_max_states=8,
        packaging_tau=0,
        seed=0,
    )["kernel_size_viable"]
    k_off = compute_ring_metrics(
        cfg_off,
        safe_r_min=1,
        empowerment_H=2,
        empowerment_max_states=8,
        packaging_tau=0,
        seed=0,
    )["kernel_size_viable"]

    out_dir = Path("results") / "protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"protocol_horizon_{hash_on}_{hash_off}"
    json_path = out_dir / f"{stem}.json"
    png_path = out_dir / f"{stem}.png"

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "config_hash_on": hash_on,
        "config_hash_off": hash_off,
        "config_on": cfg_on_dict,
        "config_off": cfg_off_dict,
        "H_list": H_list,
        "emp_on": emp_on,
        "emp_off": emp_off,
        "kernel_size_viable_on": k_on,
        "kernel_size_viable_off": k_off,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(6, 4))
    plt.plot(H_list, emp_on, marker="o", label="protocol_on")
    plt.plot(H_list, emp_off, marker="o", label="protocol_off")
    plt.xlabel("H")
    plt.ylabel("median feasible empowerment on K (bits)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    gaps = [a - b for a, b in zip(emp_on, emp_off)]
    max_gap = max(gaps)
    max_idx = gaps.index(max_gap)
    max_H = H_list[max_idx]

    print(f"hash_on: {hash_on}")
    print(f"hash_off: {hash_off}")
    print(f"emp_on: {emp_on}")
    print(f"emp_off: {emp_off}")
    print(f"json: {json_path}")
    print(f"png: {png_path}")
    print(f"max_gap: {max_gap} at H={max_H}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
