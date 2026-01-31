#!/usr/bin/env python3
"""Run primitive ablations and summarize metrics."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sbt_agency.exp_configs import ablations_suite
from sbt_agency.metrics import compute_ring_metrics


def main() -> int:
    out_dir = Path("results") / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    suite = ablations_suite()
    for name in sorted(suite.keys()):
        config = suite[name]
        metrics = compute_ring_metrics(
            config,
            safe_r_min=1,
            empowerment_H=2,
            empowerment_max_states=32,
            packaging_tau=2,
            seed=0,
        )
        config_hash = metrics["config_hash"]

        run_payload = {
            "name": name,
            "config_hash": config_hash,
            "config": asdict(config),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        run_path = out_dir / f"run_{name}_{config_hash}.json"
        run_path.write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")

        rows.append(
            {
                "name": name,
                "config_hash": config_hash,
                "n_states": metrics["n_states"],
                "n_actions": metrics["n_actions"],
                "kernel_size_viable": metrics["kernel_size_viable"],
                "empowerment_median_on_K": metrics["empowerment_median_on_K"],
                "idempotence_defect": metrics["idempotence_defect"],
            }
        )

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "name",
        "config_hash",
        "n_states",
        "n_actions",
        "kernel_size_viable",
        "empowerment_median_on_K",
        "idempotence_defect",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
