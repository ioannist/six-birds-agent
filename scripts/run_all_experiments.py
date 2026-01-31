#!/usr/bin/env python3
"""Run all experiment scripts and audit results in strict mode."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.clean:
        for rel in [
            "results/rollouts",
            "results/packaging",
            "results/ablations",
            "results/sweeps",
            "results/protocol",
            "results/learning",
        ]:
            path = repo_root / rel
            if path.exists():
                shutil.rmtree(path)

    scripts = [
        repo_root / "scripts" / "run_rollouts.py",
        repo_root / "scripts" / "measure_packaging_ring.py",
        repo_root / "scripts" / "run_ablations.py",
        repo_root / "scripts" / "sweep_noise_maintenance.py",
        repo_root / "scripts" / "measure_protocol_horizon.py",
        repo_root / "scripts" / "measure_learning_theta.py",
    ]

    for script in scripts:
        _run([sys.executable, str(script)])

    _run([sys.executable, str(repo_root / "scripts" / "audit_results.py"), "--strict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
