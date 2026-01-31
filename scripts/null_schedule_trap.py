#!/usr/bin/env python3
"""Null schedule trap demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sbt_agency.channel import build_channel_matrix, enumerate_action_seqs
from sbt_agency.empowerment import capacity_bits
from sbt_agency.kernel import FiniteKernel


def _build_right_kernel() -> FiniteKernel:
    P = np.zeros((2, 4, 4))
    for a in range(2):
        for x in (0, 1):
            for s_ext in (0, 1):
                idx = 2 * x + s_ext
                x_next = s_ext
                s_next = 1 - s_ext
                idx_next = 2 * x_next + s_next
                P[a, idx, idx_next] = 1.0
    return FiniteKernel(P)


def _build_wrong_kernel() -> FiniteKernel:
    P = np.zeros((2, 2, 2))
    for x in (0, 1):
        P[0, x, 0] = 1.0
        P[1, x, 1] = 1.0
    return FiniteKernel(P)


def main() -> int:
    kernel_right = _build_right_kernel()
    d0 = np.array([0.5, 0.5, 0.0, 0.0])
    proj_x = lambda idx: idx // 2
    seqs = enumerate_action_seqs([0, 1], 1)
    W_right = build_channel_matrix(kernel_right, d0, seqs, proj_x)
    right_emp = capacity_bits(W_right)

    kernel_wrong = _build_wrong_kernel()
    s0 = 0
    proj = lambda s: s
    W_wrong = build_channel_matrix(kernel_wrong, s0, seqs, proj)
    wrong_emp = capacity_bits(W_wrong)

    print(f"wrong_emp={wrong_emp} right_emp={right_emp}")
    print(
        "The exogenous schedule s_ext drives x while actions have no effect, so the correct model yields identical"
    )
    print(
        "action rows and near-zero empowerment. Treating the schedule as if it were an action lets the agent pick x"
    )
    print("directly, producing spurious capacity around 1 bit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
