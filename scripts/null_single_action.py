#!/usr/bin/env python3
"""Null test: single-action regime yields zero empowerment."""

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


def main() -> int:
    P = np.zeros((1, 3, 3))
    P[0, 0, 1] = 1.0
    P[0, 1, 2] = 1.0
    P[0, 2, 0] = 1.0
    kernel = FiniteKernel(P)

    s0 = 0
    proj = lambda s: s
    caps = []
    for H in (1, 2, 3):
        seqs = enumerate_action_seqs([0], H)
        W = build_channel_matrix(kernel, s0, seqs, proj)
        caps.append(capacity_bits(W))

    print(f"H=1,2,3: {caps[0]} {caps[1]} {caps[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
