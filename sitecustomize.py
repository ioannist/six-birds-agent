"""Ensure src/ is on sys.path when running from repo root."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(__file__)
SRC_PATH = os.path.join(REPO_ROOT, "src")

if os.path.isdir(SRC_PATH) and SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

