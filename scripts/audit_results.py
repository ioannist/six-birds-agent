#!/usr/bin/env python3
"""Audit results artifacts under a root directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sbt_agency.audit import audit_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = audit_results(args.root, strict=args.strict)
    print(
        f"AUDIT summary: checked={result['checked']} errors={result['errors']} warnings={result['warnings']}"
    )

    if result["errors"] > 0:
        return 1
    if args.strict and result["warnings"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
