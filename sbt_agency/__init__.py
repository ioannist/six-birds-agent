"""Bootstrap import for local src/ layout."""

from __future__ import annotations

from pathlib import Path

# Resolve submodules directly. Reimporting this package after removing it from
# sys.modules recurses when src is already on sys.path behind the repository.
__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "sbt_agency")]
