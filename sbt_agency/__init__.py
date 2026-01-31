"""Bootstrap import for local src/ layout."""

from __future__ import annotations

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

REAL_PKG = os.path.join(SRC_PATH, "sbt_agency", "__init__.py")
if os.path.isfile(REAL_PKG):
    sys.modules.pop(__name__, None)
    _module = importlib.import_module(__name__)
    sys.modules[__name__] = _module
    globals().update(_module.__dict__)

