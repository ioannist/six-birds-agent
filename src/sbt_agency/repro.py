"""Reproducibility helpers for deterministic configs and manifests."""

from __future__ import annotations

from dataclasses import is_dataclass, fields
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds and export PYTHONHASHSEED for child processes."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stringify_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    return _stable_json_dumps(_canonicalize(key))


def _canonicalize(obj: Any) -> Any:
    if is_dataclass(obj):
        return {field.name: _canonicalize(getattr(obj, field.name)) for field in fields(obj)}

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}

    if isinstance(obj, np.generic):
        return _canonicalize(obj.item())

    if isinstance(obj, dict):
        items = [(_stringify_key(k), _canonicalize(v)) for k, v in obj.items()]
        items.sort(key=lambda item: item[0])
        return {k: v for k, v in items}

    if isinstance(obj, (list, tuple)):
        return [_canonicalize(item) for item in obj]

    if isinstance(obj, (set, frozenset)):
        normalized = [_canonicalize(item) for item in obj]
        normalized.sort(key=_stable_json_dumps)
        return normalized

    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj

    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj

    raise TypeError(f"Unsupported type for stable hashing: {type(obj)!r}")


def stable_hash(obj: Any) -> str:
    """Return a deterministic sha256 hex digest for config-like objects."""
    canonical = _canonicalize(obj)
    payload = _stable_json_dumps(canonical).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_run_manifest(
    path: str | Path,
    config: Any,
    versions: dict[str, Any] | None = None,
    notes: Any | None = None,
) -> dict[str, Any]:
    """Write a manifest JSON file and return the manifest dict."""
    target = Path(path)
    if target.suffix != ".json":
        target = target / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    canonical_config = _canonicalize(config)
    versions_payload = _canonicalize(versions) if versions is not None else {}
    if not isinstance(versions_payload, dict):
        raise TypeError("versions must be a mapping")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": stable_hash(config),
        "config": canonical_config,
        "versions": versions_payload,
        "notes": notes,
    }
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

