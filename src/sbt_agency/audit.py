"""Audit utilities for results artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import math

import numpy as np

from sbt_agency.repro import stable_hash


def _is_mapping(obj: Any) -> bool:
    return isinstance(obj, dict)


def _add_issue(issues: list[dict], level: str, path: Path, message: str) -> None:
    issues.append({"level": level, "path": str(path), "message": message})


def _validate_kernel_P(P: Any, path: Path, issues: list[dict]) -> None:
    try:
        arr = np.asarray(P, dtype=float)
    except Exception:
        _add_issue(issues, "error", path, "kernel P is not numeric")
        return
    if arr.ndim != 3:
        _add_issue(issues, "error", path, "kernel P must be 3D")
        return
    if not np.all(np.isfinite(arr)):
        _add_issue(issues, "error", path, "kernel P contains non-finite values")
        return
    if np.any(arr < -1e-12):
        _add_issue(issues, "error", path, "kernel P has negative entries")
        return
    row_sums = arr.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-9, rtol=0.0):
        _add_issue(issues, "error", path, "kernel P rows do not sum to 1")


def _require_keys(data: dict, path: Path, issues: list[dict], keys: Iterable[str]) -> bool:
    missing = [k for k in keys if k not in data]
    if missing:
        _add_issue(issues, "error", path, f"missing keys: {missing}")
        return False
    return True


def _check_config_hash(data: dict, path: Path, issues: list[dict]) -> None:
    if "config" in data and "config_hash" in data:
        if not isinstance(data["config"], dict) or not isinstance(data["config_hash"], str):
            _add_issue(issues, "error", path, "config or config_hash has wrong type")
            return
        expected = stable_hash(data["config"])
        if expected != data["config_hash"]:
            _add_issue(issues, "error", path, "config_hash does not match config")


def _check_timestamp_and_versions(data: dict, path: Path, issues: list[dict]) -> None:
    if not any(k in data for k in ("timestamp", "created_at_utc", "created_at")):
        _add_issue(issues, "warning", path, "missing timestamp")
    if "versions" not in data:
        _add_issue(issues, "warning", path, "missing versions")
    elif not isinstance(data["versions"], dict):
        _add_issue(issues, "error", path, "versions must be a dict")


def _detect_type(data: dict) -> str:
    if all(k in data for k in ("runs", "action_names", "n_steps")):
        return "rollout_trace"
    if all(k in data for k in ("defect_off", "defect_on", "tau_list")):
        return "packaging_ring"
    if all(
        k in data
        for k in (
            "config_on",
            "config_off",
            "config_hash_on",
            "config_hash_off",
            "H_list",
            "emp_on",
            "emp_off",
        )
    ):
        return "protocol_horizon"
    if all(k in data for k in ("metrics", "config_hash", "config")):
        return "metrics_run"
    return "generic"


def _validate_rollout_trace(data: dict, path: Path, issues: list[dict], strict: bool) -> None:
    if not _require_keys(
        data,
        path,
        issues,
        ("config_hash", "config", "seed", "n_steps", "action_names", "runs"),
    ):
        return
    if not isinstance(data["runs"], dict):
        _add_issue(issues, "error", path, "runs must be a dict")
        return
    n_steps = data["n_steps"]
    for name, traj in data["runs"].items():
        if not isinstance(traj, list):
            _add_issue(issues, "error", path, f"run {name} is not a list")
            continue
        required = {"t", "s", "state", "a", "s_next", "state_next"}
        for step in traj:
            if not isinstance(step, dict):
                _add_issue(issues, "error", path, f"run {name} step is not a dict")
                break
            if not required.issubset(step.keys()):
                _add_issue(issues, "error", path, f"run {name} missing step keys")
                break
        if isinstance(n_steps, int) and len(traj) != n_steps:
            level = "error" if strict else "warning"
            _add_issue(issues, level, path, f"run {name} length != n_steps")


def _validate_packaging_ring(data: dict, path: Path, issues: list[dict]) -> None:
    if not _require_keys(
        data,
        path,
        issues,
        (
            "config_hash_off",
            "config_hash_on",
            "config_off",
            "config_on",
            "tau_list",
            "defect_off",
            "defect_on",
        ),
    ):
        return
    if stable_hash(data["config_off"]) != data["config_hash_off"]:
        _add_issue(issues, "error", path, "config_hash_off mismatch")
    if stable_hash(data["config_on"]) != data["config_hash_on"]:
        _add_issue(issues, "error", path, "config_hash_on mismatch")
    if not (
        len(data["tau_list"]) == len(data["defect_off"]) == len(data["defect_on"])
    ):
        _add_issue(issues, "error", path, "tau_list/defect lengths mismatch")
    for val in list(data["defect_off"]) + list(data["defect_on"]):
        if not math.isfinite(float(val)):
            _add_issue(issues, "error", path, "defect is not finite")
            break
        if float(val) < -1e-9 or float(val) > 1.0 + 1e-9:
            _add_issue(issues, "error", path, "defect out of [0,1] range")
            break


def _validate_protocol_horizon(data: dict, path: Path, issues: list[dict]) -> None:
    if not _require_keys(
        data,
        path,
        issues,
        (
            "config_on",
            "config_off",
            "config_hash_on",
            "config_hash_off",
            "H_list",
            "emp_on",
            "emp_off",
        ),
    ):
        return
    if stable_hash(data["config_on"]) != data["config_hash_on"]:
        _add_issue(issues, "error", path, "config_hash_on mismatch")
    if stable_hash(data["config_off"]) != data["config_hash_off"]:
        _add_issue(issues, "error", path, "config_hash_off mismatch")
    if not (
        len(data["H_list"]) == len(data["emp_on"]) == len(data["emp_off"])
    ):
        _add_issue(issues, "error", path, "H_list/emp lengths mismatch")
    for val in list(data["emp_on"]) + list(data["emp_off"]):
        if not math.isfinite(float(val)):
            _add_issue(issues, "error", path, "empowerment is not finite")
            break
        if float(val) < -1e-12:
            _add_issue(issues, "error", path, "empowerment is negative")
            break


def _validate_metrics_run(data: dict, path: Path, issues: list[dict], strict: bool) -> None:
    if "metrics" not in data or not isinstance(data["metrics"], dict):
        _add_issue(issues, "error", path, "metrics missing or not a dict")
        return
    if "config_hash" in data["metrics"]:
        if data["metrics"]["config_hash"] != data.get("config_hash"):
            level = "error" if strict else "warning"
            _add_issue(issues, level, path, "metrics.config_hash mismatch")
    if "idempotence_defect" in data["metrics"]:
        val = float(data["metrics"]["idempotence_defect"])
        if val < -1e-9 or val > 1.0 + 1e-9:
            _add_issue(issues, "error", path, "idempotence_defect out of range")
    if "empowerment_median_on_K" in data["metrics"]:
        val = float(data["metrics"]["empowerment_median_on_K"])
        if not math.isfinite(val) or val < -1e-12:
            _add_issue(issues, "error", path, "empowerment_median_on_K invalid")
    if "kernel_size_viable" in data["metrics"]:
        val = data["metrics"]["kernel_size_viable"]
        if not isinstance(val, int) or val < 0:
            _add_issue(issues, "error", path, "kernel_size_viable invalid")


def audit_results(root: str | Path, strict: bool = False) -> dict:
    root_path = Path(root)
    details: list[dict] = []

    if not root_path.exists():
        return {"checked": 0, "errors": 0, "warnings": 0, "details": details}

    json_files = list(root_path.rglob("*.json"))
    checked = 0
    for path in json_files:
        checked += 1
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _add_issue(details, "error", path, "invalid JSON")
            continue
        if not isinstance(raw, dict):
            _add_issue(details, "error", path, "JSON is not an object")
            continue

        _check_config_hash(raw, path, details)
        _check_timestamp_and_versions(raw, path, details)

        artifact_type = _detect_type(raw)
        if artifact_type == "rollout_trace":
            _validate_rollout_trace(raw, path, details, strict)
        elif artifact_type == "packaging_ring":
            _validate_packaging_ring(raw, path, details)
        elif artifact_type == "protocol_horizon":
            _validate_protocol_horizon(raw, path, details)
        elif artifact_type == "metrics_run":
            _validate_metrics_run(raw, path, details, strict)
        else:
            if "config" not in raw and "config_hash" not in raw:
                level = "error" if strict else "warning"
                _add_issue(details, level, path, "untyped artifact; no config/hash")

        if "kernel_P" in raw:
            _validate_kernel_P(raw["kernel_P"], path, details)
        elif "kernel" in raw and isinstance(raw["kernel"], dict) and "P" in raw["kernel"]:
            _validate_kernel_P(raw["kernel"]["P"], path, details)

    errors = sum(1 for d in details if d["level"] == "error")
    warnings = sum(1 for d in details if d["level"] == "warning")
    return {"checked": checked, "errors": errors, "warnings": warnings, "details": details}
