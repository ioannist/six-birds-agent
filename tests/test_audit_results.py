import json
from pathlib import Path

import numpy as np

from sbt_agency.audit import audit_results
from sbt_agency.repro import stable_hash


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_results_strict(tmp_path):
    root = tmp_path / "results"

    config = {"a": 1}
    config_hash = stable_hash(config)

    metrics = {
        "config_hash": config_hash,
        "n_states": 2,
        "n_actions": 1,
        "kernel_size_viable": 1,
        "empowerment_median_on_K": 0.0,
        "idempotence_defect": 0.0,
    }
    metrics_payload = {
        "config": config,
        "config_hash": config_hash,
        "metrics": metrics,
        "timestamp": "now",
        "versions": {"python": "3"},
    }
    _write_json(root / "metrics.json", metrics_payload)

    rollout_payload = {
        "config": config,
        "config_hash": config_hash,
        "seed": 0,
        "n_steps": 2,
        "action_names": ["A"],
        "runs": {
            "pi": [
                {"t": 0, "s": 0, "state": [0], "a": 0, "s_next": 0, "state_next": [0]},
                {"t": 1, "s": 0, "state": [0], "a": 0, "s_next": 0, "state_next": [0]},
            ]
        },
        "timestamp": "now",
        "versions": {"python": "3"},
    }
    _write_json(root / "trace.json", rollout_payload)

    config_off = {"x": 1}
    config_on = {"x": 2}
    packaging_payload = {
        "config_off": config_off,
        "config_on": config_on,
        "config_hash_off": stable_hash(config_off),
        "config_hash_on": stable_hash(config_on),
        "tau_list": [1, 2],
        "defect_off": [0.0, 1.0],
        "defect_on": [0.0, 0.0],
        "timestamp": "now",
        "versions": {"python": "3"},
    }
    _write_json(root / "packaging.json", packaging_payload)

    protocol_payload = {
        "config_on": config_on,
        "config_off": config_off,
        "config_hash_on": stable_hash(config_on),
        "config_hash_off": stable_hash(config_off),
        "H_list": [1, 2],
        "emp_on": [0.1, 0.2],
        "emp_off": [0.0, 0.0],
        "created_at_utc": "now",
        "versions": {"python": "3"},
    }
    _write_json(root / "protocol.json", protocol_payload)

    result = audit_results(root, strict=True)
    assert result["checked"] == 4
    assert result["errors"] == 0
    assert result["warnings"] == 0
