import json
import re

from sbt_agency.repro import stable_hash, write_run_manifest


def test_stable_hash_key_order():
    a = {"a": 1, "b": [2, 3]}
    b = {"b": [2, 3], "a": 1}
    assert stable_hash(a) == stable_hash(b)


def test_stable_hash_differs_on_change():
    a = {"a": 1, "b": [2, 3]}
    b = {"a": 1, "b": [2, 4]}
    assert stable_hash(a) != stable_hash(b)


def test_stable_hash_is_hex():
    h = stable_hash({"b": [2, 3], "a": 1})
    assert re.fullmatch(r"[0-9a-f]{64}", h)


def test_write_run_manifest(tmp_path):
    config = {"a": 1, "b": [2, 3]}
    out_dir = tmp_path / "run"
    manifest = write_run_manifest(out_dir, config, versions={"pkg": "0.0.0"}, notes="ok")

    path = out_dir / "manifest.json"
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("created_at_utc", "config_hash", "config", "versions", "notes"):
        assert key in data
    assert data["config_hash"] == stable_hash(config)
    assert manifest["config_hash"] == data["config_hash"]

