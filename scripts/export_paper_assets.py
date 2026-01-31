#!/usr/bin/env python3
"""Export stable-named paper assets from results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sbt_agency.channel import build_channel_matrix
from sbt_agency.env_ring_agent import build_kernel
from sbt_agency.exp_configs import (
    ablations_suite,
    cfg_learning_theta,
    cfg_packaging_ring_off,
    cfg_packaging_ring_on,
    sweep_noise_maintenance_run_id,
)
from sbt_agency.repro import stable_hash


def _copy_or_fail(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Missing source asset: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _export_figures() -> list[Path]:
    figures_dir = REPO_ROOT / "paper" / "figures"

    hash_off = stable_hash(asdict(cfg_packaging_ring_off()))
    hash_on = stable_hash(asdict(cfg_packaging_ring_on()))
    packaging_src = (
        REPO_ROOT
        / "results"
        / "packaging"
        / f"packaging_ring_{hash_off}_{hash_on}.png"
    )
    packaging_dst = figures_dir / "fig_packaging_ring.png"
    _copy_or_fail(packaging_src, packaging_dst)

    suite = ablations_suite()
    hash_protocol_on = stable_hash(asdict(suite["full"]))
    hash_protocol_off = stable_hash(asdict(suite["no_protocol"]))
    protocol_src = (
        REPO_ROOT
        / "results"
        / "protocol"
        / f"protocol_horizon_{hash_protocol_on}_{hash_protocol_off}.png"
    )
    protocol_dst = figures_dir / "fig_protocol_horizon.png"
    _copy_or_fail(protocol_src, protocol_dst)

    run_id = sweep_noise_maintenance_run_id()
    sweep_k_src = (
        REPO_ROOT
        / "results"
        / "sweeps"
        / f"noise_maintenance_K_{run_id}.png"
    )
    sweep_e_src = (
        REPO_ROOT
        / "results"
        / "sweeps"
        / f"noise_maintenance_E_{run_id}.png"
    )
    sweep_k_dst = figures_dir / "fig_sweep_K.png"
    sweep_e_dst = figures_dir / "fig_sweep_E.png"
    _copy_or_fail(sweep_k_src, sweep_k_dst)
    _copy_or_fail(sweep_e_src, sweep_e_dst)

    hash_learning = stable_hash(asdict(cfg_learning_theta()))
    learning_src = (
        REPO_ROOT
        / "results"
        / "learning"
        / f"learning_theta_{hash_learning}.png"
    )
    learning_dst = figures_dir / "fig_learning_theta.png"
    _copy_or_fail(learning_src, learning_dst)

    return [
        packaging_dst,
        protocol_dst,
        sweep_k_dst,
        sweep_e_dst,
        learning_dst,
    ]


def _export_ablations_table() -> Path:
    summary_path = REPO_ROOT / "results" / "ablations" / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing ablations summary: {summary_path}")

    rows = []
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda r: r["name"])

    out_path = REPO_ROOT / "paper" / "generated" / "ablations_summary.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(val: str) -> str:
        try:
            num = float(val)
        except ValueError:
            return val
        if val.isdigit():
            return val
        return f"{num:.3f}"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("name & kernel size viable & empowerment median on K & idempotence defect \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            name = row["name"].replace("_", " ")
            f.write(
                f"{name} & {_fmt(row['kernel_size_viable'])} & "
                f"{_fmt(row['empowerment_median_on_K'])} & {_fmt(row['idempotence_defect'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    return out_path


def _select_holonomy_state(
    kernel_on: object,
    projections_on: dict,
    metadata_on: dict,
    kernel_off: object,
    projections_off: dict,
    metadata_off: dict,
    seqs_on: list[tuple[int, int]],
    seqs_off: list[tuple[int, int]],
) -> tuple[int, int, tuple[int, int, int, int, int, int], float, float]:
    state_tuples = metadata_on["state_tuples"]
    tuple_to_state_off = metadata_off["tuple_to_state"]
    best: tuple[tuple[float, float, int], int, int, tuple[int, int, int, int, int, int], float, float] | None = None

    for idx_on, (y, u, phi, r, g, theta) in enumerate(state_tuples):
        if r < 1 or u != 0:
            continue
        idx_off = tuple_to_state_off[(y, u, phi, r, g, theta)]
        W_on = build_channel_matrix(kernel_on, idx_on, seqs_on, projections_on["proj_y"])
        tvd_on = 0.5 * float(np.abs(W_on[0] - W_on[1]).sum())
        W_off = build_channel_matrix(kernel_off, idx_off, seqs_off, projections_off["proj_y"])
        tvd_off = 0.5 * float(np.abs(W_off[0] - W_off[1]).sum())
        key = (tvd_off, -tvd_on, idx_on)
        if best is None or key < best[0]:
            best = (key, idx_on, idx_off, (y, u, phi, r, g, theta), tvd_on, tvd_off)

    if best is None:
        raise SystemExit("No witness state with r>=1 and u==0 found.")

    _, idx_on, idx_off, state_tuple, tvd_on, tvd_off = best
    return idx_on, idx_off, state_tuple, tvd_on, tvd_off


def _export_holonomy_witness() -> dict:
    suite = ablations_suite()

    kernel_on, projections_on, metadata_on = build_kernel(suite["full"])
    action_names_on = metadata_on["action_names"]
    right_on = action_names_on.index("RIGHT")
    left_on = action_names_on.index("LEFT")
    seq_alpha_on = (right_on, left_on)
    seq_beta_on = (left_on, right_on)
    seqs_on = [seq_alpha_on, seq_beta_on]

    kernel_off, projections_off, metadata_off = build_kernel(suite["no_protocol"])
    action_names_off = metadata_off["action_names"]
    right_off = action_names_off.index("RIGHT")
    left_off = action_names_off.index("LEFT")
    seq_alpha_off = (right_off, left_off)
    seq_beta_off = (left_off, right_off)
    seqs_off = [seq_alpha_off, seq_beta_off]

    s_idx_on, s_idx_off, state_tuple, tvd_on, tvd_off = _select_holonomy_state(
        kernel_on,
        projections_on,
        metadata_on,
        kernel_off,
        projections_off,
        metadata_off,
        seqs_on,
        seqs_off,
    )

    y, u, phi, r, _g, _theta = state_tuple
    witness_tex = (
        "\\[\n"
        "\\mathrm{TV}\\!\\left(W_{\\alpha},W_{\\beta}\\right)\n"
        "=\n"
        "\\begin{cases}\n"
        f"{tvd_on:.4f} & \\text{{protocol ON}},\\\\\n"
        f"{tvd_off:.4f} & \\text{{protocol OFF}},\n"
        "\\end{cases}\n"
        "\\qquad\n"
        f"s^\\star:\\ (y={int(y)},\\ r={int(r)},\\ \\phi={int(phi)},\\ u={int(u)}),\\ \n"
        "\\alpha=(\\mathrm{R},\\mathrm{L}),\\ \\beta=(\\mathrm{L},\\mathrm{R}).\n"
        "\\]\n"
    )

    out_path = REPO_ROOT / "paper" / "generated" / "holonomy_witness.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(witness_tex, encoding="utf-8")

    return {
        "path": out_path,
        "tvd_on": tvd_on,
        "tvd_off": tvd_off,
        "state": {"y": int(y), "r": int(r), "phi": int(phi), "u": int(u)},
    }


def _export_numbers_snapshot() -> Path:
    hash_off = stable_hash(asdict(cfg_packaging_ring_off()))
    hash_on = stable_hash(asdict(cfg_packaging_ring_on()))
    packaging_json = (
        REPO_ROOT
        / "results"
        / "packaging"
        / f"packaging_ring_{hash_off}_{hash_on}.json"
    )
    if not packaging_json.exists():
        raise SystemExit(f"Missing packaging JSON: {packaging_json}")
    packaging = json.loads(packaging_json.read_text(encoding="utf-8"))
    tau_list = packaging["tau_list"]
    if 2 not in tau_list:
        raise SystemExit("tau=2 not found in packaging JSON")
    idx = tau_list.index(2)
    defect_off_tau2 = packaging["defect_off"][idx]
    defect_on_tau2 = packaging["defect_on"][idx]

    suite = ablations_suite()
    protocol_hash_on = stable_hash(asdict(suite["full"]))
    protocol_hash_off = stable_hash(asdict(suite["no_protocol"]))
    protocol_json = (
        REPO_ROOT
        / "results"
        / "protocol"
        / f"protocol_horizon_{protocol_hash_on}_{protocol_hash_off}.json"
    )
    if not protocol_json.exists():
        raise SystemExit(f"Missing protocol JSON: {protocol_json}")
    protocol = json.loads(protocol_json.read_text(encoding="utf-8"))

    run_id = sweep_noise_maintenance_run_id()
    sweep_npz = REPO_ROOT / "results" / "sweeps" / f"noise_maintenance_{run_id}.npz"
    if not sweep_npz.exists():
        raise SystemExit(f"Missing sweep NPZ: {sweep_npz}")
    sweep = np.load(sweep_npz, allow_pickle=False)
    K_size = sweep["K_size"]
    emp_median = sweep["emp_median"]
    sweep_metric_minmax = {
        "K_size_min": float(K_size.min()),
        "K_size_max": float(K_size.max()),
        "emp_median_min": float(emp_median.min()),
        "emp_median_max": float(emp_median.max()),
    }

    hash_learning = stable_hash(asdict(cfg_learning_theta()))
    learning_json = (
        REPO_ROOT
        / "results"
        / "learning"
        / f"learning_theta_{hash_learning}.json"
    )
    if not learning_json.exists():
        raise SystemExit(f"Missing learning JSON: {learning_json}")
    learning = json.loads(learning_json.read_text(encoding="utf-8"))
    medians = learning.get("medians_by_theta", {})

    holonomy_witness = _export_holonomy_witness()

    numbers = {
        "hash_packaging_off": hash_off,
        "hash_packaging_on": hash_on,
        "defect_off_tau2": defect_off_tau2,
        "defect_on_tau2": defect_on_tau2,
        "protocol_hash_on": protocol_hash_on,
        "protocol_hash_off": protocol_hash_off,
        "protocol_H_list": protocol["H_list"],
        "protocol_emp_on": protocol["emp_on"],
        "protocol_emp_off": protocol["emp_off"],
        "sweep_run_id": run_id,
        "sweep_metric_minmax": sweep_metric_minmax,
        "learning_hash_cfg": hash_learning,
        "learning_medians_theta0": medians.get("0"),
        "learning_medians_theta1": medians.get("1"),
        "learning_medians_theta2": medians.get("2"),
        "holonomy_witness_tvd_on": holonomy_witness["tvd_on"],
        "holonomy_witness_tvd_off": holonomy_witness["tvd_off"],
        "holonomy_witness_state": holonomy_witness["state"],
    }

    out_path = REPO_ROOT / "paper" / "generated" / "numbers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    if not args.no_run:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_all_experiments.py"), "--clean"],
            check=True,
        )

    figures = _export_figures()
    table_path = _export_ablations_table()
    numbers_path = _export_numbers_snapshot()

    for path in figures:
        print(f"figure: {path}")
    print(f"table: {table_path}")
    print(f"numbers: {numbers_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
