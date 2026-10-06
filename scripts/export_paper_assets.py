#!/usr/bin/env python3
"""Export stable-named paper assets (figures, tables, numbers) from results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sbt_agency.channel import build_channel_matrix
from sbt_agency.env_ring_agent import RingAgentConfig, build_kernel
from sbt_agency.exp_configs import (
    ablations_suite,
    cfg_learning_theta,
    cfg_packaging_ring_off,
    cfg_packaging_ring_on,
    sweep_noise_maintenance_run_id,
)
from sbt_agency.metrics import _cost_by_action_name
from sbt_agency.packaging import empirical_endomap, idempotence_defect
from sbt_agency.repro import stable_hash
from sbt_agency.viability import ledger_feasible_actions, post_support_from_kernel, viability_kernel

FIG_DIR = REPO_ROOT / "paper" / "figures"
GEN_DIR = REPO_ROOT / "paper" / "generated"

# Reference categorical slots (blue, orange) and a one-hue blue ramp.
C_ON = "#2a78d6"
C_OFF = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "mathtext.fontset": "cm",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    }
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing results file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _packaging_json() -> dict:
    hash_off = stable_hash(asdict(cfg_packaging_ring_off()))
    hash_on = stable_hash(asdict(cfg_packaging_ring_on()))
    return _load_json(REPO_ROOT / "results" / "packaging" / f"packaging_ring_{hash_off}_{hash_on}.json")


def _protocol_json() -> dict:
    suite = ablations_suite()
    h_on = stable_hash(asdict(suite["full"]))
    h_off = stable_hash(asdict(suite["no_protocol"]))
    return _load_json(REPO_ROOT / "results" / "protocol" / f"protocol_horizon_{h_on}_{h_off}.json")


def _learning_json() -> dict:
    h = stable_hash(asdict(cfg_learning_theta()))
    return _load_json(REPO_ROOT / "results" / "learning" / f"learning_theta_{h}.json")


def _sweep_npz():
    path = REPO_ROOT / "results" / "sweeps" / f"noise_maintenance_{sweep_noise_maintenance_run_id()}.npz"
    if not path.exists():
        raise SystemExit(f"Missing sweep NPZ: {path}")
    return np.load(path, allow_pickle=False)


def _style_axes(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig_packaging() -> Path:
    data = _packaging_json()
    tau = data["tau_list"]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.plot([t - 0.13 for t in tau], data["defect_off"], color=C_OFF, lw=0, marker="s", ms=6,
            markeredgecolor="white", markeredgewidth=0.8, label="no repair (always RIGHT)")
    ax.plot([t + 0.13 for t in tau], data["defect_on"], color=C_ON, lw=0, marker="o", ms=6.5,
            markeredgecolor="white", markeredgewidth=0.8, label="funded preventive repair")
    for t in tau:
        if t % 2 == 0:
            ax.axvspan(t - 0.35, t + 0.35, color="#f0efec", zorder=0, lw=0)
    ax.set_xticks(tau)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel(r"packaging horizon $\tau$ (shaded: full phase cycles)")
    ax.set_ylabel(r"idempotence defect $\mathrm{Def}(E)$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3)
    _style_axes(ax)
    out = FIG_DIR / "fig_packaging.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _fig_protocol() -> Path:
    data = _protocol_json()
    H = data["H_list"]
    on, off = data["emp_on"], data["emp_off"]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(H, on, color=C_ON, lw=1.6, marker="o", ms=5, markeredgecolor="white",
            markeredgewidth=0.8, label="protocol on (phase-dependent step)")
    ax.plot(H, off, color=C_OFF, lw=1.6, marker="o", ms=5, markeredgecolor="white",
            markeredgewidth=0.8, label="protocol off")
    ax.annotate(f"{on[-1]:.2f}", (H[-1], on[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=8)
    ax.annotate(f"{off[-1]:.2f}", (H[-1], off[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=8)
    ax.annotate(f"equal at $H=1$: {on[0]:.2f}", (H[0], on[0]), xytext=(8, -14),
                textcoords="offset points", color=INK_2, fontsize=8)
    ax.set_xticks(H)
    ax.set_xlim(0.7, H[-1] + 0.6)
    ax.set_ylim(0.8, 2.05)
    ax.set_xlabel(r"horizon $H$ (steps)")
    ax.set_ylabel("median empowerment (bits)")
    ax.legend(loc="upper left")
    _style_axes(ax)
    out = FIG_DIR / "fig_protocol_horizon.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _fig_learning() -> Path:
    data = _learning_json()
    med = data["medians_by_theta"]
    thetas = sorted(int(k) for k in med)
    vals = [med[str(t)] for t in thetas]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(thetas, vals, color=C_ON, lw=1.6, marker="o", ms=6, markeredgecolor="white",
            markeredgewidth=0.8)
    cfg = cfg_learning_theta()
    for t, v in zip(thetas, vals):
        slip = max(0.0, cfg.p_slip - t * cfg.slip_improve_per_theta)
        ax.annotate(f"{v:.2f} bits\nslip {slip:.2f}", (t, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(thetas)
    ax.set_xlim(-0.4, thetas[-1] + 0.4)
    ax.set_ylim(0.6, 1.75)
    ax.set_xlabel(r"skill level $\theta$")
    ax.set_ylabel("median empowerment (bits)")
    _style_axes(ax)
    out = FIG_DIR / "fig_learning_theta.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _fig_sweep() -> Path:
    sweep = _sweep_npz()
    p_vals = sweep["p_flip_values"]
    c_vals = sweep["repair_cost_values"]
    K = sweep["K_size"]
    E = sweep["emp_median"]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    panels = [
        (axes[0], K, r"(a) viable states $|\mathcal{K}|$", lambda v: f"{int(round(v))}", 0, max(1.0, K.max())),
        (axes[1], E, "(b) median empowerment (bits)", lambda v: f"{v:.2f}", 0, max(1e-9, E.max())),
    ]
    for ax, Z, title, fmt, vmin, vmax in panels:
        ax.imshow(Z, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                frac = (v - vmin) / (vmax - vmin)
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6.5,
                        color="white" if frac > 0.55 else INK)
        ax.axhline(0.5, color=INK, lw=1.0)
        ax.set_xticks(range(len(c_vals)))
        ax.set_xticklabels([str(int(c)) for c in c_vals])
        ax.set_yticks(range(len(p_vals)))
        ax.set_yticklabels([f"{p:.1f}" for p in p_vals])
        ax.set_xlabel("repair cost")
        ax.set_title(title, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    axes[0].set_ylabel(r"damage probability $p_{\mathrm{flip}}$")
    out = FIG_DIR / "fig_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _export_figures() -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return [_fig_packaging(), _fig_protocol(), _fig_sweep(), _fig_learning()]


ABLATION_ROWS = [
    ("full", "Reference model"),
    ("no_protocol", "Protocol off (phase-independent step)"),
    ("no_repair", "REPAIR action removed"),
    ("repair_imperfect", r"Imperfect repair (success $0.5$)"),
    ("high_noise", r"Higher noise (flip $0.4$, slip $0.4$)"),
    ("constraints_off", r"Free movement (move cost $0$)"),
    ("constraints_off_no_repair", r"Free movement and REPAIR removed"),
    ("learn_on", r"LEARN action added ($\theta\in\{0,1\}$)"),
]


def _export_ablations_table() -> Path:
    summary_path = REPO_ROOT / "results" / "ablations" / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing ablations summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        rows = {row["name"]: row for row in csv.DictReader(f)}
    missing = [name for name, _ in ABLATION_ROWS if name not in rows]
    if missing or len(rows) != len(ABLATION_ROWS):
        raise SystemExit(f"Ablation rows out of sync with the table layout: {sorted(rows)}")

    lines = [
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Configuration & States & $|\\K|$ & Median $\\Emp_{\\mathrm{feas}}$ (bits) & $\\mathrm{Def}(E)$ \\\\",
        "\\midrule",
    ]
    for name, label in ABLATION_ROWS:
        row = rows[name]
        lines.append(
            f"{label} & {int(row['n_states'])} & {int(row['kernel_size_viable'])} & "
            f"{float(row['empowerment_median_on_K']):.3f} & {float(row['idempotence_defect']):.3f} \\\\"
        )
        if name == "full":
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path = GEN_DIR / "ablations_summary.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _coherent_viable_count(cfg: RingAgentConfig, kernel, metadata) -> int:
    states = metadata["state_tuples"]
    names = metadata["action_names"]
    feasible = ledger_feasible_actions(
        range(kernel.n_actions),
        lambda s: states[s][3],
        lambda a: _cost_by_action_name(cfg, names[a]),
    )
    K = viability_kernel(
        range(kernel.n_states),
        range(kernel.n_actions),
        feasible,
        post_support_from_kernel(kernel),
        lambda s: states[s][3] >= 1 and states[s][1] == 0,
    )
    return len(K)


def _packaging_controls() -> list[dict]:
    """Funded packaging witness plus the two controls from the mathematics review."""
    off = cfg_packaging_ring_off()
    on = cfg_packaging_ring_on()
    cases = [
        ("No repair: always move RIGHT", off, "right"),
        ("Funded preventive repair", on, "repair"),
        ("Same policy, repair always fails", replace(on, p_repair=0.0), "repair"),
        ("Idle without repair", replace(off, enable_learn=True, theta_max=0, cost_learn=0), "idle"),
    ]
    out = []
    for label, cfg, kind in cases:
        kernel, proj, md = build_kernel(cfg)
        names = md["action_names"]
        states = md["state_tuples"]

        def policy(s: int, kind=kind, names=names, states=states, cfg=cfg) -> int:
            if kind == "right":
                return names.index("RIGHT")
            if kind == "idle":
                return names.index("LEARN")  # theta_max=0 and zero cost: a lawful idle step
            return names.index("REPAIR") if states[s][3] >= cfg.cost_repair else names.index("RIGHT")

        E = empirical_endomap(kernel, proj["proj_macro"], 2, policy)
        out.append(
            {
                "label": label,
                "config_hash": stable_hash(asdict(cfg)),
                "defect_tau2": float(idempotence_defect(E)),
                "coherent_viable": _coherent_viable_count(cfg, kernel, md),
                "n_states": kernel.n_states,
            }
        )
    return out


def _export_packaging_table(controls: list[dict]) -> Path:
    lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "Regime (funded substrate) & $\\mathrm{Def}(E)$ at $\\tau=2$ & coherent $|\\K|$ \\\\",
        "\\midrule",
    ]
    for c in controls:
        lines.append(f"{c['label']} & {c['defect_tau2']:.0f} & {c['coherent_viable']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path = GEN_DIR / "packaging_controls.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


WITNESS_STATE = (0, 0, 1, 2, 0, 0)  # (y, u, phi, r, g, theta)


def _order_tv(cfg: RingAgentConfig) -> float:
    kernel, proj, md = build_kernel(cfg)
    names = md["action_names"]
    R, L = names.index("RIGHT"), names.index("LEFT")
    s = md["tuple_to_state"][WITNESS_STATE]
    W = build_channel_matrix(kernel, s, [(R, L), (L, R)], proj["proj_y"])
    return 0.5 * float(np.abs(W[0] - W[1]).sum())


def _export_holonomy_witness() -> dict:
    suite = ablations_suite()
    noisy = {"on": _order_tv(suite["full"]), "off": _order_tv(suite["no_protocol"])}
    clean = {
        "on": _order_tv(replace(suite["full"], p_flip=0.0)),
        "off": _order_tv(replace(suite["no_protocol"], p_flip=0.0)),
    }
    p_flip = suite["full"].p_flip
    lines = [
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Damage noise & protocol on & protocol off \\\\",
        "\\midrule",
        f"$p_{{\\mathrm{{flip}}}}={p_flip:g}$ (as in the exhibit) & {noisy['on']:.3f} & {noisy['off']:.3f} \\\\",
        f"$p_{{\\mathrm{{flip}}}}=0$ (matched, noise-free damage) & {clean['on']:.3f} & {clean['off']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    out_path = GEN_DIR / "holonomy_witness.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    y, u, phi, r, _g, _theta = WITNESS_STATE
    return {
        "path": out_path,
        "tvd_noisy": noisy,
        "tvd_noise_free": clean,
        "state": {"y": y, "u": u, "phi": phi, "r": r},
    }


def _export_numbers_snapshot(controls: list[dict], witness: dict) -> Path:
    packaging = _packaging_json()
    tau_list = packaging["tau_list"]
    idx = tau_list.index(2)

    protocol = _protocol_json()
    sweep = _sweep_npz()
    learning = _learning_json()
    medians = learning.get("medians_by_theta", {})

    with (REPO_ROOT / "results" / "ablations" / "summary.csv").open("r", encoding="utf-8") as f:
        ablations = {
            row["name"]: {
                "config_hash": row["config_hash"],
                "n_states": int(row["n_states"]),
                "kernel_size_viable": int(row["kernel_size_viable"]),
                "empowerment_median_on_K": float(row["empowerment_median_on_K"]),
                "idempotence_defect": float(row["idempotence_defect"]),
            }
            for row in csv.DictReader(f)
        }

    numbers = {
        "hash_packaging_off": packaging["config_hash_off"],
        "hash_packaging_on": packaging["config_hash_on"],
        "defect_off_tau2": packaging["defect_off"][idx],
        "defect_on_tau2": packaging["defect_on"][idx],
        "coherent_K_off": len(packaging["coherent_K_off"]),
        "coherent_K_on": len(packaging["coherent_K_on"]),
        "packaging_controls": controls,
        "protocol_hash_on": protocol["config_hash_on"],
        "protocol_hash_off": protocol["config_hash_off"],
        "protocol_H_list": protocol["H_list"],
        "protocol_emp_on": protocol["emp_on"],
        "protocol_emp_off": protocol["emp_off"],
        "ablations": ablations,
        "sweep_run_id": sweep_noise_maintenance_run_id(),
        "sweep_p_flip_values": [float(p) for p in sweep["p_flip_values"]],
        "sweep_repair_cost_values": [int(c) for c in sweep["repair_cost_values"]],
        "sweep_K_size": sweep["K_size"].tolist(),
        "sweep_emp_median": sweep["emp_median"].tolist(),
        "learning_hash_cfg": stable_hash(asdict(cfg_learning_theta())),
        "learning_medians_theta0": medians.get("0"),
        "learning_medians_theta1": medians.get("1"),
        "learning_medians_theta2": medians.get("2"),
        "holonomy_witness_state": witness["state"],
        "holonomy_witness_tvd_noisy": witness["tvd_noisy"],
        "holonomy_witness_tvd_noise_free": witness["tvd_noise_free"],
    }
    out_path = GEN_DIR / "numbers.json"
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

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    figures = _export_figures()
    table_path = _export_ablations_table()
    controls = _packaging_controls()
    controls_path = _export_packaging_table(controls)
    witness = _export_holonomy_witness()
    numbers_path = _export_numbers_snapshot(controls, witness)

    for path in figures:
        print(f"figure: {path}")
    print(f"table: {table_path}")
    print(f"table: {controls_path}")
    print(f"table: {witness['path']}")
    print(f"numbers: {numbers_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
