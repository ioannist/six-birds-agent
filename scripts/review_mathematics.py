#!/usr/bin/env python3
"""Reproduce the mathematical review without writing any manuscript assets.

Exact rational checks independently enumerate the ring transition events and
macro fibers. Numerical capacities are bounded by a floating BA duality gap;
these are not interval-arithmetic certificates. The report includes candidate
packaging repairs without replacing the historical experiment configuration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from fractions import Fraction as Q
from itertools import product
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sbt_agency.channel import build_channel_matrix, enumerate_action_seqs
from sbt_agency.empowerment import capacity_bits
from sbt_agency.env_ring_agent import RingAgentConfig, build_kernel
from sbt_agency.exp_configs import (
    ablations_suite, cfg_learning_theta, cfg_packaging_ring_off,
    cfg_packaging_ring_on, cfg_sweep_noise_maintenance_base,
    cfg_packaging_ring_legacy_off, cfg_packaging_ring_legacy_on,
    sweep_noise_maintenance_axes,
)
from sbt_agency.metrics import compute_empowerment_medians_by_theta, compute_ring_metrics
from sbt_agency.packaging import empirical_endomap, idempotence_defect
from sbt_agency.viability import ledger_feasible_actions, post_support_from_kernel, viability_kernel


def coin(p):
    return [(event, prob) for event, prob in [(False, 1 - p), (True, p)] if prob]


def exact_step(cfg, state, action):
    """Independent decimal-rational event-tree interpretation of the ring law."""
    y, u, phi, r, g, theta = state
    costs = dict(LEFT=cfg.cost_left, RIGHT=cfg.cost_right,
                 REPAIR=cfg.cost_repair, LEARN=cfg.cost_learn)
    execute = costs[action] <= r
    move = execute and action in {"LEFT", "RIGHT"}
    slip = max(Q(0), min(Q(1), Q(str(cfg.p_slip)) - theta * Q(str(cfg.slip_improve_per_theta))))
    repair = Q(str(cfg.p_repair)) if execute and action == "REPAIR" else Q(0)
    result = defaultdict(Q)
    for (slipped, ps), (flipped, pf), (repaired, pr) in product(
            coin(slip if move else Q(0)), coin(Q(str(cfg.p_flip))), coin(repair)):
        step = 1 + int(cfg.enable_protocol and phi % 2 == 1)
        direction = (1 if action == "LEFT" else -1) * (1 if u == 0 else -1)
        yp = (y + step * direction) % cfg.L if move and not slipped else y
        up = 0 if repaired else u ^ int(flipped)
        rp = r + (cfg.gain_amount if yp in cfg.gain_positions else 0)
        rp -= cfg.maint_cost + (costs[action] if execute else 0)
        rp = min(cfg.R_max, max(0, rp))
        th = min(cfg.theta_max, theta + 1) if execute and action == "LEARN" else theta
        result[(yp, up, (phi + 1) % cfg.m_phase, rp, g, th)] += ps * pf * pr
    assert sum(result.values()) == 1
    return dict(result)


def macro(cfg, s):
    y, _, phi, r, _, _ = s
    return (phi * (cfg.R_max + 1) + r) * cfg.L + y


def exact_endomap(cfg, states, policy, tau=2):
    fibers = defaultdict(list)
    for s in states:
        fibers[macro(cfg, s)].append(s)
    E, min_gaps, ties, modal_choices = {}, [], [], {}
    for x, fiber in sorted(fibers.items()):
        dist = {s: Q(1, len(fiber)) for s in fiber}
        for _ in range(tau):
            nxt = defaultdict(Q)
            for s, prob in dist.items():
                for sp, conditional in exact_step(cfg, s, policy(s)).items():
                    nxt[sp] += prob * conditional
            dist = nxt
        outputs = defaultdict(Q)
        for s, prob in dist.items():
            outputs[macro(cfg, s)] += prob
        ranked = sorted(outputs, key=lambda label: (-outputs[label], label))
        E[x] = ranked[0]
        modal_choices[x] = [label for label in ranked if outputs[label] == outputs[ranked[0]]]
        gap = outputs[ranked[0]] - outputs[ranked[1]] if len(ranked) > 1 else Q(1)
        min_gaps.append(gap)
        if gap == 0:
            ties.append(x)
    return E, str(min(min_gaps)), ties, modal_choices


def exact_channel(cfg, s0, seq):
    dist = {s0: Q(1)}
    for action in seq:
        nxt = defaultdict(Q)
        for s, p in dist.items():
            for sp, conditional in exact_step(cfg, s, action).items():
                nxt[sp] += p * conditional
        dist = nxt
    out = defaultdict(Q)
    for s, p in dist.items():
        out[s[0]] += p
    return dict(out)


def exact_tv(a, b):
    return sum(abs(a.get(y, Q(0)) - b.get(y, Q(0))) for y in a.keys() | b.keys()) / 2


def packaging_case(cfg, kind):
    kernel, projections, metadata = build_kernel(cfg)
    actions = metadata['action_names']
    states = metadata['state_tuples']

    def policy(s):
        if kind == 'legacy_commands':
            return 'REPAIR' if s[1] == 1 else 'RIGHT'
        if kind == 'feasible_damage_repair':
            return 'REPAIR' if s[1] == 1 and s[3] >= cfg.cost_repair else 'RIGHT'
        if kind == 'funded_preventive_repair':
            return 'REPAIR' if s[3] >= cfg.cost_repair else 'RIGHT'
        if kind == 'idle_control':
            return 'LEARN'  # theta_max=0, zero cost: an explicit lawful idle.
        return 'RIGHT'

    E, gap, ties, modal_choices = exact_endomap(cfg, states, policy)
    # These structural tests establish the extreme defects for every possible
    # tied-mode selector, without enumerating exponentially many selectors.
    if all(x not in choices for x, choices in modal_choices.items()):
        tie_robust_defect = 1.0  # No selectable map can have any fixed point.
    elif all(modal_choices[y] == [y] for choices in modal_choices.values() for y in choices):
        tie_robust_defect = 0.0  # Every selectable image is forced to be fixed.
    else:
        tie_robust_defect = None
    Ef = empirical_endomap(kernel, projections['proj_macro'], 2,
                           lambda s: actions.index(policy(states[s])))
    max_kernel_error = 0.
    for i, s in enumerate(states):
        for a, name in enumerate(actions):
            exact = exact_step(cfg, s, name)
            row = np.array([float(exact.get(t, 0)) for t in states])
            max_kernel_error = max(max_kernel_error, float(np.max(np.abs(row - kernel.P[a, i]))))
    assert max_kernel_error < 1e-14
    costs = dict(LEFT=cfg.cost_left, RIGHT=cfg.cost_right,
                 REPAIR=cfg.cost_repair, LEARN=cfg.cost_learn)
    infeasible = [i for i, s in enumerate(states) if costs[policy(s)] > s[3]]
    feasible = ledger_feasible_actions(range(kernel.n_actions), lambda i: states[i][3],
                                      lambda a: costs[actions[a]])
    coherent_K = viability_kernel(range(kernel.n_states), range(kernel.n_actions), feasible,
                                 post_support_from_kernel(kernel),
                                 lambda i: states[i][3] >= 1 and states[i][1] == 0)
    return dict(config=asdict(cfg), policy=kind, exact_defect=idempotence_defect(E),
                floating_defect=idempotence_defect(Ef), exact_endomap=E,
                floating_map_disagreements=[x for x in E if E[x] != Ef[x]],
                exact_minimum_mode_gap=gap, exact_tie_labels=ties,
                defect_for_every_exact_tie_selector=tie_robust_defect,
                infeasible_policy_state_indices=infeasible,
                coherent_viability_state_indices=sorted(coherent_K),
                max_float_kernel_error=max_kernel_error)


def review():
    off, on = cfg_packaging_ring_legacy_off(), cfg_packaging_ring_legacy_on()
    funded_off, funded_on = cfg_packaging_ring_off(), cfg_packaging_ring_on()
    packaging = {
        'original_off': packaging_case(off, 'right'),
        'original_on_raw_commands': packaging_case(on, 'legacy_commands'),
        'original_on_feasible_policy': packaging_case(on, 'feasible_damage_repair'),
        'restored_funded_off': packaging_case(funded_off, 'right'),
        'restored_funded_on': packaging_case(funded_on, 'funded_preventive_repair'),
        'funded_failed_repair_control': packaging_case(
            replace(funded_on, p_repair=0.0), 'funded_preventive_repair'),
        'idle_without_repair_control': packaging_case(
            replace(off, enable_learn=True, theta_max=0, cost_learn=0), 'idle_control'),
    }
    assert packaging['restored_funded_off']['exact_defect'] == 1
    assert packaging['restored_funded_on']['exact_defect'] == 0
    assert not packaging['restored_funded_on']['infeasible_policy_state_indices']
    assert len(packaging['restored_funded_on']['coherent_viability_state_indices']) == 16
    assert not packaging['restored_funded_off']['coherent_viability_state_indices']
    assert packaging['restored_funded_off']['defect_for_every_exact_tie_selector'] == 1
    assert packaging['restored_funded_on']['defect_for_every_exact_tie_selector'] == 0
    # Exact stochastic and noise-free order witnesses, with enough initial
    # ledger for both moves. No search over infeasible one-move prefixes.
    s0 = (0, 0, 1, 2, 0, 0)
    suite = ablations_suite()
    order = []
    for damage in [0.1, 0.0]:
        a = replace(suite['full'], p_flip=damage)
        b = replace(suite['no_protocol'], p_flip=damage)
        tv_on = exact_tv(exact_channel(a, s0, ['RIGHT', 'LEFT']), exact_channel(a, s0, ['LEFT', 'RIGHT']))
        tv_off = exact_tv(exact_channel(b, s0, ['RIGHT', 'LEFT']), exact_channel(b, s0, ['LEFT', 'RIGHT']))
        order.append(dict(p_flip=damage, initial_state=s0, tv_on=str(tv_on), tv_off=str(tv_off)))
    protocol = {}
    for name in ['full', 'no_protocol']:
        protocol[name] = [compute_ring_metrics(suite[name], empowerment_H=H, packaging_tau=0)
                          for H in range(1, 6)]
    ablations = {name: compute_ring_metrics(cfg) for name, cfg in suite.items()}
    # Compare every sampled sweep channel with the generic channel builder,
    # and record the exact support phase structure at all 64 grid points.
    sweep = []
    from sweep_noise_maintenance import _compute_empowerment_median
    for p_flip in sweep_noise_maintenance_axes()[0]:
        row = []
        for repair_cost in sweep_noise_maintenance_axes()[1]:
            cfg = replace(cfg_sweep_noise_maintenance_base(), p_flip=float(p_flip), cost_repair=repair_cost)
            k, pr, m = build_kernel(cfg)
            states = m['state_tuples']
            costs = [cfg.cost_left, cfg.cost_right, cfg.cost_repair]
            feasible = ledger_feasible_actions(range(3), lambda s: states[s][3], lambda a: costs[a])
            K = viability_kernel(range(k.n_states), range(3), feasible,
                                 post_support_from_kernel(k), lambda s: states[s][3] >= 1 and states[s][1] == 0)
            fast = _compute_empowerment_median(k, m, pr, K, lambda a: costs[a])
            if K:
                from sbt_agency.empowerment import feasible_capacity_bits
                seqs = enumerate_action_seqs([0, 1, 2], 2)
                generic = float(np.median([
                    feasible_capacity_bits(build_channel_matrix(k, s, seqs, pr['proj_y']), seqs,
                                           lambda a: costs[a], states[s][3], tol=1e-6, max_iter=500)
                    for s in sorted(K)[:16]]))
                assert abs(fast - generic) < 1e-10
            row.append(dict(repair_cost=repair_cost, viable_states=len(K), sample_median_bits=fast))
        sweep.append(dict(p_flip=float(p_flip), measurements=row))
    assert all([m['viable_states'] for m in row['measurements']] ==
               [m['viable_states'] for m in sweep[1]['measurements']] for row in sweep[1:])
    return dict(
        schema='agency-mathematics-review-v1',
        source_files_sha256={str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(list((REPO_ROOT / 'src' / 'sbt_agency').glob('*.py'))
                                             + list((REPO_ROOT / 'scripts').glob('*.py'))
                                             + list((REPO_ROOT / 'lean' / 'Agency').glob('*.lean'))
                                             + [REPO_ROOT / 'sbt_agency' / '__init__.py'])},
        scope='finite witnesses and bounded finite-set theorems; paper untouched',
        capacity_reference='https://arxiv.org/html/2407.06013v1#S2.SS1',
        capacity_error='floating lower-upper gap <= 1e-12 nats (sweep: 1e-6); no interval certification',
        packaging=packaging, order_witnesses=order, protocol=protocol, ablations=ablations,
        learning_medians=compute_empowerment_medians_by_theta(cfg_learning_theta()),
        sweep=sweep,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'docs' / 'mathematics-review.json')
    args = parser.parse_args()
    report = review()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)
    for name, case in report['packaging'].items():
        print(f"{name}: exact defect={case['exact_defect']}, infeasible policy states={len(case['infeasible_policy_state_indices'])}")
    print('Order witnesses:', report['order_witnesses'])
    print('Learning medians:', report['learning_medians'])


if __name__ == '__main__':
    main()
