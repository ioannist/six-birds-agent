"""Adversarial checks of the mathematical domains and the fixed-point bridge."""
from itertools import product

import numpy as np
import pytest

from sbt_agency.channel import build_channel_matrix
from sbt_agency.empowerment import feasible_capacity_bits
from sbt_agency.env_ring_agent import RingAgentConfig, build_kernel
from sbt_agency.kernel import FiniteKernel
from sbt_agency.packaging import empirical_endomap, idempotence_defect
from sbt_agency.viability import (
    ledger_feasible_actions, post_support_from_kernel,
    viability_kernel, viability_kernel_history,
)


def test_viability_exhaustive_two_state_systems():
    # An independent oracle: enumerate all safe controlled-invariant subsets.
    # Covers every nonempty support, state-dependent action gate and safe set
    # on two states and two actions, including empty viability domains.
    subsets = [set(), {0}, {1}, {0, 1}]
    for supports in product(subsets[1:], repeat=4):
        post = lambda s, a: supports[2 * s + a]
        for allowed in product(subsets, repeat=2):
            feasible = lambda s: allowed[s]
            for safe_set in subsets:
                invariant_sets = [
                    S for S in subsets if S <= safe_set and all(
                        any(post(s, a) <= S for a in allowed[s]) for s in S
                    )
                ]
                expected = set().union(*invariant_sets)
                hist = viability_kernel_history([0, 1], [0, 1], feasible, post, safe_set.__contains__)
                assert hist[-1] == expected
                assert len(hist) - 2 <= len(safe_set)
                assert all(b <= a for a, b in zip(hist, hist[1:]))


def test_viability_checks_declared_action_alphabet():
    with pytest.raises(ValueError, match="action alphabet"):
        viability_kernel([0], [0], lambda _: [1], lambda s, a: {0}, lambda _: True)


def test_support_keeps_arbitrarily_small_positive_successors():
    P = np.array([[[1.0 - 1e-14, 1e-14], [0., 1.]]])
    kernel = FiniteKernel(P)
    post = post_support_from_kernel(kernel)
    assert post(0, 0) == {0, 1}
    assert viability_kernel([0, 1], [0], lambda _: [0], post, lambda s: s == 0) == set()
    with pytest.raises(ValueError, match="all successors"):
        post_support_from_kernel(kernel, atol=1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_feasibility_rejects_invalid_costs_and_budgets(value):
    with pytest.raises(ValueError):
        ledger_feasible_actions([0], lambda _: 1., lambda _: value)(0)
    with pytest.raises(ValueError):
        feasible_capacity_bits(np.ones((1, 1)), [(0,)], lambda _: value, 1.)
    with pytest.raises(ValueError):
        feasible_capacity_bits(np.ones((1, 1)), [(0,)], lambda _: 0., value)


def test_exact_budget_gate_does_not_grant_tolerance_credit():
    assert ledger_feasible_actions([0], lambda _: 0., lambda _: 1e-14)(0) == []
    W = np.eye(2)
    assert feasible_capacity_bits(W, [(0,), (1,)], lambda a: a * 1e-14, 0.) == 0.


@pytest.mark.parametrize("P", [np.zeros((1, 0, 0)), np.zeros((0, 1, 1)),
                               np.array([[[float("inf"), -float("inf")], [0., 1.]]]),
                               np.array([[[1., -1e-14], [0., 1.]]])])
def test_kernel_validation_requires_actual_probability_laws(P):
    with pytest.raises(ValueError):
        FiniteKernel(P).validate()


def test_packaging_rejects_a_lens_that_silently_discards_outputs():
    P = np.array([[[0., 1.], [0., 1.]]])
    with pytest.raises(ValueError, match="projection image"):
        empirical_endomap(FiniteKernel(P), lambda s: s, 1, lambda s: 0, macro_labels=[0])


def test_output_lens_rejects_fractional_labels():
    with pytest.raises(ValueError, match="integer"):
        build_channel_matrix(FiniteKernel(np.eye(2)[None]), 0, [(0,)], lambda s: s / 2)


@pytest.mark.parametrize("kwargs", [{"p_flip": float("nan")}, {"p_repair": 1.1},
                                    {"cost_left": -1}, {"cost_right": 0.5},
                                    {"m_phase": 0}, {"gain_positions": (8,)}])
def test_ring_configuration_rejects_values_outside_its_domain(kwargs):
    with pytest.raises(ValueError):
        RingAgentConfig(**kwargs)


def test_identity_off_removes_identity_sectors():
    cfg = RingAgentConfig(L=2, m_phase=1, R_max=0, theta_max=0, g_size=3, identity_on=False)
    k, _, metadata = build_kernel(cfg)
    assert k.n_states == 4
    assert metadata['dims']['g_size'] == 1
    assert all(t[4] == 0 for t in metadata['state_tuples'])


def test_mode_idempotence_does_not_establish_stochastic_label_invariance():
    k = FiniteKernel(np.array([[[0.6, 0.4], [0.4, 0.6]]]))
    E = empirical_endomap(k, lambda s: s, 1, lambda s: 0)
    assert idempotence_defect(E) == 0.
    assert post_support_from_kernel(k)(0, 0) == {0, 1}


def test_empowerment_on_K_does_not_require_safe_interventions():
    # From either safe state, the idle action maintains safety. A second
    # affordable action exposes an unsafe output, giving positive empowerment.
    P = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [0., 1.]]])
    k = FiniteKernel(P)
    K = viability_kernel([0, 1], [0, 1], lambda s: [0, 1],
                         post_support_from_kernel(k), lambda s: s == 0)
    assert K == {0}
    W = build_channel_matrix(k, 0, [(0,), (1,)], lambda s: s)
    assert feasible_capacity_bits(W, [(0,), (1,)], lambda a: 0., 0.) == 1.
    assert feasible_capacity_bits(W[:1], [(0,)], lambda a: 0., 0.) == 0.


def test_empty_successor_support_is_not_an_infinite_safe_stochastic_move():
    with pytest.raises(ValueError, match="nonempty"):
        viability_kernel([0], [0], lambda s: [0], lambda s, a: set(), lambda s: True)
