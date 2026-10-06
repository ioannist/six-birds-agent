"""Exact event-tree checks of the restored witness and its scalar limitation."""
import importlib.util
from dataclasses import replace
from pathlib import Path

from sbt_agency.exp_configs import cfg_packaging_ring_off, cfg_packaging_ring_on


def _review_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'review_mathematics.py'
    spec = importlib.util.spec_from_file_location('agency_exact_review', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restored_defect_separation_is_exact_and_independent_of_tie_selector():
    review = _review_module()
    for cfg, kind, expected, viable in [
        (cfg_packaging_ring_off(), 'right', 1., 0),
        (cfg_packaging_ring_on(), 'funded_preventive_repair', 0., 16),
    ]:
        case = review.packaging_case(cfg, kind)
        assert case['exact_defect'] == expected
        assert case['defect_for_every_exact_tie_selector'] == expected
        assert not case['infeasible_policy_state_indices']
        assert not case['floating_map_disagreements']
        assert len(case['coherent_viability_state_indices']) == viable


def test_failed_repair_control_separates_modal_defect_from_coherence():
    review = _review_module()
    case = review.packaging_case(replace(cfg_packaging_ring_on(), p_repair=0.), 'funded_preventive_repair')
    assert case['exact_defect'] == 0.
    assert case['coherent_viability_state_indices'] == []
