from sbt_agency.exp_configs import cfg_learning_theta
from sbt_agency.metrics import compute_empowerment_medians_by_theta


def test_learning_theta_gap():
    cfg = cfg_learning_theta()
    medians = compute_empowerment_medians_by_theta(
        cfg,
        safe_r_min=1,
        empowerment_H=2,
        restrict_u=0,
        restrict_phi=0,
        action_subset=("LEFT", "RIGHT"),
    )
    assert medians[2] >= medians[0] + 0.2

