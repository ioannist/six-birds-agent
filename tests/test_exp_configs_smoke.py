from dataclasses import asdict

from sbt_agency.env_ring_agent import build_kernel
from sbt_agency.exp_configs import (
    ablations_suite,
    cfg_packaging_ring_off,
    cfg_packaging_ring_on,
    cfg_sweep_noise_maintenance_base,
    sweep_noise_maintenance_axes,
    sweep_noise_maintenance_run_id,
)
from sbt_agency.repro import stable_hash


def test_exp_configs_smoke():
    build_kernel(cfg_packaging_ring_off())
    build_kernel(cfg_packaging_ring_on())

    suite = ablations_suite()
    assert len(suite) >= 7
    for cfg in suite.values():
        build_kernel(cfg)

    p_flip_values, repair_cost_values = sweep_noise_maintenance_axes()
    assert len(p_flip_values) == 8
    assert len(repair_cost_values) == 8


def test_exp_config_hash_locks():
    off = cfg_packaging_ring_off()
    on = cfg_packaging_ring_on()
    assert (
        stable_hash(asdict(off))
        == "a6cb93244f7b9ea35a3becdee4fc19bd7ceea6c5288fa35a145ec2e5f111b035"
    )
    assert (
        stable_hash(asdict(on))
        == "df5dc0d3aeb20f7a90789a55f11bd1d96e6a1c10187ba028b939d631564f4892"
    )

    suite = ablations_suite()
    expected = {
        "full": "6a3a640708dd552b56d217960cc950ce1134ddf896aa0a761aea589362cbbdd3",
        "no_protocol": "c65aaf545518534638f34842585ed24da87fd9c9172897857f5ca93984fff681",
        "no_repair": "d08950d1fa3ed9aaea8a3b81c89b474793a9763e52b5f73ce3cd729d39564414",
        "constraints_off": "6dd35428192b34e65d676d310211abbb4f3851af1c1aac06026e04cee696f47e",
        "learn_on": "a3a79deb23e7b3e4a4e6cdab47af41a048fa7635b4a33b202da6434660613879",
        "high_noise": "accd757af2ba4d36ed8f261321e534bc278603d75a23a8ad6e7dc3056863db89",
        "repair_imperfect": "e8ec106b76f6ca1bfffa513fcf068d045fb9c90198d6929259d44b1e41f62b74",
    }
    for name, cfg in suite.items():
        assert stable_hash(asdict(cfg)) == expected[name]

    assert (
        sweep_noise_maintenance_run_id()
        == "1300e5235fa57af14a29f228b755231f30b57df4da65e032e32372920ae48c45"
    )

