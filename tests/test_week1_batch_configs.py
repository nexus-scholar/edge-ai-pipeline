from __future__ import annotations

from pathlib import Path

import pytest

from edge_al_pipeline.config import load_experiment_config


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/week1_batch/cdgp_week1_batch_cifar10_random.json",
        "configs/week1_batch/cdgp_week1_batch_cifar10_entropy.json",
        "configs/week1_batch/cdgp_week1_batch_cifar10_domain_guided_w02.json",
        "configs/week1_batch/cdgp_week1_batch_cifar10_domain_guided_w05.json",
        "configs/week1_batch/cdgp_week1_batch_cifar10_domain_guided_w08.json",
        "configs/week1_batch/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02.json",
        "configs/week1_batch/cdgp_week1_batch_cifar10_domain_guided_calibrated_w05.json",
        "configs/week1_batch/cdgp_week1_batch_cifar10_domain_guided_calibrated_w08.json",
    ],
)
def test_week1_batch_configs_are_valid(config_path: str):
    config = load_experiment_config(Path(config_path))
    assert config.experiment_name
    assert config.strategy_name
    assert len(config.seeds) >= 5
