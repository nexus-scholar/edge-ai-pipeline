from __future__ import annotations

from pathlib import Path

import pytest

from edge_al_pipeline.config import load_experiment_config


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/cloud/cdgp_toy_colab.json",
        "configs/cloud/cdgp_clean_colab.json",
        "configs/cloud/cdgp_clean_kaggle.json",
        "configs/cloud/cdgp_week1_batch_cifar10_random_kaggle.json",
        "configs/cloud/cdgp_week1_batch_cifar10_entropy_kaggle.json",
        "configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02_kaggle.json",
        "configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w05_kaggle.json",
        "configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w08_kaggle.json",
    ],
)
def test_cloud_configs_are_valid(config_path: str):
    config = load_experiment_config(Path(config_path))
    assert config.experiment_name
    assert config.model_name
    assert config.strategy_name
