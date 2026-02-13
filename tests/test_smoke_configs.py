from __future__ import annotations

from pathlib import Path

import pytest

from edge_al_pipeline.config import load_experiment_config


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/smoke/phase1b_cifar10_smoke_strategy_random.json",
        "configs/smoke/phase1b_cifar10_smoke_strategy_entropy.json",
        "configs/smoke/phase1b_cifar10_smoke_strategy_domain_guided.json",
        "configs/smoke/phase1b_cifar10_smoke_backbone_mobilenet_v3_small.json",
        "configs/smoke/phase1b_cifar10_smoke_backbone_mobilenet_v3_large.json",
        "configs/smoke/phase1b_cifar10_smoke_backbone_resnet18.json",
        "configs/smoke/phase1b_cifar10_smoke_backbone_resnet50.json",
        "configs/smoke/phase1b_cifar10_smoke_backbone_mobilenet_v4.json",
        "configs/cdgp_week1_toy_cifar10_random.json",
        "configs/cdgp_week1_toy_cifar10_entropy.json",
        "configs/cdgp_week1_toy_cifar10_domain_guided.json",
    ],
)
def test_new_smoke_and_toy_configs_are_valid(config_path: str):
    path = Path(config_path)
    config = load_experiment_config(path)
    assert config.experiment_name
    assert config.strategy_name
