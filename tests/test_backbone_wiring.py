from __future__ import annotations

import pytest

from edge_al_pipeline.config import BootstrapConfig, DatasetConfig, ExperimentConfig
from edge_al_pipeline.experiments.phase1b_cifar10 import (
    _runner_config_from_experiment as phase1b_runner_config_from_experiment,
)
from edge_al_pipeline.experiments.phase2_agri_classification import (
    _runner_config_from_experiment as phase2_runner_config_from_experiment,
)
from edge_al_pipeline.models.cifar10_runner import _build_classifier as build_cifar_classifier
from edge_al_pipeline.models.image_folder_mobilenet_runner import (
    _build_classifier as build_imagefolder_classifier,
)


def test_phase1b_runner_config_defaults_to_simple_cnn_backbone():
    config = ExperimentConfig(
        experiment_name="phase1b",
        output_root="runs",
        dataset=DatasetConfig(
            name="cifar10",
            root="data/cifar10",
            version="1.0",
            task="classification",
            num_classes=10,
        ),
        model_name="simple_cnn_cifar10",
        strategy_name="entropy",
        rounds=1,
        query_size=1,
        seeds=[1],
        quantization_mode="fp32",
        teacher_enabled=False,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=20,
            initial_labeled_size=2,
            val_size=2,
            test_size=2,
        ),
    )
    runner_config = phase1b_runner_config_from_experiment(config)
    assert runner_config.backbone_name == "simple_cnn"
    assert runner_config.image_size == 32


def test_phase1b_runner_config_honors_backbone_name_override():
    config = ExperimentConfig(
        experiment_name="phase1b_resnet",
        output_root="runs",
        dataset=DatasetConfig(
            name="cifar10",
            root="data/cifar10",
            version="1.0",
            task="classification",
            num_classes=10,
        ),
        model_name="simple_cnn_cifar10",
        model_params={"backbone_name": "resnet18"},
        strategy_name="entropy",
        rounds=1,
        query_size=1,
        seeds=[1],
        quantization_mode="fp32",
        teacher_enabled=False,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=20,
            initial_labeled_size=2,
            val_size=2,
            test_size=2,
        ),
    )
    runner_config = phase1b_runner_config_from_experiment(config)
    assert runner_config.backbone_name == "resnet18"
    assert runner_config.image_size == 224


def test_phase2_runner_config_uses_model_name_as_backbone_fallback():
    config = ExperimentConfig(
        experiment_name="phase2",
        output_root="runs",
        dataset=DatasetConfig(
            name="fruits360",
            root="data/fruits360",
            version="1.0",
            task="classification",
            num_classes=2,
        ),
        model_name="mobilenet_v3_small",
        strategy_name="entropy",
        rounds=1,
        query_size=1,
        seeds=[1],
        quantization_mode="fp32",
        teacher_enabled=False,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=20,
            initial_labeled_size=2,
            val_size=2,
            test_size=2,
        ),
    )
    runner_config = phase2_runner_config_from_experiment(config, num_classes=2)
    assert runner_config.backbone_name == "mobilenet_v3_small"


def test_phase1b_runner_config_accepts_mobilenet_v4_name():
    config = ExperimentConfig(
        experiment_name="phase1b_mobilenetv4",
        output_root="runs",
        dataset=DatasetConfig(
            name="cifar10",
            root="data/cifar10",
            version="1.0",
            task="classification",
            num_classes=10,
        ),
        model_name="simple_cnn_cifar10",
        model_params={"backbone_name": "mobilenet_v4"},
        strategy_name="entropy",
        rounds=1,
        query_size=1,
        seeds=[1],
        quantization_mode="fp32",
        teacher_enabled=False,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=20,
            initial_labeled_size=2,
            val_size=2,
            test_size=2,
        ),
    )
    runner_config = phase1b_runner_config_from_experiment(config)
    assert runner_config.backbone_name == "mobilenet_v4"


def test_mobilenet_v4_fails_fast_without_backend_support():
    with pytest.raises(ValueError, match="MobileNetV4"):
        build_cifar_classifier(
            backbone_name="mobilenet_v4",
            num_classes=10,
            embedding_dim=64,
            pretrained_backbone=True,
        )
    with pytest.raises(ValueError, match="MobileNetV4"):
        build_imagefolder_classifier(
            num_classes=10,
            backbone_name="mobilenet_v4",
            pretrained_backbone=True,
        )
