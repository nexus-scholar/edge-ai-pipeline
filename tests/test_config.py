from __future__ import annotations

import json

from edge_al_pipeline.config import load_experiment_config, save_experiment_config


def test_config_load_and_round_trip(tmp_path):
    raw = {
        "experiment_name": "phase1_smoke",
        "output_root": "runs",
        "dataset": {
            "name": "fashion_mnist",
            "root": "data/fashion_mnist",
            "version": "1.0",
            "task": "classification",
            "num_classes": 10,
        },
        "model_name": "simple_cnn",
        "model_params": {"batch_size": 32},
        "strategy_name": "entropy",
        "rounds": 2,
        "query_size": 8,
        "seeds": [7],
        "quantization_mode": "fp32",
        "teacher_enabled": False,
        "edge_device": "cpu",
        "bootstrap": {
            "pool_size": 100,
            "initial_labeled_size": 10,
            "val_size": 20,
            "test_size": 10,
        },
    }
    config_path = tmp_path / "exp.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_experiment_config(config_path)
    assert config.experiment_name == "phase1_smoke"
    assert config.model_params["batch_size"] == 32
    assert config.bootstrap.pool_size == 100

    round_trip_path = tmp_path / "exp_out.json"
    save_experiment_config(config, round_trip_path)
    loaded = load_experiment_config(round_trip_path)
    assert loaded.to_dict() == config.to_dict()


def test_detection_config_allows_single_class(tmp_path):
    raw = {
        "experiment_name": "phase3_detection",
        "output_root": "runs",
        "dataset": {
            "name": "wgisd",
            "root": "data/wgisd",
            "version": "1.0",
            "task": "detection",
            "num_classes": 1,
        },
        "model_name": "detector",
        "strategy_name": "entropy",
        "rounds": 1,
        "query_size": 1,
        "seeds": [1],
        "quantization_mode": "fp32",
        "teacher_enabled": True,
        "edge_device": "cpu",
        "bootstrap": {
            "pool_size": 10,
            "initial_labeled_size": 2,
            "val_size": 2,
            "test_size": 2,
        },
    }
    config_path = tmp_path / "det.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_experiment_config(config_path)
    assert config.dataset.num_classes == 1
