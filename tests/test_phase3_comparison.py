from __future__ import annotations

import csv

from edge_al_pipeline.config import BootstrapConfig, DatasetConfig, ExperimentConfig
from edge_al_pipeline.experiments.phase3_uncertainty_comparison import (
    build_paired_rows,
    read_final_round_metrics,
    read_final_uncertainty_metrics,
)


def test_read_final_round_metrics(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "round_index,seed,split,metric,value",
                "0,1,train,loss,1.2",
                "0,1,train,map50_proxy_test,0.20",
                "1,1,train,loss,0.8",
                "1,1,train,map50_proxy_test,0.35",
            ]
        ),
        encoding="utf-8",
    )
    final_round, metric_map = read_final_round_metrics(metrics)
    assert final_round == 1
    assert metric_map["loss"] == 0.8
    assert metric_map["map50_proxy_test"] == 0.35


def test_read_final_uncertainty_metrics(tmp_path):
    path = tmp_path / "uncertainty_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "round_index",
                "selected_count",
                "mean_uncertainty_classification",
                "mean_uncertainty_localization",
                "mean_uncertainty_combined",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "round_index": 0,
                "selected_count": 4,
                "mean_uncertainty_classification": 0.6,
                "mean_uncertainty_localization": 0.3,
                "mean_uncertainty_combined": 0.45,
            }
        )
        writer.writerow(
            {
                "round_index": 2,
                "selected_count": 5,
                "mean_uncertainty_classification": 0.7,
                "mean_uncertainty_localization": 0.4,
                "mean_uncertainty_combined": 0.55,
            }
        )
    metrics = read_final_uncertainty_metrics(path)
    assert metrics["mean_uncertainty_combined"] == 0.55
    assert metrics["selected_count"] == 5.0


def test_build_paired_rows():
    from edge_al_pipeline.experiments.phase3_uncertainty_comparison import (
        Phase3VariantSeedMetrics,
        Phase3VariantSummary,
    )

    cls = Phase3VariantSummary(
        variant_name="classification_only",
        experiment_name="cls_exp",
        seed_metrics=[
            Phase3VariantSeedMetrics(
                seed=1,
                run_dir=tmp_path_placeholder(),
                final_round=1,
                loss=1.0,
                precision50_test=0.5,
                recall50_test=0.4,
                map50_proxy_test=0.45,
                mean_uncertainty_classification=0.7,
                mean_uncertainty_localization=0.2,
                mean_uncertainty_combined=0.7,
            )
        ],
    )
    loc = Phase3VariantSummary(
        variant_name="localization_only",
        experiment_name="loc_exp",
        seed_metrics=[
            Phase3VariantSeedMetrics(
                seed=1,
                run_dir=tmp_path_placeholder(),
                final_round=1,
                loss=1.1,
                precision50_test=0.55,
                recall50_test=0.45,
                map50_proxy_test=0.50,
                mean_uncertainty_classification=0.6,
                mean_uncertainty_localization=0.5,
                mean_uncertainty_combined=0.5,
            )
        ],
    )
    rows = build_paired_rows(cls, loc)
    assert len(rows) == 1
    assert round(float(rows[0]["delta_map50_proxy_test"]), 6) == 0.05


def tmp_path_placeholder():
    from pathlib import Path

    return Path("runs/dummy")


def test_detection_single_class_config_validation():
    config = ExperimentConfig(
        experiment_name="phase3",
        output_root="runs",
        dataset=DatasetConfig(
            name="wgisd",
            root="data/wgisd",
            version="1.0",
            task="detection",
            num_classes=1,
        ),
        model_name="detector",
        strategy_name="entropy",
        rounds=1,
        query_size=1,
        seeds=[1],
        quantization_mode="fp32",
        teacher_enabled=True,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=10,
            initial_labeled_size=2,
            val_size=2,
            test_size=2,
        ),
    )
    config.validate()
