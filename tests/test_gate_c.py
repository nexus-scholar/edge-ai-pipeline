from __future__ import annotations

import csv
from pathlib import Path

from edge_al_pipeline.evaluation.gate_c import (
    Phase3SeedMetricBundle,
    build_seed_rows,
    read_final_uncertainty_combined,
    read_metric_series,
)


def test_read_metric_series_filters_metric_and_split(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "round_index,seed,split,metric,value",
                "0,1,train,map50_proxy_test,0.10",
                "0,1,train,loss,1.20",
                "1,1,val,map50_proxy_test,0.30",
                "1,1,train,map50_proxy_test,0.25",
            ]
        ),
        encoding="utf-8",
    )
    series = read_metric_series(metrics_path, "map50_proxy_test")
    assert series == [(0, 0.10), (1, 0.25)]


def test_read_final_uncertainty_combined_picks_last_round(tmp_path):
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
                "mean_uncertainty_classification": 0.5,
                "mean_uncertainty_localization": 0.2,
                "mean_uncertainty_combined": 0.35,
            }
        )
        writer.writerow(
            {
                "round_index": 3,
                "selected_count": 4,
                "mean_uncertainty_classification": 0.4,
                "mean_uncertainty_localization": 0.1,
                "mean_uncertainty_combined": 0.25,
            }
        )
    assert read_final_uncertainty_combined(path) == 0.25


def test_build_seed_rows_joins_all_sections():
    random_by_seed = {
        7: Phase3SeedMetricBundle(
            seed=7,
            run_dir=Path("runs/random"),
            final_map50_proxy_test=0.20,
            auc_map50_proxy_test=0.15,
            final_uncertainty_combined=0.60,
        )
    }
    uncertainty_by_seed = {
        7: Phase3SeedMetricBundle(
            seed=7,
            run_dir=Path("runs/uncertainty"),
            final_map50_proxy_test=0.30,
            auc_map50_proxy_test=0.25,
            final_uncertainty_combined=0.55,
        )
    }
    fp32_by_seed = {
        7: Phase3SeedMetricBundle(
            seed=7,
            run_dir=Path("runs/fp32"),
            final_map50_proxy_test=0.33,
            auc_map50_proxy_test=0.24,
            final_uncertainty_combined=0.52,
        )
    }
    int8_by_seed = {
        7: Phase3SeedMetricBundle(
            seed=7,
            run_dir=Path("runs/int8"),
            final_map50_proxy_test=0.29,
            auc_map50_proxy_test=0.20,
            final_uncertainty_combined=0.58,
        )
    }
    loc_vs_cls_by_seed = {
        7: {
            "classification_run_dir": "runs/cls",
            "localization_run_dir": "runs/loc",
            "classification_map50_proxy_test": 0.28,
            "localization_map50_proxy_test": 0.31,
            "delta_map50_proxy_test": 0.03,
        }
    }

    rows = build_seed_rows(
        random_by_seed=random_by_seed,
        uncertainty_by_seed=uncertainty_by_seed,
        loc_vs_cls_by_seed=loc_vs_cls_by_seed,
        fp32_by_seed=fp32_by_seed,
        int8_by_seed=int8_by_seed,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.seed == 7
    assert round(row.delta_final_map50_proxy_test, 6) == 0.10
    assert round(row.delta_auc_map50_proxy_test, 6) == 0.10
    assert round(row.delta_localization_vs_classification_map50_proxy_test, 6) == 0.03
    assert round(row.delta_int8_minus_fp32_map50_proxy_test, 6) == -0.04
    assert round(row.delta_int8_minus_fp32_uncertainty_combined, 6) == 0.06
