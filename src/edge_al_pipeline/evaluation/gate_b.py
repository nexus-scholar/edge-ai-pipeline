from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edge_al_pipeline.config import ExperimentConfig
from edge_al_pipeline.experiments.phase2_agri_classification import (
    Phase2RunSummary,
    Phase2SeedResult,
    run_phase2_agri_classification,
)


@dataclass(frozen=True)
class GateBSeedRow:
    seed: int
    pretrain_backbone_path: Path
    imagenet_run_dir: Path
    agri_transfer_run_dir: Path
    imagenet_final_test_accuracy: float
    agri_final_test_accuracy: float
    delta_final_test_accuracy: float
    imagenet_auc_test_accuracy: float
    agri_auc_test_accuracy: float
    delta_auc_test_accuracy: float


@dataclass(frozen=True)
class GateBReport:
    passed: bool
    reason: str
    rows: list[GateBSeedRow]
    mean_delta_final_test_accuracy: float
    mean_delta_auc_test_accuracy: float
    report_csv_path: Path
    report_json_path: Path
    report_markdown_path: Path
    created_at_utc: str


def run_gate_b_transfer(
    pretrain_config: ExperimentConfig,
    transfer_config: ExperimentConfig,
    pretrain_config_source: Path | None = None,
    transfer_config_source: Path | None = None,
) -> GateBReport:
    _validate_gate_b_inputs(pretrain_config=pretrain_config, transfer_config=transfer_config)

    pretrain_summary = run_phase2_agri_classification(
        pretrain_config, config_source=pretrain_config_source
    )
    pretrain_by_seed = {item.seed: item for item in pretrain_summary.results}
    shared_seeds = sorted(set(pretrain_by_seed.keys()) & set(transfer_config.seeds))
    if not shared_seeds:
        raise ValueError("No shared seeds between pretrain and transfer configs for Gate B.")

    imagenet_config = _clone_transfer_variant(
        base=transfer_config,
        variant_name="imagenet_init",
        seeds=shared_seeds,
        model_param_updates={
            "pretrained_backbone": True,
            "backbone_checkpoint": None,
        },
    )
    imagenet_summary = run_phase2_agri_classification(
        imagenet_config, config_source=transfer_config_source
    )
    imagenet_by_seed = {item.seed: item for item in imagenet_summary.results}

    agri_by_seed: dict[int, Phase2SeedResult] = {}
    for seed in shared_seeds:
        checkpoint = pretrain_by_seed[seed].backbone_path
        transfer_variant = _clone_transfer_variant(
            base=transfer_config,
            variant_name=f"agri_transfer_seed{seed}",
            seeds=[seed],
            model_param_updates={
                "pretrained_backbone": False,
                "backbone_checkpoint": str(checkpoint),
            },
        )
        summary = run_phase2_agri_classification(
            transfer_variant, config_source=transfer_config_source
        )
        agri_by_seed[seed] = summary.results[0]

    rows = _build_seed_rows(
        shared_seeds=shared_seeds,
        pretrain_by_seed=pretrain_by_seed,
        imagenet_by_seed=imagenet_by_seed,
        agri_by_seed=agri_by_seed,
    )
    mean_delta_final = statistics.fmean([row.delta_final_test_accuracy for row in rows])
    mean_delta_auc = statistics.fmean([row.delta_auc_test_accuracy for row in rows])
    passed = mean_delta_final > 0.0 and mean_delta_auc > 0.0
    reason = (
        "Agri-pretrained transfer improves convergence and final test accuracy."
        if passed
        else (
            "Agri-pretrained transfer did not improve both convergence (AUC) "
            "and final test accuracy versus ImageNet initialization."
        )
    )

    report_root = Path(transfer_config.output_root) / "gate_b_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = report_root / f"gate_b_{timestamp}.csv"
    json_path = report_root / f"gate_b_{timestamp}.json"
    md_path = report_root / f"gate_b_{timestamp}.md"
    _write_rows_csv(csv_path, rows)
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    _write_report_json(
        path=json_path,
        rows=rows,
        passed=passed,
        reason=reason,
        mean_delta_final=mean_delta_final,
        mean_delta_auc=mean_delta_auc,
        created_at_utc=created_at_utc,
    )
    _write_report_md(
        path=md_path,
        rows=rows,
        passed=passed,
        reason=reason,
        mean_delta_final=mean_delta_final,
        mean_delta_auc=mean_delta_auc,
        created_at_utc=created_at_utc,
    )
    return GateBReport(
        passed=passed,
        reason=reason,
        rows=rows,
        mean_delta_final_test_accuracy=mean_delta_final,
        mean_delta_auc_test_accuracy=mean_delta_auc,
        report_csv_path=csv_path,
        report_json_path=json_path,
        report_markdown_path=md_path,
        created_at_utc=created_at_utc,
    )


def _build_seed_rows(
    shared_seeds: list[int],
    pretrain_by_seed: dict[int, Phase2SeedResult],
    imagenet_by_seed: dict[int, Phase2SeedResult],
    agri_by_seed: dict[int, Phase2SeedResult],
) -> list[GateBSeedRow]:
    rows: list[GateBSeedRow] = []
    for seed in shared_seeds:
        pretrain = pretrain_by_seed[seed]
        imagenet = imagenet_by_seed[seed]
        agri = agri_by_seed[seed]

        imagenet_series = read_metric_series(imagenet.run_dir / "metrics.csv", "test_accuracy")
        agri_series = read_metric_series(agri.run_dir / "metrics.csv", "test_accuracy")
        imagenet_final = _final_metric(imagenet_series)
        agri_final = _final_metric(agri_series)
        imagenet_auc = _metric_auc(imagenet_series)
        agri_auc = _metric_auc(agri_series)
        rows.append(
            GateBSeedRow(
                seed=seed,
                pretrain_backbone_path=pretrain.backbone_path,
                imagenet_run_dir=imagenet.run_dir,
                agri_transfer_run_dir=agri.run_dir,
                imagenet_final_test_accuracy=imagenet_final,
                agri_final_test_accuracy=agri_final,
                delta_final_test_accuracy=agri_final - imagenet_final,
                imagenet_auc_test_accuracy=imagenet_auc,
                agri_auc_test_accuracy=agri_auc,
                delta_auc_test_accuracy=agri_auc - imagenet_auc,
            )
        )
    return rows


def read_metric_series(metrics_path: Path, metric_name: str) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != "train":
                continue
            if row.get("metric") != metric_name:
                continue
            series.append((int(row["round_index"]), float(row["value"])))
    return sorted(series, key=lambda item: item[0])


def _metric_auc(series: list[tuple[int, float]]) -> float:
    if not series:
        return math.nan
    values = [value for _, value in series]
    return statistics.fmean(values)


def _final_metric(series: list[tuple[int, float]]) -> float:
    if not series:
        return math.nan
    return series[-1][1]


def _clone_transfer_variant(
    base: ExperimentConfig,
    variant_name: str,
    seeds: list[int],
    model_param_updates: dict[str, Any],
) -> ExperimentConfig:
    payload = base.to_dict()
    payload["experiment_name"] = f"{base.experiment_name}_{variant_name}"
    payload["seeds"] = seeds
    params = dict(payload.get("model_params", {}))
    params.update(model_param_updates)
    payload["model_params"] = params
    clone = ExperimentConfig.from_dict(payload)
    clone.validate()
    return clone


def _validate_gate_b_inputs(
    pretrain_config: ExperimentConfig, transfer_config: ExperimentConfig
) -> None:
    if pretrain_config.dataset.task != "classification":
        raise ValueError("Gate B pretrain config must be classification.")
    if transfer_config.dataset.task != "classification":
        raise ValueError("Gate B transfer config must be classification.")
    if pretrain_config.teacher_enabled or transfer_config.teacher_enabled:
        raise ValueError("Gate B does not support teacher mode.")


def _write_rows_csv(path: Path, rows: list[GateBSeedRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "pretrain_backbone_path",
                "imagenet_run_dir",
                "agri_transfer_run_dir",
                "imagenet_final_test_accuracy",
                "agri_final_test_accuracy",
                "delta_final_test_accuracy",
                "imagenet_auc_test_accuracy",
                "agri_auc_test_accuracy",
                "delta_auc_test_accuracy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row.seed,
                    "pretrain_backbone_path": str(row.pretrain_backbone_path),
                    "imagenet_run_dir": str(row.imagenet_run_dir),
                    "agri_transfer_run_dir": str(row.agri_transfer_run_dir),
                    "imagenet_final_test_accuracy": row.imagenet_final_test_accuracy,
                    "agri_final_test_accuracy": row.agri_final_test_accuracy,
                    "delta_final_test_accuracy": row.delta_final_test_accuracy,
                    "imagenet_auc_test_accuracy": row.imagenet_auc_test_accuracy,
                    "agri_auc_test_accuracy": row.agri_auc_test_accuracy,
                    "delta_auc_test_accuracy": row.delta_auc_test_accuracy,
                }
            )


def _write_report_json(
    path: Path,
    rows: list[GateBSeedRow],
    passed: bool,
    reason: str,
    mean_delta_final: float,
    mean_delta_auc: float,
    created_at_utc: str,
) -> None:
    payload = {
        "created_at_utc": created_at_utc,
        "passed": passed,
        "reason": reason,
        "mean_delta_final_test_accuracy": mean_delta_final,
        "mean_delta_auc_test_accuracy": mean_delta_auc,
        "rows": [
            {
                "seed": row.seed,
                "pretrain_backbone_path": str(row.pretrain_backbone_path),
                "imagenet_run_dir": str(row.imagenet_run_dir),
                "agri_transfer_run_dir": str(row.agri_transfer_run_dir),
                "imagenet_final_test_accuracy": row.imagenet_final_test_accuracy,
                "agri_final_test_accuracy": row.agri_final_test_accuracy,
                "delta_final_test_accuracy": row.delta_final_test_accuracy,
                "imagenet_auc_test_accuracy": row.imagenet_auc_test_accuracy,
                "agri_auc_test_accuracy": row.agri_auc_test_accuracy,
                "delta_auc_test_accuracy": row.delta_auc_test_accuracy,
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_report_md(
    path: Path,
    rows: list[GateBSeedRow],
    passed: bool,
    reason: str,
    mean_delta_final: float,
    mean_delta_auc: float,
    created_at_utc: str,
) -> None:
    lines = [
        "# Gate B Report",
        "",
        f"- Status: {'PASS' if passed else 'FAIL'}",
        f"- Reason: {reason}",
        f"- Mean delta final test accuracy (agri - imagenet): {mean_delta_final:.6f}",
        f"- Mean delta AUC test accuracy (agri - imagenet): {mean_delta_auc:.6f}",
        "",
        "## Per-Seed Comparison",
        "| Seed | ImageNet final | Agri final | Delta final | ImageNet AUC | Agri AUC | Delta AUC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.seed} | "
            f"{row.imagenet_final_test_accuracy:.6f} | "
            f"{row.agri_final_test_accuracy:.6f} | "
            f"{row.delta_final_test_accuracy:.6f} | "
            f"{row.imagenet_auc_test_accuracy:.6f} | "
            f"{row.agri_auc_test_accuracy:.6f} | "
            f"{row.delta_auc_test_accuracy:.6f} |"
        )
    lines.extend(["", f"- Created at (UTC): {created_at_utc}"])
    path.write_text("\n".join(lines), encoding="utf-8")
