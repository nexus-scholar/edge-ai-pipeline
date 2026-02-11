from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edge_al_pipeline.config import ExperimentConfig

if TYPE_CHECKING:
    from edge_al_pipeline.experiments.phase3_wgisd_detection import (
        Phase3RunSummary,
        Phase3SeedResult,
    )


@dataclass(frozen=True)
class Phase3VariantSeedMetrics:
    seed: int
    run_dir: Path
    final_round: int
    loss: float
    precision50_test: float
    recall50_test: float
    map50_proxy_test: float
    mean_uncertainty_classification: float
    mean_uncertainty_localization: float
    mean_uncertainty_combined: float


@dataclass(frozen=True)
class Phase3VariantSummary:
    variant_name: str
    experiment_name: str
    seed_metrics: list[Phase3VariantSeedMetrics]

    def by_seed(self) -> dict[int, Phase3VariantSeedMetrics]:
        return {item.seed: item for item in self.seed_metrics}


@dataclass(frozen=True)
class Phase3ComparisonReport:
    classification_summary: Phase3VariantSummary
    localization_summary: Phase3VariantSummary
    paired_rows: list[dict[str, Any]]
    report_csv_path: Path
    report_json_path: Path
    report_markdown_path: Path
    created_at_utc: str


def run_phase3_uncertainty_comparison(
    config: ExperimentConfig, config_source: Path | None = None
) -> Phase3ComparisonReport:
    from edge_al_pipeline.experiments.phase3_wgisd_detection import (
        run_phase3_wgisd_detection,
    )

    cls_config = _clone_phase3_variant(
        config=config,
        variant_name="classification_only",
        uncertainty_alpha=1.0,
        localization_tta=False,
    )
    loc_config = _clone_phase3_variant(
        config=config,
        variant_name="localization_only",
        uncertainty_alpha=0.0,
        localization_tta=True,
    )

    cls_run_summary = run_phase3_wgisd_detection(cls_config, config_source=config_source)
    loc_run_summary = run_phase3_wgisd_detection(loc_config, config_source=config_source)
    cls_summary = summarize_variant_runs(
        run_summary=cls_run_summary,
        variant_name="classification_only",
        experiment_name=cls_config.experiment_name,
    )
    loc_summary = summarize_variant_runs(
        run_summary=loc_run_summary,
        variant_name="localization_only",
        experiment_name=loc_config.experiment_name,
    )
    paired_rows = build_paired_rows(cls_summary, loc_summary)

    report_root = Path(config.output_root) / "phase3_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = report_root / f"{config.experiment_name}_uncertainty_compare_{timestamp}.csv"
    json_path = report_root / f"{config.experiment_name}_uncertainty_compare_{timestamp}.json"
    md_path = report_root / f"{config.experiment_name}_uncertainty_compare_{timestamp}.md"
    _write_paired_csv(csv_path, paired_rows)

    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "created_at_utc": created_at_utc,
        "classification_summary": _variant_payload(cls_summary),
        "localization_summary": _variant_payload(loc_summary),
        "paired_rows": paired_rows,
        "aggregates": _paired_aggregates(paired_rows),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        _to_markdown(
            cls_summary=cls_summary,
            loc_summary=loc_summary,
            paired_rows=paired_rows,
            created_at_utc=created_at_utc,
        ),
        encoding="utf-8",
    )
    return Phase3ComparisonReport(
        classification_summary=cls_summary,
        localization_summary=loc_summary,
        paired_rows=paired_rows,
        report_csv_path=csv_path,
        report_json_path=json_path,
        report_markdown_path=md_path,
        created_at_utc=created_at_utc,
    )


def summarize_variant_runs(
    run_summary: "Phase3RunSummary", variant_name: str, experiment_name: str
) -> Phase3VariantSummary:
    metrics: list[Phase3VariantSeedMetrics] = []
    for result in run_summary.results:
        final_round, metric_map = read_final_round_metrics(result.run_dir / "metrics.csv")
        uncertainty_map = read_final_uncertainty_metrics(result.uncertainty_summary_path)
        metrics.append(
            Phase3VariantSeedMetrics(
                seed=result.seed,
                run_dir=result.run_dir,
                final_round=final_round,
                loss=float(metric_map.get("loss", math.nan)),
                precision50_test=float(metric_map.get("precision50_test", math.nan)),
                recall50_test=float(metric_map.get("recall50_test", math.nan)),
                map50_proxy_test=float(metric_map.get("map50_proxy_test", math.nan)),
                mean_uncertainty_classification=float(
                    uncertainty_map.get("mean_uncertainty_classification", math.nan)
                ),
                mean_uncertainty_localization=float(
                    uncertainty_map.get("mean_uncertainty_localization", math.nan)
                ),
                mean_uncertainty_combined=float(
                    uncertainty_map.get("mean_uncertainty_combined", math.nan)
                ),
            )
        )
    return Phase3VariantSummary(
        variant_name=variant_name,
        experiment_name=experiment_name,
        seed_metrics=metrics,
    )


def build_paired_rows(
    classification_summary: Phase3VariantSummary,
    localization_summary: Phase3VariantSummary,
) -> list[dict[str, Any]]:
    cls_by_seed = classification_summary.by_seed()
    loc_by_seed = localization_summary.by_seed()
    shared = sorted(set(cls_by_seed.keys()) & set(loc_by_seed.keys()))

    rows: list[dict[str, Any]] = []
    for seed in shared:
        cls = cls_by_seed[seed]
        loc = loc_by_seed[seed]
        rows.append(
            {
                "seed": seed,
                "classification_run_dir": str(cls.run_dir),
                "localization_run_dir": str(loc.run_dir),
                "classification_map50_proxy_test": cls.map50_proxy_test,
                "localization_map50_proxy_test": loc.map50_proxy_test,
                "delta_map50_proxy_test": loc.map50_proxy_test - cls.map50_proxy_test,
                "classification_precision50_test": cls.precision50_test,
                "localization_precision50_test": loc.precision50_test,
                "delta_precision50_test": loc.precision50_test - cls.precision50_test,
                "classification_recall50_test": cls.recall50_test,
                "localization_recall50_test": loc.recall50_test,
                "delta_recall50_test": loc.recall50_test - cls.recall50_test,
                "classification_uncertainty_combined": cls.mean_uncertainty_combined,
                "localization_uncertainty_combined": loc.mean_uncertainty_combined,
                "delta_uncertainty_combined": (
                    loc.mean_uncertainty_combined - cls.mean_uncertainty_combined
                ),
            }
        )
    return rows


def read_final_round_metrics(metrics_path: Path) -> tuple[int, dict[str, float]]:
    rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != "train":
                continue
            rows.append(row)
    if not rows:
        return -1, {}

    final_round = max(int(row["round_index"]) for row in rows)
    metric_map: dict[str, float] = {}
    for row in rows:
        if int(row["round_index"]) != final_round:
            continue
        metric_map[row["metric"]] = float(row["value"])
    return final_round, metric_map


def read_final_uncertainty_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}
    final_row = max(rows, key=lambda item: int(item["round_index"]))
    return {key: float(value) for key, value in final_row.items() if key != "round_index"}


def _clone_phase3_variant(
    config: ExperimentConfig,
    variant_name: str,
    uncertainty_alpha: float,
    localization_tta: bool,
) -> ExperimentConfig:
    payload = config.to_dict()
    payload["experiment_name"] = f"{config.experiment_name}_{variant_name}"
    params = dict(payload.get("model_params", {}))
    params["uncertainty_alpha"] = uncertainty_alpha
    params["localization_tta"] = localization_tta
    payload["model_params"] = params
    clone = ExperimentConfig.from_dict(payload)
    clone.validate()
    return clone


def _write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _variant_payload(summary: Phase3VariantSummary) -> dict[str, Any]:
    return {
        "variant_name": summary.variant_name,
        "experiment_name": summary.experiment_name,
        "seed_metrics": [
            {
                "seed": item.seed,
                "run_dir": str(item.run_dir),
                "final_round": item.final_round,
                "loss": item.loss,
                "precision50_test": item.precision50_test,
                "recall50_test": item.recall50_test,
                "map50_proxy_test": item.map50_proxy_test,
                "mean_uncertainty_classification": item.mean_uncertainty_classification,
                "mean_uncertainty_localization": item.mean_uncertainty_localization,
                "mean_uncertainty_combined": item.mean_uncertainty_combined,
            }
            for item in summary.seed_metrics
        ],
    }


def _paired_aggregates(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_delta_map50_proxy_test": math.nan,
            "mean_delta_precision50_test": math.nan,
            "mean_delta_recall50_test": math.nan,
            "mean_delta_uncertainty_combined": math.nan,
        }
    return {
        "mean_delta_map50_proxy_test": _mean(
            [float(row["delta_map50_proxy_test"]) for row in rows]
        ),
        "mean_delta_precision50_test": _mean(
            [float(row["delta_precision50_test"]) for row in rows]
        ),
        "mean_delta_recall50_test": _mean(
            [float(row["delta_recall50_test"]) for row in rows]
        ),
        "mean_delta_uncertainty_combined": _mean(
            [float(row["delta_uncertainty_combined"]) for row in rows]
        ),
    }


def _to_markdown(
    cls_summary: Phase3VariantSummary,
    loc_summary: Phase3VariantSummary,
    paired_rows: list[dict[str, Any]],
    created_at_utc: str,
) -> str:
    lines = [
        "# Phase 3 Uncertainty Comparison",
        "",
        f"- Classification experiment: `{cls_summary.experiment_name}`",
        f"- Localization experiment: `{loc_summary.experiment_name}`",
        f"- Shared seeds: {len(paired_rows)}",
        "",
        "## Mean Deltas (Localization - Classification)",
    ]
    aggregates = _paired_aggregates(paired_rows)
    lines.append(f"- map50_proxy_test: {aggregates['mean_delta_map50_proxy_test']:.6f}")
    lines.append(f"- precision50_test: {aggregates['mean_delta_precision50_test']:.6f}")
    lines.append(f"- recall50_test: {aggregates['mean_delta_recall50_test']:.6f}")
    lines.append(
        f"- uncertainty_combined: {aggregates['mean_delta_uncertainty_combined']:.6f}"
    )
    lines.extend(
        [
            "",
            "## Per-Seed Side-by-Side",
            "| Seed | cls_map50 | loc_map50 | delta_map50 | cls_precision | loc_precision | cls_recall | loc_recall |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| "
            f"{row['seed']} | "
            f"{float(row['classification_map50_proxy_test']):.6f} | "
            f"{float(row['localization_map50_proxy_test']):.6f} | "
            f"{float(row['delta_map50_proxy_test']):.6f} | "
            f"{float(row['classification_precision50_test']):.6f} | "
            f"{float(row['localization_precision50_test']):.6f} | "
            f"{float(row['classification_recall50_test']):.6f} | "
            f"{float(row['localization_recall50_test']):.6f} |"
        )
    lines.extend(["", f"- Created at (UTC): {created_at_utc}"])
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan
