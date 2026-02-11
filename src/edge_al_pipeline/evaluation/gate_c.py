from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from edge_al_pipeline.config import ExperimentConfig
from edge_al_pipeline.experiments.phase3_uncertainty_comparison import (
    run_phase3_uncertainty_comparison,
)
from edge_al_pipeline.experiments.phase3_wgisd_detection import (
    Phase3RunSummary,
    run_phase3_wgisd_detection,
)


@dataclass(frozen=True)
class GateCSeedRow:
    seed: int
    random_run_dir: Path
    uncertainty_run_dir: Path
    random_final_map50_proxy_test: float
    uncertainty_final_map50_proxy_test: float
    delta_final_map50_proxy_test: float
    random_auc_map50_proxy_test: float
    uncertainty_auc_map50_proxy_test: float
    delta_auc_map50_proxy_test: float
    classification_run_dir: Path | None
    localization_run_dir: Path | None
    classification_map50_proxy_test: float
    localization_map50_proxy_test: float
    delta_localization_vs_classification_map50_proxy_test: float
    fp32_run_dir: Path | None
    int8_run_dir: Path | None
    fp32_final_map50_proxy_test: float
    int8_final_map50_proxy_test: float
    delta_int8_minus_fp32_map50_proxy_test: float
    fp32_final_uncertainty_combined: float
    int8_final_uncertainty_combined: float
    delta_int8_minus_fp32_uncertainty_combined: float


@dataclass(frozen=True)
class GateCReport:
    passed: bool
    reason: str
    rows: list[GateCSeedRow]
    minimum_required_improvement: float
    mean_delta_final_map50_proxy_test: float
    mean_delta_auc_map50_proxy_test: float
    mean_delta_localization_vs_classification_map50_proxy_test: float
    mean_delta_int8_minus_fp32_map50_proxy_test: float
    mean_delta_int8_minus_fp32_uncertainty_combined: float
    report_csv_path: Path
    report_json_path: Path
    report_markdown_path: Path
    created_at_utc: str


@dataclass(frozen=True)
class Phase3SeedMetricBundle:
    seed: int
    run_dir: Path
    final_map50_proxy_test: float
    auc_map50_proxy_test: float
    final_uncertainty_combined: float


def run_gate_c_field_validation(
    config: ExperimentConfig,
    config_source: Path | None = None,
    minimum_required_improvement: float = 0.0,
) -> GateCReport:
    _validate_gate_c_inputs(
        config=config,
        minimum_required_improvement=minimum_required_improvement,
    )

    random_config = _clone_phase3_variant(
        base=config,
        variant_name="gate_c_random",
        strategy_name="random",
    )
    uncertainty_config = _clone_phase3_variant(
        base=config,
        variant_name="gate_c_uncertainty",
        strategy_name="entropy",
    )

    random_summary = run_phase3_wgisd_detection(random_config, config_source=config_source)
    uncertainty_summary = run_phase3_wgisd_detection(
        uncertainty_config, config_source=config_source
    )
    random_by_seed = summarize_phase3_runs(random_summary)
    uncertainty_by_seed = summarize_phase3_runs(uncertainty_summary)

    comparison_report = run_phase3_uncertainty_comparison(
        uncertainty_config, config_source=config_source
    )
    loc_vs_cls_by_seed: dict[int, Mapping[str, Any]] = {
        int(item["seed"]): item for item in comparison_report.paired_rows
    }

    fp32_config = _clone_phase3_variant(
        base=uncertainty_config,
        variant_name="gate_c_uncertainty_fp32",
        quantization_mode="fp32",
    )
    int8_config = _clone_phase3_variant(
        base=uncertainty_config,
        variant_name="gate_c_uncertainty_int8",
        quantization_mode="int8",
    )
    fp32_by_seed = summarize_phase3_runs(
        run_phase3_wgisd_detection(fp32_config, config_source=config_source)
    )
    int8_by_seed = summarize_phase3_runs(
        run_phase3_wgisd_detection(int8_config, config_source=config_source)
    )

    rows = build_seed_rows(
        random_by_seed=random_by_seed,
        uncertainty_by_seed=uncertainty_by_seed,
        loc_vs_cls_by_seed=loc_vs_cls_by_seed,
        fp32_by_seed=fp32_by_seed,
        int8_by_seed=int8_by_seed,
    )

    mean_delta_final = _mean([item.delta_final_map50_proxy_test for item in rows])
    mean_delta_auc = _mean([item.delta_auc_map50_proxy_test for item in rows])
    mean_delta_loc_vs_cls = _mean(
        [
            item.delta_localization_vs_classification_map50_proxy_test
            for item in rows
        ]
    )
    mean_delta_int8_fp32_map50 = _mean(
        [item.delta_int8_minus_fp32_map50_proxy_test for item in rows]
    )
    mean_delta_int8_fp32_uncertainty = _mean(
        [item.delta_int8_minus_fp32_uncertainty_combined for item in rows]
    )

    passed = bool(
        rows
        and _is_finite(mean_delta_final)
        and _is_finite(mean_delta_auc)
        and mean_delta_final >= minimum_required_improvement
        and mean_delta_auc >= minimum_required_improvement
    )
    reason = _build_reason(
        passed=passed,
        rows=rows,
        mean_delta_final=mean_delta_final,
        mean_delta_auc=mean_delta_auc,
        minimum_required_improvement=minimum_required_improvement,
    )

    report_root = Path(config.output_root) / "gate_c_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = report_root / f"gate_c_{timestamp}.csv"
    json_path = report_root / f"gate_c_{timestamp}.json"
    md_path = report_root / f"gate_c_{timestamp}.md"
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    _write_rows_csv(csv_path, rows)
    _write_report_json(
        path=json_path,
        rows=rows,
        passed=passed,
        reason=reason,
        minimum_required_improvement=minimum_required_improvement,
        mean_delta_final=mean_delta_final,
        mean_delta_auc=mean_delta_auc,
        mean_delta_loc_vs_cls=mean_delta_loc_vs_cls,
        mean_delta_int8_fp32_map50=mean_delta_int8_fp32_map50,
        mean_delta_int8_fp32_uncertainty=mean_delta_int8_fp32_uncertainty,
        created_at_utc=created_at_utc,
    )
    _write_report_md(
        path=md_path,
        rows=rows,
        passed=passed,
        reason=reason,
        minimum_required_improvement=minimum_required_improvement,
        mean_delta_final=mean_delta_final,
        mean_delta_auc=mean_delta_auc,
        mean_delta_loc_vs_cls=mean_delta_loc_vs_cls,
        mean_delta_int8_fp32_map50=mean_delta_int8_fp32_map50,
        mean_delta_int8_fp32_uncertainty=mean_delta_int8_fp32_uncertainty,
        created_at_utc=created_at_utc,
    )
    return GateCReport(
        passed=passed,
        reason=reason,
        rows=rows,
        minimum_required_improvement=minimum_required_improvement,
        mean_delta_final_map50_proxy_test=mean_delta_final,
        mean_delta_auc_map50_proxy_test=mean_delta_auc,
        mean_delta_localization_vs_classification_map50_proxy_test=mean_delta_loc_vs_cls,
        mean_delta_int8_minus_fp32_map50_proxy_test=mean_delta_int8_fp32_map50,
        mean_delta_int8_minus_fp32_uncertainty_combined=mean_delta_int8_fp32_uncertainty,
        report_csv_path=csv_path,
        report_json_path=json_path,
        report_markdown_path=md_path,
        created_at_utc=created_at_utc,
    )


def summarize_phase3_runs(
    run_summary: Phase3RunSummary,
) -> dict[int, Phase3SeedMetricBundle]:
    by_seed: dict[int, Phase3SeedMetricBundle] = {}
    for item in run_summary.results:
        metric_series = read_metric_series(
            item.run_dir / "metrics.csv",
            metric_name="map50_proxy_test",
        )
        by_seed[item.seed] = Phase3SeedMetricBundle(
            seed=item.seed,
            run_dir=item.run_dir,
            final_map50_proxy_test=_final_metric(metric_series),
            auc_map50_proxy_test=_metric_auc(metric_series),
            final_uncertainty_combined=read_final_uncertainty_combined(
                item.uncertainty_summary_path
            ),
        )
    return by_seed


def build_seed_rows(
    random_by_seed: Mapping[int, Phase3SeedMetricBundle],
    uncertainty_by_seed: Mapping[int, Phase3SeedMetricBundle],
    loc_vs_cls_by_seed: Mapping[int, Mapping[str, Any]],
    fp32_by_seed: Mapping[int, Phase3SeedMetricBundle],
    int8_by_seed: Mapping[int, Phase3SeedMetricBundle],
) -> list[GateCSeedRow]:
    seeds = sorted(set(random_by_seed.keys()) & set(uncertainty_by_seed.keys()))
    rows: list[GateCSeedRow] = []
    for seed in seeds:
        random_metrics = random_by_seed[seed]
        uncertainty_metrics = uncertainty_by_seed[seed]
        loc_vs_cls = loc_vs_cls_by_seed.get(seed, {})
        fp32_metrics = fp32_by_seed.get(seed)
        int8_metrics = int8_by_seed.get(seed)
        fp32_map = (
            fp32_metrics.final_map50_proxy_test if fp32_metrics is not None else math.nan
        )
        int8_map = (
            int8_metrics.final_map50_proxy_test if int8_metrics is not None else math.nan
        )
        fp32_uncertainty = (
            fp32_metrics.final_uncertainty_combined
            if fp32_metrics is not None
            else math.nan
        )
        int8_uncertainty = (
            int8_metrics.final_uncertainty_combined
            if int8_metrics is not None
            else math.nan
        )
        rows.append(
            GateCSeedRow(
                seed=seed,
                random_run_dir=random_metrics.run_dir,
                uncertainty_run_dir=uncertainty_metrics.run_dir,
                random_final_map50_proxy_test=random_metrics.final_map50_proxy_test,
                uncertainty_final_map50_proxy_test=(
                    uncertainty_metrics.final_map50_proxy_test
                ),
                delta_final_map50_proxy_test=(
                    uncertainty_metrics.final_map50_proxy_test
                    - random_metrics.final_map50_proxy_test
                ),
                random_auc_map50_proxy_test=random_metrics.auc_map50_proxy_test,
                uncertainty_auc_map50_proxy_test=(
                    uncertainty_metrics.auc_map50_proxy_test
                ),
                delta_auc_map50_proxy_test=(
                    uncertainty_metrics.auc_map50_proxy_test
                    - random_metrics.auc_map50_proxy_test
                ),
                classification_run_dir=_path_or_none(
                    loc_vs_cls.get("classification_run_dir")
                ),
                localization_run_dir=_path_or_none(
                    loc_vs_cls.get("localization_run_dir")
                ),
                classification_map50_proxy_test=float(
                    loc_vs_cls.get("classification_map50_proxy_test", math.nan)
                ),
                localization_map50_proxy_test=float(
                    loc_vs_cls.get("localization_map50_proxy_test", math.nan)
                ),
                delta_localization_vs_classification_map50_proxy_test=float(
                    loc_vs_cls.get("delta_map50_proxy_test", math.nan)
                ),
                fp32_run_dir=fp32_metrics.run_dir if fp32_metrics is not None else None,
                int8_run_dir=int8_metrics.run_dir if int8_metrics is not None else None,
                fp32_final_map50_proxy_test=fp32_map,
                int8_final_map50_proxy_test=int8_map,
                delta_int8_minus_fp32_map50_proxy_test=int8_map - fp32_map,
                fp32_final_uncertainty_combined=fp32_uncertainty,
                int8_final_uncertainty_combined=int8_uncertainty,
                delta_int8_minus_fp32_uncertainty_combined=(
                    int8_uncertainty - fp32_uncertainty
                ),
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


def read_final_uncertainty_combined(path: Path) -> float:
    if not path.exists():
        return math.nan
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return math.nan
    final_row = max(rows, key=lambda item: int(item["round_index"]))
    return float(final_row.get("mean_uncertainty_combined", math.nan))


def _clone_phase3_variant(
    base: ExperimentConfig,
    variant_name: str,
    strategy_name: str | None = None,
    quantization_mode: str | None = None,
) -> ExperimentConfig:
    payload = base.to_dict()
    payload["experiment_name"] = f"{base.experiment_name}_{variant_name}"
    if strategy_name is not None:
        payload["strategy_name"] = strategy_name
    if quantization_mode is not None:
        payload["quantization_mode"] = quantization_mode
    clone = ExperimentConfig.from_dict(payload)
    clone.validate()
    return clone


def _validate_gate_c_inputs(
    config: ExperimentConfig, minimum_required_improvement: float
) -> None:
    if config.dataset.task not in {"detection", "segmentation"}:
        raise ValueError("Gate C requires detection/segmentation task.")
    if minimum_required_improvement < 0.0:
        raise ValueError("minimum_required_improvement must be >= 0.")


def _metric_auc(series: list[tuple[int, float]]) -> float:
    if not series:
        return math.nan
    values = [value for _, value in series]
    return statistics.fmean(values)


def _final_metric(series: list[tuple[int, float]]) -> float:
    if not series:
        return math.nan
    return series[-1][1]


def _mean(values: list[float]) -> float:
    finite = [value for value in values if _is_finite(value)]
    return statistics.fmean(finite) if finite else math.nan


def _is_finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    return Path(token)


def _build_reason(
    passed: bool,
    rows: list[GateCSeedRow],
    mean_delta_final: float,
    mean_delta_auc: float,
    minimum_required_improvement: float,
) -> str:
    if not rows:
        return "No shared seeds between random and uncertainty runs."
    if not _is_finite(mean_delta_final) or not _is_finite(mean_delta_auc):
        return "Missing map50_proxy_test metrics for random/uncertainty comparison."
    if not passed:
        return (
            "Uncertainty AL did not exceed random baseline at required threshold for "
            "both final and AUC map50_proxy_test."
        )
    if minimum_required_improvement <= 0.0:
        return "Gate C passed."
    return (
        "Gate C passed with uncertainty AL improvement at or above threshold "
        f"({minimum_required_improvement:.4f})."
    )


def _write_rows_csv(path: Path, rows: list[GateCSeedRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "random_run_dir",
                "uncertainty_run_dir",
                "random_final_map50_proxy_test",
                "uncertainty_final_map50_proxy_test",
                "delta_final_map50_proxy_test",
                "random_auc_map50_proxy_test",
                "uncertainty_auc_map50_proxy_test",
                "delta_auc_map50_proxy_test",
                "classification_run_dir",
                "localization_run_dir",
                "classification_map50_proxy_test",
                "localization_map50_proxy_test",
                "delta_localization_vs_classification_map50_proxy_test",
                "fp32_run_dir",
                "int8_run_dir",
                "fp32_final_map50_proxy_test",
                "int8_final_map50_proxy_test",
                "delta_int8_minus_fp32_map50_proxy_test",
                "fp32_final_uncertainty_combined",
                "int8_final_uncertainty_combined",
                "delta_int8_minus_fp32_uncertainty_combined",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row.seed,
                    "random_run_dir": str(row.random_run_dir),
                    "uncertainty_run_dir": str(row.uncertainty_run_dir),
                    "random_final_map50_proxy_test": row.random_final_map50_proxy_test,
                    "uncertainty_final_map50_proxy_test": (
                        row.uncertainty_final_map50_proxy_test
                    ),
                    "delta_final_map50_proxy_test": row.delta_final_map50_proxy_test,
                    "random_auc_map50_proxy_test": row.random_auc_map50_proxy_test,
                    "uncertainty_auc_map50_proxy_test": row.uncertainty_auc_map50_proxy_test,
                    "delta_auc_map50_proxy_test": row.delta_auc_map50_proxy_test,
                    "classification_run_dir": str(row.classification_run_dir or ""),
                    "localization_run_dir": str(row.localization_run_dir or ""),
                    "classification_map50_proxy_test": row.classification_map50_proxy_test,
                    "localization_map50_proxy_test": row.localization_map50_proxy_test,
                    "delta_localization_vs_classification_map50_proxy_test": (
                        row.delta_localization_vs_classification_map50_proxy_test
                    ),
                    "fp32_run_dir": str(row.fp32_run_dir or ""),
                    "int8_run_dir": str(row.int8_run_dir or ""),
                    "fp32_final_map50_proxy_test": row.fp32_final_map50_proxy_test,
                    "int8_final_map50_proxy_test": row.int8_final_map50_proxy_test,
                    "delta_int8_minus_fp32_map50_proxy_test": (
                        row.delta_int8_minus_fp32_map50_proxy_test
                    ),
                    "fp32_final_uncertainty_combined": row.fp32_final_uncertainty_combined,
                    "int8_final_uncertainty_combined": row.int8_final_uncertainty_combined,
                    "delta_int8_minus_fp32_uncertainty_combined": (
                        row.delta_int8_minus_fp32_uncertainty_combined
                    ),
                }
            )


def _write_report_json(
    path: Path,
    rows: list[GateCSeedRow],
    passed: bool,
    reason: str,
    minimum_required_improvement: float,
    mean_delta_final: float,
    mean_delta_auc: float,
    mean_delta_loc_vs_cls: float,
    mean_delta_int8_fp32_map50: float,
    mean_delta_int8_fp32_uncertainty: float,
    created_at_utc: str,
) -> None:
    payload = {
        "created_at_utc": created_at_utc,
        "passed": passed,
        "reason": reason,
        "minimum_required_improvement": minimum_required_improvement,
        "mean_delta_final_map50_proxy_test": mean_delta_final,
        "mean_delta_auc_map50_proxy_test": mean_delta_auc,
        "mean_delta_localization_vs_classification_map50_proxy_test": (
            mean_delta_loc_vs_cls
        ),
        "mean_delta_int8_minus_fp32_map50_proxy_test": mean_delta_int8_fp32_map50,
        "mean_delta_int8_minus_fp32_uncertainty_combined": (
            mean_delta_int8_fp32_uncertainty
        ),
        "rows": [
            {
                "seed": row.seed,
                "random_run_dir": str(row.random_run_dir),
                "uncertainty_run_dir": str(row.uncertainty_run_dir),
                "random_final_map50_proxy_test": row.random_final_map50_proxy_test,
                "uncertainty_final_map50_proxy_test": row.uncertainty_final_map50_proxy_test,
                "delta_final_map50_proxy_test": row.delta_final_map50_proxy_test,
                "random_auc_map50_proxy_test": row.random_auc_map50_proxy_test,
                "uncertainty_auc_map50_proxy_test": row.uncertainty_auc_map50_proxy_test,
                "delta_auc_map50_proxy_test": row.delta_auc_map50_proxy_test,
                "classification_run_dir": str(row.classification_run_dir or ""),
                "localization_run_dir": str(row.localization_run_dir or ""),
                "classification_map50_proxy_test": row.classification_map50_proxy_test,
                "localization_map50_proxy_test": row.localization_map50_proxy_test,
                "delta_localization_vs_classification_map50_proxy_test": (
                    row.delta_localization_vs_classification_map50_proxy_test
                ),
                "fp32_run_dir": str(row.fp32_run_dir or ""),
                "int8_run_dir": str(row.int8_run_dir or ""),
                "fp32_final_map50_proxy_test": row.fp32_final_map50_proxy_test,
                "int8_final_map50_proxy_test": row.int8_final_map50_proxy_test,
                "delta_int8_minus_fp32_map50_proxy_test": (
                    row.delta_int8_minus_fp32_map50_proxy_test
                ),
                "fp32_final_uncertainty_combined": row.fp32_final_uncertainty_combined,
                "int8_final_uncertainty_combined": row.int8_final_uncertainty_combined,
                "delta_int8_minus_fp32_uncertainty_combined": (
                    row.delta_int8_minus_fp32_uncertainty_combined
                ),
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_report_md(
    path: Path,
    rows: list[GateCSeedRow],
    passed: bool,
    reason: str,
    minimum_required_improvement: float,
    mean_delta_final: float,
    mean_delta_auc: float,
    mean_delta_loc_vs_cls: float,
    mean_delta_int8_fp32_map50: float,
    mean_delta_int8_fp32_uncertainty: float,
    created_at_utc: str,
) -> None:
    lines = [
        "# Gate C Report",
        "",
        f"- Status: {'PASS' if passed else 'FAIL'}",
        f"- Reason: {reason}",
        f"- Required improvement (uncertainty-random): {minimum_required_improvement:.6f}",
        f"- Mean delta final map50_proxy_test (uncertainty-random): {mean_delta_final:.6f}",
        f"- Mean delta AUC map50_proxy_test (uncertainty-random): {mean_delta_auc:.6f}",
        (
            "- Mean delta map50_proxy_test "
            f"(localization-classification): {mean_delta_loc_vs_cls:.6f}"
        ),
        (
            "- Mean delta map50_proxy_test (int8-fp32): "
            f"{mean_delta_int8_fp32_map50:.6f}"
        ),
        (
            "- Mean delta uncertainty_combined (int8-fp32): "
            f"{mean_delta_int8_fp32_uncertainty:.6f}"
        ),
        "",
        "## Per-Seed",
        (
            "| Seed | d_final_u-r | d_auc_u-r | d_map50_loc-cls | "
            "d_map50_int8-fp32 | d_unc_int8-fp32 |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.seed} | "
            f"{row.delta_final_map50_proxy_test:.6f} | "
            f"{row.delta_auc_map50_proxy_test:.6f} | "
            f"{row.delta_localization_vs_classification_map50_proxy_test:.6f} | "
            f"{row.delta_int8_minus_fp32_map50_proxy_test:.6f} | "
            f"{row.delta_int8_minus_fp32_uncertainty_combined:.6f} |"
        )
    lines.extend(["", f"- Created at (UTC): {created_at_utc}"])
    path.write_text("\n".join(lines), encoding="utf-8")
