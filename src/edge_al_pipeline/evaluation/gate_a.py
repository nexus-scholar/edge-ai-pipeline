from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from edge_al_pipeline.config import ExperimentConfig

if TYPE_CHECKING:
    from edge_al_pipeline.experiments.phase1_fashion_mnist import Phase1RunSummary

_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class SeedBudgetResult:
    seed: int
    run_dir: Path
    round_index: int
    labeled_count: int
    target_labeled_count: int
    target_reached: bool
    test_accuracy: float


@dataclass(frozen=True)
class StrategySummary:
    strategy_name: str
    seed_results: list[SeedBudgetResult]
    mean_accuracy: float
    std_accuracy: float

    def by_seed(self) -> dict[int, SeedBudgetResult]:
        return {item.seed: item for item in self.seed_results}


@dataclass(frozen=True)
class PairedImprovementSummary:
    n: int
    shared_seeds: list[int]
    mean_improvement: float
    std_improvement: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class GateAReport:
    passed: bool
    reason: str
    budget_ratio: float
    target_labeled_count: int
    minimum_required_improvement: float
    random_summary: StrategySummary
    entropy_summary: StrategySummary
    paired_improvement: PairedImprovementSummary
    report_json_path: Path
    report_markdown_path: Path
    created_at_utc: str


def run_gate_a(
    config: ExperimentConfig,
    config_source: Path | None = None,
    budget_ratio: float = 0.10,
    minimum_required_improvement: float = 0.05,
) -> GateAReport:
    from edge_al_pipeline.experiments.phase1_fashion_mnist import (
        run_phase1_fashion_mnist,
    )

    if config.dataset.name.lower() != "fashion_mnist":
        raise ValueError("Gate A currently supports only Fashion-MNIST.")
    if config.dataset.task != "classification":
        raise ValueError("Gate A requires classification task.")
    if budget_ratio <= 0 or budget_ratio > 1:
        raise ValueError("budget_ratio must be in the range (0, 1].")
    if minimum_required_improvement < 0:
        raise ValueError("minimum_required_improvement must be >= 0.")

    random_config = _clone_with_strategy(config, "random")
    entropy_config = _clone_with_strategy(config, "entropy")

    random_runs = run_phase1_fashion_mnist(random_config, config_source=config_source)
    entropy_runs = run_phase1_fashion_mnist(entropy_config, config_source=config_source)

    target_labeled_count = math.ceil(config.bootstrap.pool_size * budget_ratio)
    random_summary = summarize_strategy_runs(
        random_runs, random_config, target_labeled_count=target_labeled_count
    )
    entropy_summary = summarize_strategy_runs(
        entropy_runs, entropy_config, target_labeled_count=target_labeled_count
    )
    paired = compute_paired_improvement(
        random_summary.by_seed(),
        entropy_summary.by_seed(),
    )

    all_reached = _all_targets_reached(random_summary) and _all_targets_reached(
        entropy_summary
    )
    passed = all_reached and paired.mean_improvement >= minimum_required_improvement
    reason = _build_reason(
        passed=passed,
        all_reached=all_reached,
        paired=paired,
        minimum_required_improvement=minimum_required_improvement,
    )

    report_root = Path(config.output_root) / "gate_a_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_json_path = report_root / f"gate_a_{timestamp}.json"
    report_markdown_path = report_root / f"gate_a_{timestamp}.md"
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    report = GateAReport(
        passed=passed,
        reason=reason,
        budget_ratio=budget_ratio,
        target_labeled_count=target_labeled_count,
        minimum_required_improvement=minimum_required_improvement,
        random_summary=random_summary,
        entropy_summary=entropy_summary,
        paired_improvement=paired,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        created_at_utc=created_at_utc,
    )
    _write_report_files(report)
    return report


def summarize_strategy_runs(
    phase_summary: Phase1RunSummary,
    config: ExperimentConfig,
    target_labeled_count: int,
) -> StrategySummary:
    per_seed: list[SeedBudgetResult] = []
    for seed_run in phase_summary.results:
        round_to_acc = read_test_accuracy_by_round(seed_run.run_dir / "metrics.csv")
        if not round_to_acc:
            raise ValueError(f"No test_accuracy rows found in {seed_run.run_dir}.")

        available_rounds = sorted(round_to_acc.keys())
        round_index, labeled_count, reached = select_round_for_budget(
            available_rounds=available_rounds,
            initial_labeled_count=config.bootstrap.initial_labeled_size,
            query_size=config.query_size,
            target_labeled_count=target_labeled_count,
        )
        per_seed.append(
            SeedBudgetResult(
                seed=seed_run.seed,
                run_dir=seed_run.run_dir,
                round_index=round_index,
                labeled_count=labeled_count,
                target_labeled_count=target_labeled_count,
                target_reached=reached,
                test_accuracy=round_to_acc[round_index],
            )
        )

    values = [item.test_accuracy for item in per_seed]
    mean_accuracy = statistics.fmean(values)
    std_accuracy = statistics.stdev(values) if len(values) > 1 else 0.0
    return StrategySummary(
        strategy_name=config.strategy_name,
        seed_results=per_seed,
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
    )


def read_test_accuracy_by_round(metrics_path: Path) -> dict[int, float]:
    by_round: dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            if row.get("split") != "train":
                continue
            if row.get("metric") != "test_accuracy":
                continue
            round_index = int(row["round_index"])
            by_round[round_index] = float(row["value"])
    return by_round


def select_round_for_budget(
    available_rounds: list[int],
    initial_labeled_count: int,
    query_size: int,
    target_labeled_count: int,
) -> tuple[int, int, bool]:
    if not available_rounds:
        raise ValueError("available_rounds must include at least one round index.")
    if query_size <= 0:
        raise ValueError("query_size must be greater than 0.")

    selected_round = available_rounds[-1]
    reached = False
    for round_index in available_rounds:
        labeled_count = initial_labeled_count + (round_index * query_size)
        if labeled_count >= target_labeled_count:
            selected_round = round_index
            reached = True
            break

    labeled_count = initial_labeled_count + (selected_round * query_size)
    return selected_round, labeled_count, reached


def compute_paired_improvement(
    random_by_seed: Mapping[int, SeedBudgetResult],
    entropy_by_seed: Mapping[int, SeedBudgetResult],
) -> PairedImprovementSummary:
    shared_seeds = sorted(set(random_by_seed.keys()) & set(entropy_by_seed.keys()))
    if not shared_seeds:
        return PairedImprovementSummary(
            n=0,
            shared_seeds=[],
            mean_improvement=math.nan,
            std_improvement=math.nan,
            ci95_low=math.nan,
            ci95_high=math.nan,
        )

    diffs = [
        entropy_by_seed[seed].test_accuracy - random_by_seed[seed].test_accuracy
        for seed in shared_seeds
    ]
    mean_improvement = statistics.fmean(diffs)
    if len(diffs) == 1:
        return PairedImprovementSummary(
            n=1,
            shared_seeds=shared_seeds,
            mean_improvement=mean_improvement,
            std_improvement=0.0,
            ci95_low=mean_improvement,
            ci95_high=mean_improvement,
        )

    std_improvement = statistics.stdev(diffs)
    stderr = std_improvement / math.sqrt(len(diffs))
    t_critical = _t_critical_95(df=len(diffs) - 1)
    margin = t_critical * stderr
    return PairedImprovementSummary(
        n=len(diffs),
        shared_seeds=shared_seeds,
        mean_improvement=mean_improvement,
        std_improvement=std_improvement,
        ci95_low=mean_improvement - margin,
        ci95_high=mean_improvement + margin,
    )


def _clone_with_strategy(config: ExperimentConfig, strategy_name: str) -> ExperimentConfig:
    payload = config.to_dict()
    payload["strategy_name"] = strategy_name
    payload["experiment_name"] = f"{config.experiment_name}_gate_a_{strategy_name}"
    clone = ExperimentConfig.from_dict(payload)
    clone.validate()
    return clone


def _all_targets_reached(summary: StrategySummary) -> bool:
    return all(item.target_reached for item in summary.seed_results)


def _build_reason(
    passed: bool,
    all_reached: bool,
    paired: PairedImprovementSummary,
    minimum_required_improvement: float,
) -> str:
    if paired.n == 0:
        return "No shared seeds between random and entropy runs."
    if not all_reached:
        return (
            "Configured rounds/query size did not reach target labeled budget for all seeds."
        )
    if not passed:
        return (
            "Entropy improvement at target budget is below required threshold "
            f"({minimum_required_improvement:.3f})."
        )
    return "Gate A passed."


def _write_report_files(report: GateAReport) -> None:
    payload = {
        "created_at_utc": report.created_at_utc,
        "passed": report.passed,
        "reason": report.reason,
        "budget_ratio": report.budget_ratio,
        "target_labeled_count": report.target_labeled_count,
        "minimum_required_improvement": report.minimum_required_improvement,
        "random": _strategy_payload(report.random_summary),
        "entropy": _strategy_payload(report.entropy_summary),
        "paired_improvement": {
            "n": report.paired_improvement.n,
            "shared_seeds": report.paired_improvement.shared_seeds,
            "mean_improvement": report.paired_improvement.mean_improvement,
            "std_improvement": report.paired_improvement.std_improvement,
            "ci95_low": report.paired_improvement.ci95_low,
            "ci95_high": report.paired_improvement.ci95_high,
        },
    }
    report.report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.report_markdown_path.write_text(_to_markdown(report), encoding="utf-8")


def _strategy_payload(summary: StrategySummary) -> dict[str, object]:
    return {
        "strategy_name": summary.strategy_name,
        "mean_accuracy": summary.mean_accuracy,
        "std_accuracy": summary.std_accuracy,
        "seeds": [
            {
                "seed": item.seed,
                "run_dir": str(item.run_dir),
                "round_index": item.round_index,
                "labeled_count": item.labeled_count,
                "target_labeled_count": item.target_labeled_count,
                "target_reached": item.target_reached,
                "test_accuracy": item.test_accuracy,
            }
            for item in summary.seed_results
        ],
    }


def _to_markdown(report: GateAReport) -> str:
    paired = report.paired_improvement
    lines = [
        "# Gate A Report",
        "",
        f"- Status: {'PASS' if report.passed else 'FAIL'}",
        f"- Reason: {report.reason}",
        f"- Target budget ratio: {report.budget_ratio:.3f}",
        f"- Target labeled count: {report.target_labeled_count}",
        f"- Required improvement: {report.minimum_required_improvement:.3f}",
        "",
        "## Paired Improvement (Entropy - Random)",
        f"- Shared seeds: {paired.shared_seeds}",
        f"- N: {paired.n}",
        f"- Mean improvement: {paired.mean_improvement:.6f}",
        f"- Std improvement: {paired.std_improvement:.6f}",
        f"- 95% CI: [{paired.ci95_low:.6f}, {paired.ci95_high:.6f}]",
        "",
        "## Strategy Means",
        (
            f"- Random mean test accuracy: {report.random_summary.mean_accuracy:.6f} "
            f"(std={report.random_summary.std_accuracy:.6f})"
        ),
        (
            f"- Entropy mean test accuracy: {report.entropy_summary.mean_accuracy:.6f} "
            f"(std={report.entropy_summary.std_accuracy:.6f})"
        ),
        "",
        "## Per-Seed Details",
    ]
    lines.extend(_strategy_seed_lines(report.random_summary))
    lines.extend(_strategy_seed_lines(report.entropy_summary))
    lines.append("")
    lines.append(f"- Created at (UTC): {report.created_at_utc}")
    return "\n".join(lines)


def _strategy_seed_lines(summary: StrategySummary) -> list[str]:
    lines = [f"- `{summary.strategy_name}`:"]
    for item in summary.seed_results:
        lines.append(
            "  "
            f"- seed={item.seed}, acc={item.test_accuracy:.6f}, "
            f"round={item.round_index}, labeled={item.labeled_count}, "
            f"target_reached={item.target_reached}, run_dir={item.run_dir}"
        )
    return lines


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    return 1.96
