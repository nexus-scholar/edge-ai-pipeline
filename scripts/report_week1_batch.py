from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROFILES = {
    "base": {
        "random": "cdgp_week1_batch_cifar10_random",
        "entropy": "cdgp_week1_batch_cifar10_entropy",
        "domain_guided_w02": "cdgp_week1_batch_cifar10_domain_guided_w02",
        "domain_guided_w05": "cdgp_week1_batch_cifar10_domain_guided_w05",
        "domain_guided_w08": "cdgp_week1_batch_cifar10_domain_guided_w08",
    },
    "calibrated": {
        "random": "cdgp_week1_batch_cifar10_random",
        "entropy": "cdgp_week1_batch_cifar10_entropy",
        "domain_guided_calibrated_w02": "cdgp_week1_batch_cifar10_domain_guided_calibrated_w02",
        "domain_guided_calibrated_w05": "cdgp_week1_batch_cifar10_domain_guided_calibrated_w05",
        "domain_guided_calibrated_w08": "cdgp_week1_batch_cifar10_domain_guided_calibrated_w08",
    },
}

PAIR_DEFINITIONS = {
    "base": [
        ("entropy_minus_random", "random", "entropy"),
        ("domain_guided_w02_minus_random", "random", "domain_guided_w02"),
        ("domain_guided_w05_minus_random", "random", "domain_guided_w05"),
        ("domain_guided_w08_minus_random", "random", "domain_guided_w08"),
        ("domain_guided_w05_minus_entropy", "entropy", "domain_guided_w05"),
    ],
    "calibrated": [
        ("entropy_minus_random", "random", "entropy"),
        (
            "domain_guided_calibrated_w02_minus_random",
            "random",
            "domain_guided_calibrated_w02",
        ),
        (
            "domain_guided_calibrated_w05_minus_random",
            "random",
            "domain_guided_calibrated_w05",
        ),
        (
            "domain_guided_calibrated_w08_minus_random",
            "random",
            "domain_guided_calibrated_w08",
        ),
        (
            "domain_guided_calibrated_w05_minus_entropy",
            "entropy",
            "domain_guided_calibrated_w05",
        ),
    ],
}


@dataclass(frozen=True)
class SeedMetrics:
    strategy: str
    seed: int
    run_dir: Path
    rounds: list[int]
    test_series: list[float]
    final_test_accuracy: float
    auc_test_accuracy: float
    final_test_accuracy_blur: float
    final_test_accuracy_low_contrast: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Week 1 batch comparison report from run artifacts."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory containing experiment run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/reports"),
        help="Directory where report artifacts are written.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="base",
        help="Comparison profile: 'base' or 'calibrated'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = PROFILES[args.profile]
    pair_definitions = PAIR_DEFINITIONS[args.profile]
    selected_runs = select_latest_run_per_seed(args.runs_root, experiments)
    per_seed = collect_seed_metrics(selected_runs)
    strategy_summary = summarize_strategies(per_seed, strategies=experiments.keys())
    pair_rows = build_pair_rows(
        per_seed,
        strategies=experiments.keys(),
        pair_definitions=pair_definitions,
    )
    pair_summary = summarize_pairs(pair_rows)
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"week1_batch_cifar10_{args.profile}_comparison"
    json_path = args.output_dir / f"{prefix}_{stamp}.json"
    summary_csv_path = args.output_dir / f"{prefix}_{stamp}.csv"
    pairs_csv_path = (
        args.output_dir / f"{prefix}_pairs_{stamp}.csv"
    )
    markdown_path = args.output_dir / f"{prefix}_{stamp}.md"

    payload = {
        "created_at_utc": created_at_utc,
        "profile": args.profile,
        "experiments": experiments,
        "selected_runs": {
            strategy: {
                str(seed): str(run_dir)
                for seed, (run_dir, _metrics) in by_seed.items()
            }
            for strategy, by_seed in selected_runs.items()
        },
        "strategy_summary": strategy_summary,
        "pair_summary": pair_summary,
        "per_seed": [as_payload(item) for item in per_seed],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv_path, strategy_summary)
    write_pairs_csv(pairs_csv_path, pair_rows)
    markdown_path.write_text(
        build_markdown(created_at_utc, strategy_summary, pair_summary), encoding="utf-8"
    )

    print("Report generated:")
    print(f"- JSON: {json_path}")
    print(f"- Summary CSV: {summary_csv_path}")
    print(f"- Pairs CSV: {pairs_csv_path}")
    print(f"- Markdown: {markdown_path}")
    return 0


def select_latest_run_per_seed(
    runs_root: Path,
    experiments: dict[str, str],
) -> dict[str, dict[int, tuple[Path, dict[tuple[int, int], dict[str, float]]]]]:
    selected: dict[str, dict[int, tuple[Path, dict[tuple[int, int], dict[str, float]]]]] = {}
    for strategy, experiment_name in experiments.items():
        experiment_root = runs_root / experiment_name
        latest_by_seed: dict[int, tuple[Path, dict[tuple[int, int], dict[str, float]]]] = {}
        if not experiment_root.exists():
            selected[strategy] = latest_by_seed
            continue
        for run_dir in [item for item in experiment_root.iterdir() if item.is_dir()]:
            metrics_path = run_dir / "metrics.csv"
            if not metrics_path.exists():
                continue
            grouped = read_metrics(metrics_path)
            if not grouped:
                continue
            seeds = sorted({key[0] for key in grouped})
            if len(seeds) != 1:
                continue
            seed = seeds[0]
            previous = latest_by_seed.get(seed)
            if previous is None or run_dir.stat().st_mtime > previous[0].stat().st_mtime:
                latest_by_seed[seed] = (run_dir, grouped)
        selected[strategy] = latest_by_seed
    return selected


def read_metrics(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    grouped: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if row.get("split") != "train":
            continue
        key = (int(row["seed"]), int(row["round_index"]))
        grouped.setdefault(key, {})[row["metric"]] = float(row["value"])
    return grouped


def collect_seed_metrics(
    selected_runs: dict[str, dict[int, tuple[Path, dict[tuple[int, int], dict[str, float]]]]]
) -> list[SeedMetrics]:
    rows: list[SeedMetrics] = []
    for strategy, by_seed in selected_runs.items():
        for seed, (run_dir, grouped) in sorted(by_seed.items()):
            rounds = sorted([key[1] for key in grouped if key[0] == seed])
            if not rounds:
                continue
            test_series = [
                grouped[(seed, round_idx)].get("test_accuracy", math.nan)
                for round_idx in rounds
            ]
            blur_series = [
                grouped[(seed, round_idx)].get("test_accuracy_blur", math.nan)
                for round_idx in rounds
            ]
            low_contrast_series = [
                grouped[(seed, round_idx)].get("test_accuracy_low_contrast", math.nan)
                for round_idx in rounds
            ]
            rows.append(
                SeedMetrics(
                    strategy=strategy,
                    seed=seed,
                    run_dir=run_dir,
                    rounds=rounds,
                    test_series=test_series,
                    final_test_accuracy=test_series[-1],
                    auc_test_accuracy=sum(test_series) / len(test_series),
                    final_test_accuracy_blur=blur_series[-1],
                    final_test_accuracy_low_contrast=low_contrast_series[-1],
                )
            )
    return rows


def summarize_strategies(
    per_seed: list[SeedMetrics],
    strategies: Sequence[str],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for strategy in strategies:
        rows = [item for item in per_seed if item.strategy == strategy]
        finals = [item.final_test_accuracy for item in rows]
        aucs = [item.auc_test_accuracy for item in rows]
        blurs = [item.final_test_accuracy_blur for item in rows]
        lows = [item.final_test_accuracy_low_contrast for item in rows]
        round_mean_test_accuracy: dict[str, float] = {}
        max_round = max((max(item.rounds) for item in rows), default=-1)
        for round_idx in range(max_round + 1):
            values: list[float] = []
            for item in rows:
                if round_idx in item.rounds:
                    pos = item.rounds.index(round_idx)
                    values.append(item.test_series[pos])
            if values:
                round_mean_test_accuracy[str(round_idx)] = sum(values) / len(values)
        summary.append(
            {
                "strategy": strategy,
                "n_seeds": len(rows),
                "seeds": [item.seed for item in rows],
                "mean_final_test_accuracy": statistics.fmean(finals)
                if finals
                else math.nan,
                "std_final_test_accuracy": statistics.stdev(finals)
                if len(finals) > 1
                else 0.0,
                "mean_auc_test_accuracy": statistics.fmean(aucs) if aucs else math.nan,
                "mean_final_test_accuracy_blur": statistics.fmean(blurs)
                if blurs
                else math.nan,
                "mean_final_test_accuracy_low_contrast": statistics.fmean(lows)
                if lows
                else math.nan,
                "round_mean_test_accuracy": round_mean_test_accuracy,
            }
        )
    return summary


def build_pair_rows(
    per_seed: list[SeedMetrics],
    strategies: Sequence[str],
    pair_definitions: Sequence[tuple[str, str, str]],
) -> list[dict[str, object]]:
    by_strategy_seed: dict[str, dict[int, SeedMetrics]] = {}
    for strategy in strategies:
        by_strategy_seed[strategy] = {
            item.seed: item for item in per_seed if item.strategy == strategy
        }

    pairs: list[dict[str, object]] = []
    for pair_name, left, right in pair_definitions:
        if left not in by_strategy_seed or right not in by_strategy_seed:
            continue
        pairs.extend(pairwise(pair_name, left, right, by_strategy_seed))
    return pairs


def pairwise(
    pair_name: str,
    left: str,
    right: str,
    by_strategy_seed: dict[str, dict[int, SeedMetrics]],
) -> list[dict[str, object]]:
    seeds = sorted(set(by_strategy_seed[left]) & set(by_strategy_seed[right]))
    rows: list[dict[str, object]] = []
    for seed in seeds:
        left_item = by_strategy_seed[left][seed]
        right_item = by_strategy_seed[right][seed]
        rows.append(
            {
                "pair": pair_name,
                "seed": seed,
                "left_strategy": left,
                "right_strategy": right,
                "delta_final_test_accuracy": right_item.final_test_accuracy
                - left_item.final_test_accuracy,
                "delta_auc_test_accuracy": right_item.auc_test_accuracy
                - left_item.auc_test_accuracy,
            }
        )
    return rows


def summarize_pairs(pair_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    pair_names = sorted({row["pair"] for row in pair_rows})
    for pair_name in pair_names:
        rows = [row for row in pair_rows if row["pair"] == pair_name]
        delta_final = [float(row["delta_final_test_accuracy"]) for row in rows]
        delta_auc = [float(row["delta_auc_test_accuracy"]) for row in rows]
        summary[pair_name] = {
            "n": len(rows),
            "mean_delta_final_test_accuracy": statistics.fmean(delta_final)
            if delta_final
            else math.nan,
            "mean_delta_auc_test_accuracy": statistics.fmean(delta_auc)
            if delta_auc
            else math.nan,
            "per_seed": rows,
        }
    return summary


def as_payload(item: SeedMetrics) -> dict[str, object]:
    return {
        "strategy": item.strategy,
        "seed": item.seed,
        "run_dir": str(item.run_dir),
        "rounds": item.rounds,
        "test_series": item.test_series,
        "final_test_accuracy": item.final_test_accuracy,
        "auc_test_accuracy": item.auc_test_accuracy,
        "final_test_accuracy_blur": item.final_test_accuracy_blur,
        "final_test_accuracy_low_contrast": item.final_test_accuracy_low_contrast,
    }


def write_summary_csv(path: Path, strategy_summary: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "n_seeds",
                "seeds",
                "mean_final_test_accuracy",
                "std_final_test_accuracy",
                "mean_auc_test_accuracy",
                "mean_final_test_accuracy_blur",
                "mean_final_test_accuracy_low_contrast",
                "round0_mean_test_accuracy",
                "round1_mean_test_accuracy",
                "round2_mean_test_accuracy",
                "round3_mean_test_accuracy",
                "round4_mean_test_accuracy",
                "round5_mean_test_accuracy",
            ],
        )
        writer.writeheader()
        for row in strategy_summary:
            round_means = row["round_mean_test_accuracy"]
            writer.writerow(
                {
                    "strategy": row["strategy"],
                    "n_seeds": row["n_seeds"],
                    "seeds": "|".join(str(seed) for seed in row["seeds"]),
                    "mean_final_test_accuracy": row["mean_final_test_accuracy"],
                    "std_final_test_accuracy": row["std_final_test_accuracy"],
                    "mean_auc_test_accuracy": row["mean_auc_test_accuracy"],
                    "mean_final_test_accuracy_blur": row[
                        "mean_final_test_accuracy_blur"
                    ],
                    "mean_final_test_accuracy_low_contrast": row[
                        "mean_final_test_accuracy_low_contrast"
                    ],
                    "round0_mean_test_accuracy": round_means.get("0", math.nan),
                    "round1_mean_test_accuracy": round_means.get("1", math.nan),
                    "round2_mean_test_accuracy": round_means.get("2", math.nan),
                    "round3_mean_test_accuracy": round_means.get("3", math.nan),
                    "round4_mean_test_accuracy": round_means.get("4", math.nan),
                    "round5_mean_test_accuracy": round_means.get("5", math.nan),
                }
            )


def write_pairs_csv(path: Path, pair_rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair",
                "seed",
                "left_strategy",
                "right_strategy",
                "delta_final_test_accuracy",
                "delta_auc_test_accuracy",
            ],
        )
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(row)


def build_markdown(
    created_at_utc: str,
    strategy_summary: list[dict[str, object]],
    pair_summary: dict[str, dict[str, object]],
) -> str:
    lines = [
        "# Week 1 Batch Comparison (CIFAR10)",
        "",
        f"- Created at (UTC): {created_at_utc}",
        "",
        "## Strategy Summary",
        "| Strategy | Seeds | Mean Final Test Acc | Mean AUC Test Acc | Mean Final Blur Acc | Mean Final Low-Contrast Acc |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in strategy_summary:
        lines.append(
            f"| {row['strategy']} | {row['seeds']} | "
            f"{float(row['mean_final_test_accuracy']):.6f} | "
            f"{float(row['mean_auc_test_accuracy']):.6f} | "
            f"{float(row['mean_final_test_accuracy_blur']):.6f} | "
            f"{float(row['mean_final_test_accuracy_low_contrast']):.6f} |"
        )
    lines.extend(["", "## Mean Paired Deltas"])
    for name, values in sorted(pair_summary.items()):
        lines.append(
            f"- {name}: "
            f"mean_delta_final={float(values['mean_delta_final_test_accuracy']):.6f}, "
            f"mean_delta_auc={float(values['mean_delta_auc_test_accuracy']):.6f}, "
            f"n={int(values['n'])}"
        )
    lines.extend(["", "## Adaptation Curve Means (Test Accuracy)"])
    for row in strategy_summary:
        round_means = row["round_mean_test_accuracy"]
        parts = [
            f"r{round_idx}={float(round_means.get(str(round_idx), math.nan)):.6f}"
            for round_idx in range(0, 6)
        ]
        lines.append(f"- {row['strategy']}: " + ", ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
