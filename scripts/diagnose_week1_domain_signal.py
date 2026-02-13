from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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


@dataclass(frozen=True)
class RunRef:
    strategy: str
    seed: int
    run_dir: Path


@dataclass(frozen=True)
class RoundDiagnostic:
    strategy: str
    seed: int
    run_dir: str
    round_index: int
    selected_count: int
    mean_domain_confusion: float
    std_domain_confusion: float
    mean_entropy: float
    mean_combined: float
    mean_selection_score: float
    test_accuracy: float
    next_test_accuracy: float
    next_round_gain: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Week-1 domain-guided signal quality using selection metadata "
            "and next-round test accuracy gains."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root runs directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/reports"),
        help="Where to write diagnostic reports.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="base",
        help="Diagnosis profile: 'base' or 'calibrated'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = PROFILES[args.profile]
    run_refs = collect_latest_run_per_seed(args.runs_root, experiments)
    diagnostics = collect_round_diagnostics(run_refs)
    summary = summarize_diagnostics(diagnostics, experiments=experiments)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    prefix = f"week1_{args.profile}_domain_signal_diagnostics"
    csv_path = args.output_dir / f"{prefix}_{stamp}.csv"
    json_path = args.output_dir / f"{prefix}_{stamp}.json"
    md_path = args.output_dir / f"{prefix}_{stamp}.md"

    write_round_csv(csv_path, diagnostics)
    payload = {
        "created_at_utc": created_at_utc,
        "profile": args.profile,
        "experiments": experiments,
        "selected_runs": {
            strategy: {str(item.seed): str(item.run_dir) for item in refs}
            for strategy, refs in run_refs.items()
        },
        "summary": summary,
        "round_diagnostics": [as_payload(item) for item in diagnostics],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(created_at_utc, summary), encoding="utf-8")

    print("Domain-signal diagnostic report generated:")
    print(f"- CSV: {csv_path}")
    print(f"- JSON: {json_path}")
    print(f"- Markdown: {md_path}")
    return 0


def collect_latest_run_per_seed(
    runs_root: Path,
    experiments: dict[str, str],
) -> dict[str, list[RunRef]]:
    by_strategy: dict[str, list[RunRef]] = {}
    for strategy, experiment_name in experiments.items():
        experiment_root = runs_root / experiment_name
        latest_by_seed: dict[int, Path] = {}
        if experiment_root.exists():
            for run_dir in [item for item in experiment_root.iterdir() if item.is_dir()]:
                metrics_path = run_dir / "metrics.csv"
                if not metrics_path.exists():
                    continue
                seed = infer_seed_from_metrics(metrics_path)
                if seed is None:
                    continue
                previous = latest_by_seed.get(seed)
                if previous is None or run_dir.stat().st_mtime > previous.stat().st_mtime:
                    latest_by_seed[seed] = run_dir
        refs = [
            RunRef(strategy=strategy, seed=seed, run_dir=run_dir)
            for seed, run_dir in sorted(latest_by_seed.items())
        ]
        by_strategy[strategy] = refs
    return by_strategy


def infer_seed_from_metrics(metrics_path: Path) -> int | None:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        seeds = {int(row["seed"]) for row in reader if row.get("seed")}
    if len(seeds) != 1:
        return None
    return next(iter(seeds))


def collect_round_diagnostics(run_refs: dict[str, list[RunRef]]) -> list[RoundDiagnostic]:
    rows: list[RoundDiagnostic] = []
    for strategy, refs in run_refs.items():
        for ref in refs:
            test_by_round = read_test_accuracy_by_round(ref.run_dir / "metrics.csv", ref.seed)
            if not test_by_round:
                continue
            max_round = max(test_by_round.keys())
            for round_index in range(max_round):
                selected_path = ref.run_dir / f"round_{round_index}_selected.csv"
                if not selected_path.exists():
                    continue
                selection = read_round_selection(selected_path)
                if not selection:
                    continue
                test_accuracy = test_by_round.get(round_index, math.nan)
                next_test_accuracy = test_by_round.get(round_index + 1, math.nan)
                if math.isnan(test_accuracy) or math.isnan(next_test_accuracy):
                    continue
                next_round_gain = next_test_accuracy - test_accuracy
                rows.append(
                    RoundDiagnostic(
                        strategy=strategy,
                        seed=ref.seed,
                        run_dir=str(ref.run_dir),
                        round_index=round_index,
                        selected_count=selection["selected_count"],
                        mean_domain_confusion=selection["mean_domain_confusion"],
                        std_domain_confusion=selection["std_domain_confusion"],
                        mean_entropy=selection["mean_entropy"],
                        mean_combined=selection["mean_combined"],
                        mean_selection_score=selection["mean_selection_score"],
                        test_accuracy=test_accuracy,
                        next_test_accuracy=next_test_accuracy,
                        next_round_gain=next_round_gain,
                    )
                )
    return rows


def read_test_accuracy_by_round(metrics_path: Path, seed: int) -> dict[int, float]:
    values: dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["seed"]) != seed:
                continue
            if row.get("split") != "train":
                continue
            if row.get("metric") != "test_accuracy":
                continue
            values[int(row["round_index"])] = float(row["value"])
    return values


def read_round_selection(path: Path) -> dict[str, float | int]:
    domain_confusion: list[float] = []
    entropy: list[float] = []
    combined: list[float] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata = json.loads(row.get("metadata", "{}"))
            scores.append(float(row.get("score", math.nan)))
            domain_confusion.append(
                to_float(metadata.get("domain_confusion", math.nan), default=math.nan)
            )
            entropy.append(
                to_float(
                    metadata.get(
                        "uncertainty_entropy", metadata.get("uncertainty_combined", math.nan)
                    ),
                    default=math.nan,
                )
            )
            combined.append(
                to_float(metadata.get("uncertainty_combined", math.nan), default=math.nan)
            )
    dc = finite(domain_confusion)
    ent = finite(entropy)
    cmb = finite(combined)
    sc = finite(scores)
    return {
        "selected_count": len(scores),
        "mean_domain_confusion": statistics.fmean(dc) if dc else math.nan,
        "std_domain_confusion": statistics.stdev(dc) if len(dc) > 1 else 0.0,
        "mean_entropy": statistics.fmean(ent) if ent else math.nan,
        "mean_combined": statistics.fmean(cmb) if cmb else math.nan,
        "mean_selection_score": statistics.fmean(sc) if sc else math.nan,
    }


def summarize_diagnostics(
    diagnostics: list[RoundDiagnostic],
    experiments: dict[str, str],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    summary["n_points"] = len(diagnostics)

    all_dc = [item.mean_domain_confusion for item in diagnostics]
    all_gain = [item.next_round_gain for item in diagnostics]
    all_ent = [item.mean_entropy for item in diagnostics]

    summary["overall"] = {
        "mean_domain_confusion": fmean_safe(all_dc),
        "mean_next_round_gain": fmean_safe(all_gain),
        "corr_domain_confusion_vs_next_gain": pearson_corr(all_dc, all_gain),
        "corr_entropy_vs_next_gain": pearson_corr(all_ent, all_gain),
    }

    per_strategy: dict[str, object] = {}
    for strategy in experiments:
        rows = [item for item in diagnostics if item.strategy == strategy]
        dc = [item.mean_domain_confusion for item in rows]
        gain = [item.next_round_gain for item in rows]
        ent = [item.mean_entropy for item in rows]
        per_strategy[strategy] = {
            "n_points": len(rows),
            "mean_domain_confusion": fmean_safe(dc),
            "mean_next_round_gain": fmean_safe(gain),
            "corr_domain_confusion_vs_next_gain": pearson_corr(dc, gain),
            "corr_entropy_vs_next_gain": pearson_corr(ent, gain),
        }
    summary["per_strategy"] = per_strategy

    q1, q2, q3 = quantiles(all_dc)
    bucket_rows = [
        {
            "bucket": "low",
            "count": len([x for x in all_dc if x <= q1]),
            "mean_next_round_gain": fmean_safe(
                [g for x, g in zip(all_dc, all_gain) if x <= q1]
            ),
        },
        {
            "bucket": "mid_low",
            "count": len([x for x in all_dc if q1 < x <= q2]),
            "mean_next_round_gain": fmean_safe(
                [g for x, g in zip(all_dc, all_gain) if q1 < x <= q2]
            ),
        },
        {
            "bucket": "mid_high",
            "count": len([x for x in all_dc if q2 < x <= q3]),
            "mean_next_round_gain": fmean_safe(
                [g for x, g in zip(all_dc, all_gain) if q2 < x <= q3]
            ),
        },
        {
            "bucket": "high",
            "count": len([x for x in all_dc if x > q3]),
            "mean_next_round_gain": fmean_safe(
                [g for x, g in zip(all_dc, all_gain) if x > q3]
            ),
        },
    ]
    summary["domain_confusion_gain_buckets"] = {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "rows": bucket_rows,
    }
    return summary


def quantiles(values: list[float]) -> tuple[float, float, float]:
    data = sorted(finite(values))
    if not data:
        return math.nan, math.nan, math.nan
    return (
        percentile(data, 0.25),
        percentile(data, 0.50),
        percentile(data, 0.75),
    )


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def pearson_corr(first: list[float], second: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(first, second) if is_finite(a) and is_finite(b)]
    if len(pairs) < 2:
        return math.nan
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return math.nan
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def write_round_csv(path: Path, diagnostics: list[RoundDiagnostic]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "seed",
                "run_dir",
                "round_index",
                "selected_count",
                "mean_domain_confusion",
                "std_domain_confusion",
                "mean_entropy",
                "mean_combined",
                "mean_selection_score",
                "test_accuracy",
                "next_test_accuracy",
                "next_round_gain",
            ],
        )
        writer.writeheader()
        for item in diagnostics:
            writer.writerow(as_payload(item))


def build_markdown(created_at_utc: str, summary: dict[str, object]) -> str:
    overall = summary["overall"]
    lines = [
        "# Week 1 Domain-Signal Diagnostics",
        "",
        f"- Created at (UTC): {created_at_utc}",
        f"- Data points (seed x round): {summary['n_points']}",
        "",
        "## Overall",
        f"- Mean domain_confusion: {to_str(overall['mean_domain_confusion'])}",
        f"- Mean next-round gain: {to_str(overall['mean_next_round_gain'])}",
        (
            "- Correlation(domain_confusion, next-round gain): "
            f"{to_str(overall['corr_domain_confusion_vs_next_gain'])}"
        ),
        (
            "- Correlation(entropy, next-round gain): "
            f"{to_str(overall['corr_entropy_vs_next_gain'])}"
        ),
        "",
        "## Per Strategy",
        "| Strategy | Points | Mean domain_confusion | Mean next-round gain | Corr(domain_confusion, gain) | Corr(entropy, gain) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for strategy, values in summary["per_strategy"].items():
        lines.append(
            f"| {strategy} | {values['n_points']} | "
            f"{to_str(values['mean_domain_confusion'])} | "
            f"{to_str(values['mean_next_round_gain'])} | "
            f"{to_str(values['corr_domain_confusion_vs_next_gain'])} | "
            f"{to_str(values['corr_entropy_vs_next_gain'])} |"
        )

    bucket_summary = summary["domain_confusion_gain_buckets"]
    lines.extend(
        [
            "",
            "## Domain-Confusion Buckets vs Gain",
            (
                f"- Quantiles: q1={to_str(bucket_summary['q1'])}, "
                f"q2={to_str(bucket_summary['q2'])}, q3={to_str(bucket_summary['q3'])}"
            ),
            "| Bucket | Count | Mean next-round gain |",
            "| --- | --- | --- |",
        ]
    )
    for row in bucket_summary["rows"]:
        lines.append(
            f"| {row['bucket']} | {row['count']} | {to_str(row['mean_next_round_gain'])} |"
        )
    return "\n".join(lines)


def as_payload(item: RoundDiagnostic) -> dict[str, object]:
    return {
        "strategy": item.strategy,
        "seed": item.seed,
        "run_dir": item.run_dir,
        "round_index": item.round_index,
        "selected_count": item.selected_count,
        "mean_domain_confusion": item.mean_domain_confusion,
        "std_domain_confusion": item.std_domain_confusion,
        "mean_entropy": item.mean_entropy,
        "mean_combined": item.mean_combined,
        "mean_selection_score": item.mean_selection_score,
        "test_accuracy": item.test_accuracy,
        "next_test_accuracy": item.next_test_accuracy,
        "next_round_gain": item.next_round_gain,
    }


def is_finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def finite(values: list[float]) -> list[float]:
    return [value for value in values if is_finite(value)]


def to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmean_safe(values: list[float]) -> float:
    items = finite(values)
    return statistics.fmean(items) if items else math.nan


def to_str(value: object) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not is_finite(value):
            return "nan"
        return f"{float(value):.6f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
