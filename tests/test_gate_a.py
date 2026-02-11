from __future__ import annotations

from pathlib import Path

from edge_al_pipeline.config import BootstrapConfig, DatasetConfig, ExperimentConfig
from edge_al_pipeline.evaluation.gate_a import (
    GateAReport,
    compute_paired_improvement,
    read_test_accuracy_by_round,
    select_round_for_budget,
    summarize_strategy_runs,
)
from edge_al_pipeline.experiments.phase1_fashion_mnist import (
    Phase1RunSummary,
    SeedRunResult,
)


def _config(strategy_name: str = "entropy") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="gate_a_test",
        output_root="runs",
        dataset=DatasetConfig(
            name="fashion_mnist",
            root="data/fashion_mnist",
            version="1.0",
            task="classification",
            num_classes=10,
        ),
        model_name="simple_cnn",
        strategy_name=strategy_name,
        rounds=4,
        query_size=100,
        seeds=[1, 2],
        quantization_mode="fp32",
        teacher_enabled=False,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=6000,
            initial_labeled_size=600,
            val_size=600,
            test_size=1000,
        ),
    )


def test_select_round_for_budget_reached_and_not_reached():
    round_index, labeled_count, reached = select_round_for_budget(
        available_rounds=[0, 1, 2, 3],
        initial_labeled_count=600,
        query_size=100,
        target_labeled_count=850,
    )
    assert round_index == 3
    assert labeled_count == 900
    assert reached

    round_index, labeled_count, reached = select_round_for_budget(
        available_rounds=[0, 1, 2, 3],
        initial_labeled_count=600,
        query_size=100,
        target_labeled_count=1200,
    )
    assert round_index == 3
    assert labeled_count == 900
    assert not reached


def test_read_test_accuracy_by_round_parses_metric_rows(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "round_index,seed,split,metric,value",
                "0,1,train,loss,2.2",
                "0,1,train,test_accuracy,0.71",
                "1,1,train,test_accuracy,0.75",
            ]
        ),
        encoding="utf-8",
    )
    parsed = read_test_accuracy_by_round(metrics)
    assert parsed == {0: 0.71, 1: 0.75}


def test_compute_paired_improvement_returns_ci():
    random = {
        1: _seed_result(seed=1, test_accuracy=0.70),
        2: _seed_result(seed=2, test_accuracy=0.72),
        3: _seed_result(seed=3, test_accuracy=0.69),
    }
    entropy = {
        1: _seed_result(seed=1, test_accuracy=0.78),
        2: _seed_result(seed=2, test_accuracy=0.79),
        3: _seed_result(seed=3, test_accuracy=0.76),
    }
    paired = compute_paired_improvement(random, entropy)
    assert paired.n == 3
    assert paired.mean_improvement > 0.05
    assert paired.ci95_low <= paired.mean_improvement <= paired.ci95_high


def test_summarize_strategy_runs_chooses_target_round(tmp_path):
    seed1 = _write_metrics(tmp_path, seed=1, values=[0.50, 0.60, 0.65, 0.70])
    seed2 = _write_metrics(tmp_path, seed=2, values=[0.52, 0.62, 0.67, 0.71])
    summary = Phase1RunSummary(
        results=[
            SeedRunResult(seed=1, run_dir=seed1, final_labeled_count=900, final_unlabeled_count=0),
            SeedRunResult(seed=2, run_dir=seed2, final_labeled_count=900, final_unlabeled_count=0),
        ]
    )
    strategy_summary = summarize_strategy_runs(
        summary,
        _config(strategy_name="entropy"),
        target_labeled_count=800,
    )
    assert len(strategy_summary.seed_results) == 2
    for item in strategy_summary.seed_results:
        assert item.round_index == 2
        assert item.labeled_count == 800
        assert item.target_reached


def _seed_result(seed: int, test_accuracy: float):
    from edge_al_pipeline.evaluation.gate_a import SeedBudgetResult

    return SeedBudgetResult(
        seed=seed,
        run_dir=Path("runs/dummy"),
        round_index=0,
        labeled_count=600,
        target_labeled_count=600,
        target_reached=True,
        test_accuracy=test_accuracy,
    )


def _write_metrics(tmp_path, seed: int, values: list[float]) -> Path:
    run_dir = tmp_path / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = ["round_index,seed,split,metric,value"]
    for idx, value in enumerate(values):
        rows.append(f"{idx},{seed},train,test_accuracy,{value}")
    (run_dir / "metrics.csv").write_text("\n".join(rows), encoding="utf-8")
    return run_dir
