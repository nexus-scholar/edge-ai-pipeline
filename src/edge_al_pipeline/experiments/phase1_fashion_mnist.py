from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.config import ExperimentConfig
from edge_al_pipeline.data_pool import DataPoolManager
from edge_al_pipeline.experiments.bootstrap import build_bootstrap_splits, build_run_dir
from edge_al_pipeline.models.fashion_mnist_runner import (
    FashionMnistCnnRunner,
    FashionMnistRunnerConfig,
)
from edge_al_pipeline.pipeline import ActiveLearningPipeline
from edge_al_pipeline.profiling.edge_profiler import EdgeProfiler
from edge_al_pipeline.strategies import build_strategy


@dataclass(frozen=True)
class SeedRunResult:
    seed: int
    run_dir: Path
    final_labeled_count: int
    final_unlabeled_count: int


@dataclass(frozen=True)
class Phase1RunSummary:
    results: list[SeedRunResult]


def run_phase1_fashion_mnist(
    config: ExperimentConfig, config_source: Path | None = None
) -> Phase1RunSummary:
    if config.dataset.name.lower() != "fashion_mnist":
        raise ValueError(
            "Phase 1 runner currently supports only dataset.name='fashion_mnist'."
        )
    if config.dataset.task != "classification":
        raise ValueError("Phase 1 runner requires dataset.task='classification'.")
    if config.teacher_enabled:
        raise ValueError("teacher_enabled is not supported in Phase 1 runner.")

    results: list[SeedRunResult] = []
    for seed in config.seeds:
        run_dir = build_run_dir(config.output_root, config.experiment_name, seed)
        artifacts = ArtifactStore(run_dir)
        artifacts.initialize(config, config_source=config_source)

        splits = build_bootstrap_splits(config.bootstrap, seed=seed)
        artifacts.write_splits(
            splits,
            dataset_hash=f"{config.dataset.name}:{config.dataset.version}",
        )

        pool = DataPoolManager.from_splits(splits)
        strategy = build_strategy(config.strategy_name)
        profiler = EdgeProfiler(
            device=config.edge_device,
            quantization_mode=config.quantization_mode,
        )

        model_runner = FashionMnistCnnRunner(
            config=_runner_config_from_experiment(config),
            val_ids=splits.val,
            test_ids=None,
        )
        pipeline = ActiveLearningPipeline(
            pool=pool,
            model_runner=model_runner,
            strategy=strategy,
            artifacts=artifacts,
            profiler=profiler,
            query_size=config.query_size,
            teacher=None,
        )
        pipeline.run_seed(seed=seed, rounds=config.rounds)

        final_splits = pool.to_splits()
        artifacts.write_splits(
            final_splits,
            dataset_hash=f"{config.dataset.name}:{config.dataset.version}",
        )
        final_counts = final_splits.counts()
        results.append(
            SeedRunResult(
                seed=seed,
                run_dir=run_dir,
                final_labeled_count=final_counts["labeled"],
                final_unlabeled_count=final_counts["unlabeled"],
            )
        )
    return Phase1RunSummary(results=results)


def _runner_config_from_experiment(config: ExperimentConfig) -> FashionMnistRunnerConfig:
    params = config.model_params
    return FashionMnistRunnerConfig(
        data_root=config.dataset.root,
        batch_size=int(params.get("batch_size", 128)),
        score_batch_size=int(params.get("score_batch_size", 256)),
        epochs_per_round=int(params.get("epochs_per_round", 1)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        num_workers=int(params.get("num_workers", 0)),
        device=str(params.get("device", "cpu")),
        download=bool(params.get("download", True)),
        embedding_dim=int(params.get("embedding_dim", 64)),
    )
