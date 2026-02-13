from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edge_al_pipeline.backbones import resolve_backbone_name
from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.config import BootstrapConfig, ExperimentConfig
from edge_al_pipeline.data_pool import DataPoolManager
from edge_al_pipeline.experiments.bootstrap import build_bootstrap_splits, build_run_dir
from edge_al_pipeline.models.image_folder_mobilenet_runner import (
    ImageFolderMobileNetRunner,
    ImageFolderMobileNetRunnerConfig,
)
from edge_al_pipeline.pipeline import ActiveLearningPipeline
from edge_al_pipeline.profiling.edge_profiler import EdgeProfiler
from edge_al_pipeline.strategies import build_strategy


@dataclass(frozen=True)
class Phase2SeedResult:
    seed: int
    run_dir: Path
    backbone_path: Path
    final_labeled_count: int
    final_unlabeled_count: int
    dataset_size_used: int


@dataclass(frozen=True)
class Phase2RunSummary:
    results: list[Phase2SeedResult]


def run_phase2_agri_classification(
    config: ExperimentConfig, config_source: Path | None = None
) -> Phase2RunSummary:
    if config.dataset.task != "classification":
        raise ValueError("Phase 2 runner requires dataset.task='classification'.")
    if config.teacher_enabled:
        raise ValueError("teacher_enabled is not supported in Phase 2 runner.")

    max_samples = config.model_params.get("max_samples")
    max_samples_int = int(max_samples) if max_samples is not None else None
    dataset_size, class_count = ImageFolderMobileNetRunner.inspect_dataset(
        root=config.dataset.root,
        max_samples=max_samples_int,
    )
    if dataset_size <= 0:
        raise ValueError("Phase 2 dataset is empty.")
    if config.dataset.num_classes is not None and config.dataset.num_classes != class_count:
        raise ValueError(
            f"dataset.num_classes ({config.dataset.num_classes}) does not match "
            f"dataset classes on disk ({class_count})."
        )

    effective_bootstrap = _effective_bootstrap(config.bootstrap, dataset_size)
    results: list[Phase2SeedResult] = []
    
    # Collect all IDs once (deterministically)
    # We replicate the logic from bootstrap.py to fetch real IDs
    dataset_root = Path(config.dataset.root)
    all_ids = []
    if dataset_root.exists() and dataset_root.is_dir():
        for class_dir in sorted(dataset_root.iterdir()):
            if class_dir.is_dir():
                for image_file in sorted(class_dir.iterdir()):
                    if image_file.is_file():
                         all_ids.append(image_file.name)
    else:
        # Fallback should arguably be an error here, but for consistency:
        all_ids = [f"sample_{index:06d}" for index in range(effective_bootstrap.pool_size)]

    for seed in config.seeds:
        print(f"--- Starting Phase 2 Pretraining (Seed {seed}) ---")
        run_dir = build_run_dir(config.output_root, config.experiment_name, seed)
        artifacts = ArtifactStore(run_dir)
        artifacts.initialize(config, config_source=config_source)

        splits = build_bootstrap_splits(effective_bootstrap, all_ids, seed=seed)
        artifacts.write_splits(
            splits,
            dataset_hash=f"{config.dataset.name}:{config.dataset.version}",
        )

        pool = DataPoolManager.from_splits(splits)
        strategy = build_strategy(config.strategy_name, config.strategy_params)
        profiler = EdgeProfiler(
            device=config.edge_device,
            quantization_mode=config.quantization_mode,
        )
        model_runner = ImageFolderMobileNetRunner(
            config=_runner_config_from_experiment(config, num_classes=class_count),
            val_ids=splits.val,
            test_ids=splits.test,
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
        backbone_path = model_runner.export_backbone(
            run_dir / "checkpoints" / f"agri_backbone_seed{seed}.pt"
        )
        final_counts = final_splits.counts()
        results.append(
            Phase2SeedResult(
                seed=seed,
                run_dir=run_dir,
                backbone_path=backbone_path,
                final_labeled_count=final_counts["labeled"],
                final_unlabeled_count=final_counts["unlabeled"],
                dataset_size_used=dataset_size,
            )
        )
    return Phase2RunSummary(results=results)


def _runner_config_from_experiment(
    config: ExperimentConfig, num_classes: int
) -> ImageFolderMobileNetRunnerConfig:
    params = config.model_params
    backbone_name = resolve_backbone_name(config.model_name, params)
    if backbone_name is None:
        backbone_name = "mobilenet_v3_small"
    max_samples = params.get("max_samples")
    max_samples_int = int(max_samples) if max_samples is not None else None
    return ImageFolderMobileNetRunnerConfig(
        data_root=config.dataset.root,
        batch_size=int(params.get("batch_size", 32)),
        score_batch_size=int(params.get("score_batch_size", 64)),
        epochs_per_round=int(params.get("epochs_per_round", 1)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        num_workers=int(params.get("num_workers", 0)),
        device=str(params.get("device", "cpu")),
        image_size=int(params.get("image_size", 224)),
        pretrained_backbone=bool(params.get("pretrained_backbone", True)),
        freeze_backbone=bool(params.get("freeze_backbone", False)),
        backbone_name=backbone_name,
        backbone_checkpoint=(
            str(params["backbone_checkpoint"])
            if "backbone_checkpoint" in params
            and params.get("backbone_checkpoint") is not None
            else None
        ),
        num_classes=num_classes,
        max_samples=max_samples_int,
    )


def _effective_bootstrap(bootstrap: BootstrapConfig, dataset_size: int) -> BootstrapConfig:
    pool_size = min(bootstrap.pool_size, dataset_size)
    known_total = bootstrap.initial_labeled_size + bootstrap.val_size + bootstrap.test_size
    if known_total >= pool_size:
        raise ValueError(
            "Phase 2 bootstrap sizes are too large for the available dataset size "
            f"({dataset_size})."
        )
    return BootstrapConfig(
        pool_size=pool_size,
        initial_labeled_size=bootstrap.initial_labeled_size,
        val_size=bootstrap.val_size,
        test_size=bootstrap.test_size,
    )

