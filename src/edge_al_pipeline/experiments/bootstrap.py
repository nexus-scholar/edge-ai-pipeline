from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.config import BootstrapConfig, ExperimentConfig
from edge_al_pipeline.contracts import DatasetSplits


@dataclass(frozen=True)
class BootstrapResult:
    run_dir: Path
    labeled_count: int
    unlabeled_count: int
    val_count: int
    test_count: int


def initialize_run(
    config: ExperimentConfig, config_source: Path | None = None
) -> BootstrapResult:
    run_dir = build_run_dir(config.output_root, config.experiment_name, config.seeds[0])

    artifacts = ArtifactStore(run_dir)
    artifacts.initialize(config, config_source=config_source)

    splits = build_bootstrap_splits(config.bootstrap, seed=config.seeds[0])
    dataset_hash = f"{config.dataset.name}:{config.dataset.version}"
    artifacts.write_splits(splits, dataset_hash=dataset_hash)

    counts = splits.counts()
    return BootstrapResult(
        run_dir=run_dir,
        labeled_count=counts["labeled"],
        unlabeled_count=counts["unlabeled"],
        val_count=counts["val"],
        test_count=counts["test"],
    )


def build_run_dir(output_root: str, experiment_name: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / experiment_name / f"{timestamp}_seed{seed}"


def build_bootstrap_splits(bootstrap: BootstrapConfig, seed: int) -> DatasetSplits:
    ids = [f"sample_{index:06d}" for index in range(bootstrap.pool_size)]
    rng = random.Random(seed)
    rng.shuffle(ids)

    labeled_end = bootstrap.initial_labeled_size
    val_end = labeled_end + bootstrap.val_size
    test_end = val_end + bootstrap.test_size
    splits = DatasetSplits(
        labeled=ids[:labeled_end],
        val=ids[labeled_end:val_end],
        test=ids[val_end:test_end],
        unlabeled=ids[test_end:],
    )
    splits.validate()
    return splits
