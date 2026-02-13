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

    # Inspect dataset to get real IDs if possible
    from edge_al_pipeline.models.image_folder_mobilenet_runner import ImageFolderMobileNetRunner
    # We can't easily get the IDs without instantiating the dataset or walking the dir.
    # Let's walk the directory since ImageFolderMobileNetRunner.inspect_dataset only returns counts.
    
    dataset_root = Path(config.dataset.root)
    if dataset_root.exists() and dataset_root.is_dir():
        # Mimic ImageFolder logic: find all files
        # Or better, let's just make inspect_dataset return the IDs? 
        # No, let's keep it simple. If it's a real run, we walk.
        # If it's a dry run/test, we might fall back to synthetic.
        
        # Actually, ImageFolderMobileNetRunner logic uses Path(path).name as ID.
        # Let's collect those.
        all_ids = []
        # Sorted walk for determinism
        for class_dir in sorted(dataset_root.iterdir()):
            if class_dir.is_dir():
                for image_file in sorted(class_dir.iterdir()):
                    if image_file.is_file(): # and extension check?
                         all_ids.append(image_file.name)
    else:
        # Fallback to synthetic for tests/dry runs
        all_ids = [f"sample_{index:06d}" for index in range(config.bootstrap.pool_size)]

    splits = build_bootstrap_splits(config.bootstrap, all_ids, seed=config.seeds[0])
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


def build_bootstrap_splits(bootstrap: BootstrapConfig, all_ids: list[str], seed: int) -> DatasetSplits:
    # Ensure we have enough data (or clamp)
    # The config validator checks this, but we should be safe.
    
    # If we have real IDs, use them.
    # If len(all_ids) < bootstrap.pool_size, we should probably use all available IDs
    # effectively acting as the pool.
    
    # Note: bootstrap.pool_size in config is a "request". 
    # If the real dataset is smaller, we just use what we have.
    # If it's larger, we subsample.
    
    ids = list(all_ids) # Copy
    rng = random.Random(seed)
    rng.shuffle(ids)
    
    # Limit to pool_size if requested (e.g. for debugging/speed)
    if len(ids) > bootstrap.pool_size:
        ids = ids[:bootstrap.pool_size]

    labeled_end = bootstrap.initial_labeled_size
    val_end = labeled_end + bootstrap.val_size
    test_end = val_end + bootstrap.test_size
    
    if test_end > len(ids):
         # This shouldn't happen if config is valid AND dataset matches config
         # But let's fail gracefully or loud?
         # Failing loud is better for the user.
         raise ValueError(
             f"Dataset has {len(ids)} samples, but bootstrap config requires {test_end} "
             "(initial + val + test)."
         )

    splits = DatasetSplits(
        labeled=ids[:labeled_end],
        val=ids[labeled_end:val_end],
        test=ids[val_end:test_end],
        unlabeled=ids[test_end:],
    )
    splits.validate()
    return splits
