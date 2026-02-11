from __future__ import annotations

from pathlib import Path

from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.config import (
    BootstrapConfig,
    DatasetConfig,
    ExperimentConfig,
)
from edge_al_pipeline.contracts import (
    DatasetSplits,
    MetricRecord,
    ProfileRecord,
    SelectionRecord,
)


def _example_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="artifact_test",
        output_root="runs",
        dataset=DatasetConfig(
            name="fashion_mnist",
            root="data/fashion_mnist",
            version="1.0",
            task="classification",
            num_classes=10,
        ),
        model_name="simple_cnn",
        model_params={},
        strategy_name="entropy",
        rounds=1,
        query_size=4,
        seeds=[1],
        quantization_mode="fp32",
        teacher_enabled=False,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=20,
            initial_labeled_size=4,
            val_size=4,
            test_size=4,
        ),
    )


def test_artifact_store_writes_contract_files(tmp_path):
    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir)
    store.initialize(_example_config(), config_source=Path("configs/example.json"))

    splits = DatasetSplits(
        labeled=["sample_000000", "sample_000001"],
        unlabeled=["sample_000002"],
        val=["sample_000003"],
        test=["sample_000004"],
    )
    store.write_splits(splits, dataset_hash="fashion_mnist:1.0")
    store.write_round_selection(
        round_index=0,
        records=[
            SelectionRecord(
                round_index=0,
                seed=1,
                strategy="entropy",
                sample_id="sample_000002",
                score=0.9,
                metadata={"uncertainty": "entropy"},
            )
        ],
    )
    store.append_metrics(
        [MetricRecord(round_index=0, seed=1, split="train", metric="loss", value=0.5)]
    )
    store.append_profile(
        [
            ProfileRecord(
                round_index=0,
                stage="score_unlabeled",
                latency_ms=12.3,
                memory_mb=None,
                quantization_mode="fp32",
                device="cpu",
                notes="smoke",
            )
        ]
    )

    assert (run_dir / "splits.json").exists()
    assert (run_dir / "round_0_selected.csv").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "profile.csv").exists()
    assert (run_dir / "checkpoints").is_dir()
