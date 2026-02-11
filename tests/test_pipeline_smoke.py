from __future__ import annotations

from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.config import BootstrapConfig, DatasetConfig, ExperimentConfig
from edge_al_pipeline.contracts import DatasetSplits, SelectionCandidate
from edge_al_pipeline.data_pool import DataPoolManager
from edge_al_pipeline.pipeline import ActiveLearningPipeline
from edge_al_pipeline.profiling.edge_profiler import EdgeProfiler
from edge_al_pipeline.strategies import build_strategy


class DummyRunner:
    name = "dummy"

    def train_round(self, round_index: int, seed: int, labeled_ids):
        return {"loss": 0.1 + round_index, "train_accuracy": 0.8}

    def score_unlabeled(self, unlabeled_ids):
        candidates = []
        for sample_id in unlabeled_ids:
            idx = int(sample_id.replace("sample_", ""))
            candidates.append(
                SelectionCandidate(
                    sample_id=sample_id,
                    score=float(idx),
                    embedding=(float(idx % 5), float(idx % 7)),
                    metadata={},
                )
            )
        return candidates


def _example_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="pipeline_smoke",
        output_root="runs",
        dataset=DatasetConfig(
            name="fashion_mnist",
            root="data/fashion_mnist",
            version="1.0",
            task="classification",
            num_classes=10,
        ),
        model_name="dummy",
        strategy_name="entropy",
        rounds=2,
        query_size=2,
        seeds=[1],
        quantization_mode="fp32",
        teacher_enabled=False,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=20,
            initial_labeled_size=5,
            val_size=5,
            test_size=5,
        ),
    )


def test_pipeline_smoke_run_writes_round_artifacts(tmp_path):
    config = _example_config()
    run_dir = tmp_path / "run"
    artifacts = ArtifactStore(run_dir)
    artifacts.initialize(config, config_source=None)

    pool = DataPoolManager.from_splits(
        DatasetSplits(
            labeled=[f"sample_{i:06d}" for i in range(5)],
            unlabeled=[f"sample_{i:06d}" for i in range(5, 15)],
            val=[f"sample_{i:06d}" for i in range(15, 18)],
            test=[f"sample_{i:06d}" for i in range(18, 20)],
        )
    )
    artifacts.write_splits(pool.to_splits(), dataset_hash="dummy")

    pipeline = ActiveLearningPipeline(
        pool=pool,
        model_runner=DummyRunner(),
        strategy=build_strategy("entropy"),
        artifacts=artifacts,
        profiler=EdgeProfiler(device="cpu", quantization_mode="fp32"),
        query_size=2,
        teacher=None,
    )
    pipeline.run_seed(seed=1, rounds=2)

    assert (run_dir / "round_0_selected.csv").exists()
    assert (run_dir / "round_1_selected.csv").exists()
    assert len(pool.labeled_ids()) == 9
