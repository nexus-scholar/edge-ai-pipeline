from __future__ import annotations

from typing import Sequence

from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.contracts import MetricRecord, SelectionRecord
from edge_al_pipeline.data_pool import DataPoolManager
from edge_al_pipeline.models import ModelRunner
from edge_al_pipeline.profiling import EdgeProfiler
from edge_al_pipeline.strategies.base import QueryStrategy
from edge_al_pipeline.teacher import TeacherVerifier


class ActiveLearningPipeline:
    """Round orchestration scaffold; model/dataset implementations plug into this."""

    def __init__(
        self,
        pool: DataPoolManager,
        model_runner: ModelRunner,
        strategy: QueryStrategy,
        artifacts: ArtifactStore,
        profiler: EdgeProfiler,
        query_size: int,
        teacher: TeacherVerifier | None = None,
    ) -> None:
        self.pool = pool
        self.model_runner = model_runner
        self.strategy = strategy
        self.artifacts = artifacts
        self.profiler = profiler
        self.query_size = query_size
        self.teacher = teacher

    def run_round(self, round_index: int, seed: int) -> list[str]:
        with self.profiler.measure(round_index, "score_unlabeled"):
            candidates = self.model_runner.score_unlabeled(self.pool.unlabeled_ids())

        with self.profiler.measure(round_index, "select_candidates"):
            selected = self.strategy.select(candidates, self.query_size, seed=seed)

        if self.teacher is not None:
            with self.profiler.measure(round_index, "teacher_verify"):
                selected = self.teacher.verify(round_index, selected, self.query_size)

        selected_ids = [candidate.sample_id for candidate in selected]
        acquired_ids = self.pool.acquire(selected_ids)

        records = [
            SelectionRecord(
                round_index=round_index,
                seed=seed,
                strategy=self.strategy.name,
                sample_id=item.sample_id,
                score=item.score,
                metadata=item.metadata,
            )
            for item in selected
        ]
        self.artifacts.write_round_selection(round_index, records)
        self.artifacts.append_profile(self.profiler.flush())
        return acquired_ids

    def persist_metrics(
        self, round_index: int, seed: int, split: str, metrics: dict[str, float]
    ) -> None:
        self.artifacts.append_metrics(
            [
                MetricRecord(
                    round_index=round_index,
                    seed=seed,
                    split=split,
                    metric=name,
                    value=value,
                )
                for name, value in metrics.items()
            ]
        )

    def run_seed(self, seed: int, rounds: int) -> Sequence[str]:
        for round_index in range(rounds):
            print(f"  > AL Round {round_index}/{rounds - 1} (Seed {seed})")
            with self.profiler.measure(round_index, "train_round"):
                train_metrics = self.model_runner.train_round(
                    round_index=round_index,
                    seed=seed,
                    labeled_ids=self.pool.labeled_ids(),
                )
            print(f"    Train Metrics: {train_metrics}")
            self.persist_metrics(round_index, seed, "train", dict(train_metrics))
            self.artifacts.append_profile(self.profiler.flush())
            acquired = self.run_round(round_index=round_index, seed=seed)
            if not acquired:
                break
        return self.pool.labeled_ids()
