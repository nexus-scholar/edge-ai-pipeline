from __future__ import annotations

from edge_al_pipeline.contracts import DatasetSplits
from edge_al_pipeline.data_pool import DataPoolManager


def test_data_pool_acquire_moves_only_unlabeled_ids():
    pool = DataPoolManager.from_splits(
        DatasetSplits(
            labeled=["sample_000001"],
            unlabeled=["sample_000002", "sample_000003"],
            val=["sample_000004"],
            test=["sample_000005"],
        )
    )
    acquired = pool.acquire(["sample_000003", "sample_999999"])
    assert acquired == ["sample_000003"]

    splits = pool.to_splits()
    assert "sample_000003" in splits.labeled
    assert "sample_000003" not in splits.unlabeled
