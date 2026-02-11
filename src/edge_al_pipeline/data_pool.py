from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from edge_al_pipeline.contracts import DatasetSplits


@dataclass
class DataPoolManager:
    """Tracks L/U/V/T sets and moves queried IDs from U to L."""

    _labeled: set[str]
    _unlabeled: set[str]
    _val: set[str]
    _test: set[str]

    @classmethod
    def from_splits(cls, splits: DatasetSplits) -> "DataPoolManager":
        splits.validate()
        return cls(
            _labeled=set(splits.labeled),
            _unlabeled=set(splits.unlabeled),
            _val=set(splits.val),
            _test=set(splits.test),
        )

    def acquire(self, sample_ids: Sequence[str]) -> list[str]:
        acquired: list[str] = []
        for sample_id in sample_ids:
            if sample_id in self._unlabeled:
                self._unlabeled.remove(sample_id)
                self._labeled.add(sample_id)
                acquired.append(sample_id)
        return acquired

    def to_splits(self) -> DatasetSplits:
        return DatasetSplits(
            labeled=sorted(self._labeled),
            unlabeled=sorted(self._unlabeled),
            val=sorted(self._val),
            test=sorted(self._test),
        )

    def labeled_ids(self) -> list[str]:
        return sorted(self._labeled)

    def unlabeled_ids(self) -> list[str]:
        return sorted(self._unlabeled)
