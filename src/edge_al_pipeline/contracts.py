from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetSplits:
    labeled: list[str] = field(default_factory=list)
    unlabeled: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)

    def validate(self) -> None:
        combined = self.labeled + self.unlabeled + self.val + self.test
        unique = set(combined)
        if len(combined) != len(unique):
            raise ValueError("Duplicate sample IDs detected across splits.")

    def counts(self) -> dict[str, int]:
        return {
            "labeled": len(self.labeled),
            "unlabeled": len(self.unlabeled),
            "val": len(self.val),
            "test": len(self.test),
        }


@dataclass(frozen=True)
class SelectionCandidate:
    sample_id: str
    score: float
    embedding: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionRecord:
    round_index: int
    seed: int
    strategy: str
    sample_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRecord:
    round_index: int
    seed: int
    split: str
    metric: str
    value: float


@dataclass(frozen=True)
class ProfileRecord:
    round_index: int
    stage: str
    latency_ms: float
    memory_mb: float | None
    quantization_mode: str
    device: str
    notes: str = ""
