from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from edge_al_pipeline.contracts import ProfileRecord


class EdgeProfiler:
    """Collects stage latency records for each AL round."""

    def __init__(self, device: str, quantization_mode: str) -> None:
        self._device = device
        self._quantization_mode = quantization_mode
        self._records: list[ProfileRecord] = []

    @contextmanager
    def measure(self, round_index: int, stage: str, notes: str = "") -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._records.append(
                ProfileRecord(
                    round_index=round_index,
                    stage=stage,
                    latency_ms=elapsed_ms,
                    memory_mb=None,
                    quantization_mode=self._quantization_mode,
                    device=self._device,
                    notes=notes,
                )
            )

    def flush(self) -> list[ProfileRecord]:
        records = list(self._records)
        self._records.clear()
        return records
