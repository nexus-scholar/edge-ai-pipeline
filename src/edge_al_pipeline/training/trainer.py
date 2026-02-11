from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class Trainer(Protocol):
    """Training orchestration contract used by the active learning loop."""

    def run_training(
        self, round_index: int, seed: int, labeled_ids: Sequence[str]
    ) -> Mapping[str, float]:
        ...
