from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from edge_al_pipeline.contracts import SelectionCandidate


class ModelRunner(Protocol):
    """Minimal contract for AL model implementations."""

    name: str

    def train_round(
        self, round_index: int, seed: int, labeled_ids: Sequence[str]
    ) -> Mapping[str, float]:
        """Train or warm-start on the labeled set and return training metrics."""

    def score_unlabeled(self, unlabeled_ids: Sequence[str]) -> list[SelectionCandidate]:
        """Return uncertainty/informativeness scores for unlabeled IDs."""
