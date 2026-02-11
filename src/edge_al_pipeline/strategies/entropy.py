from __future__ import annotations

from typing import Sequence

from edge_al_pipeline.contracts import SelectionCandidate


class EntropyStrategy:
    name = "entropy"

    def select(
        self, candidates: Sequence[SelectionCandidate], k: int, seed: int | None = None
    ) -> list[SelectionCandidate]:
        del seed
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        return ranked[: min(k, len(ranked))]
