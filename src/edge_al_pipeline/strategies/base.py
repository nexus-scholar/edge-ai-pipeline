from __future__ import annotations

from typing import Protocol, Sequence

from edge_al_pipeline.contracts import SelectionCandidate


class QueryStrategy(Protocol):
    name: str

    def select(
        self, candidates: Sequence[SelectionCandidate], k: int, seed: int | None = None
    ) -> list[SelectionCandidate]:
        ...
