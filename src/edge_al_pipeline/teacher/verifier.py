from __future__ import annotations

from typing import Protocol, Sequence

from edge_al_pipeline.contracts import SelectionCandidate


class TeacherVerifier(Protocol):
    """Optional fog/cloud verifier that re-ranks edge candidates."""

    name: str

    def verify(
        self, round_index: int, candidates: Sequence[SelectionCandidate], k: int
    ) -> list[SelectionCandidate]:
        ...
