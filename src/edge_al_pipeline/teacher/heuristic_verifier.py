from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from edge_al_pipeline.contracts import SelectionCandidate


@dataclass
class HeuristicTeacherVerifier:
    """Simple reranker that down-weights clutter and promotes joint uncertainty."""

    max_detection_count: int = 120
    rerank_alpha: float = 0.5
    name: str = "heuristic_teacher"

    def verify(
        self, round_index: int, candidates: Sequence[SelectionCandidate], k: int
    ) -> list[SelectionCandidate]:
        del round_index
        if k <= 0:
            return []
        if not candidates:
            return []

        filtered = [
            candidate
            for candidate in candidates
            if int(candidate.metadata.get("det_count", 0)) <= self.max_detection_count
        ]
        if not filtered:
            filtered = list(candidates)

        rescored = []
        for candidate in filtered:
            cls_unc = float(candidate.metadata.get("uncertainty_classification", candidate.score))
            loc_unc = float(candidate.metadata.get("uncertainty_localization", 0.0))
            teacher_score = (self.rerank_alpha * cls_unc) + (
                (1.0 - self.rerank_alpha) * loc_unc
            )
            metadata = dict(candidate.metadata)
            metadata["teacher_score"] = teacher_score
            rescored.append(
                SelectionCandidate(
                    sample_id=candidate.sample_id,
                    score=teacher_score,
                    embedding=candidate.embedding,
                    metadata=metadata,
                )
            )

        ranked = sorted(rescored, key=lambda item: item.score, reverse=True)
        return ranked[: min(k, len(ranked))]
