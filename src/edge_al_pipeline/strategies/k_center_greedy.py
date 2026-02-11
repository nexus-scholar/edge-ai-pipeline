from __future__ import annotations

from typing import Sequence

from edge_al_pipeline.contracts import SelectionCandidate
from edge_al_pipeline.strategies.entropy import EntropyStrategy


class KCenterGreedyStrategy:
    name = "k_center_greedy"

    def select(
        self, candidates: Sequence[SelectionCandidate], k: int, seed: int | None = None
    ) -> list[SelectionCandidate]:
        del seed
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if not candidates:
            return []

        with_embeddings = [item for item in candidates if item.embedding is not None]
        if not with_embeddings:
            return EntropyStrategy().select(candidates, k)

        target_size = min(k, len(candidates))
        selected: list[SelectionCandidate] = []

        first = max(with_embeddings, key=lambda item: item.score)
        selected.append(first)

        remaining = [item for item in with_embeddings if item.sample_id != first.sample_id]
        min_distance = {
            item.sample_id: _squared_distance(item.embedding, first.embedding)
            for item in remaining
        }

        while remaining and len(selected) < min(k, len(with_embeddings)):
            next_item = max(remaining, key=lambda item: min_distance[item.sample_id])
            selected.append(next_item)
            remaining = [item for item in remaining if item.sample_id != next_item.sample_id]
            for item in remaining:
                distance = _squared_distance(item.embedding, next_item.embedding)
                if distance < min_distance[item.sample_id]:
                    min_distance[item.sample_id] = distance

        if len(selected) < target_size:
            selected_ids = {item.sample_id for item in selected}
            ranked = EntropyStrategy().select(candidates, target_size)
            for item in ranked:
                if item.sample_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.sample_id)
                if len(selected) == target_size:
                    break

        return selected[:target_size]


def _squared_distance(
    first: tuple[float, ...] | None, second: tuple[float, ...] | None
) -> float:
    if first is None or second is None:
        return float("inf")
    if len(first) != len(second):
        return float("inf")
    return sum((lhs - rhs) * (lhs - rhs) for lhs, rhs in zip(first, second))
