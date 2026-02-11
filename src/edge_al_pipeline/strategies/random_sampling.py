from __future__ import annotations

import random
from typing import Sequence

from edge_al_pipeline.contracts import SelectionCandidate


class RandomStrategy:
    name = "random"

    def select(
        self, candidates: Sequence[SelectionCandidate], k: int, seed: int | None = None
    ) -> list[SelectionCandidate]:
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        pool = list(candidates)
        if not pool:
            return []
        rng = random.Random(seed)
        rng.shuffle(pool)
        return pool[: min(k, len(pool))]
