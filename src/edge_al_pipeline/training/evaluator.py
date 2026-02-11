from __future__ import annotations

from typing import Mapping, Protocol


class Evaluator(Protocol):
    """Evaluation contract used after each AL round."""

    def evaluate(self, round_index: int, seed: int) -> Mapping[str, float]:
        ...
