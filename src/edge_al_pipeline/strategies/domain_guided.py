from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from edge_al_pipeline.contracts import SelectionCandidate

_VALID_SCORE_NORMALIZATION = frozenset({"none", "minmax", "rank"})
_VALID_BLEND_MODES = frozenset({"linear", "uncertainty_gated"})


class DomainGuidedStrategy:
    name = "domain_guided"

    def __init__(
        self,
        domain_weight: float = 0.5,
        uncertainty_key: str = "uncertainty_combined",
        domain_confusion_key: str = "domain_confusion",
        score_normalization: str = "none",
        blend_mode: str = "linear",
    ) -> None:
        if domain_weight < 0.0 or domain_weight > 1.0:
            raise ValueError("domain_weight must be in [0.0, 1.0].")
        normalized_score_mode = score_normalization.strip().lower()
        if normalized_score_mode not in _VALID_SCORE_NORMALIZATION:
            raise ValueError(
                "score_normalization must be one of "
                f"{sorted(_VALID_SCORE_NORMALIZATION)}."
            )
        normalized_blend_mode = blend_mode.strip().lower()
        if normalized_blend_mode not in _VALID_BLEND_MODES:
            raise ValueError(
                "blend_mode must be one of "
                f"{sorted(_VALID_BLEND_MODES)}."
            )
        self._domain_weight = domain_weight
        self._uncertainty_key = uncertainty_key
        self._domain_confusion_key = domain_confusion_key
        self._score_normalization = normalized_score_mode
        self._blend_mode = normalized_blend_mode

    def select(
        self, candidates: Sequence[SelectionCandidate], k: int, seed: int | None = None
    ) -> list[SelectionCandidate]:
        del seed
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if not candidates:
            return []

        uncertainty_scores = [
            _read_metadata_score(
                candidate,
                self._uncertainty_key,
                fallback=candidate.score,
            )
            for candidate in candidates
        ]
        domain_confusion_scores = [
            _read_metadata_score(
                candidate,
                self._domain_confusion_key,
                fallback=_read_metadata_score(
                    candidate,
                    "domain_score",
                    fallback=uncertainty_scores[index],
                ),
            )
            for index, candidate in enumerate(candidates)
        ]
        sample_ids = [candidate.sample_id for candidate in candidates]
        normalized_uncertainty = _normalize_scores(
            uncertainty_scores,
            sample_ids=sample_ids,
            mode=self._score_normalization,
        )
        normalized_domain_confusion = _normalize_scores(
            domain_confusion_scores,
            sample_ids=sample_ids,
            mode=self._score_normalization,
        )

        scored: list[tuple[float, SelectionCandidate]] = []
        for index, candidate in enumerate(candidates):
            uncertainty = normalized_uncertainty[index]
            domain_confusion = normalized_domain_confusion[index]
            combined_score = self._combined_score(uncertainty, domain_confusion)
            metadata = dict(candidate.metadata)
            metadata["strategy_domain_guided_score"] = float(combined_score)
            metadata["strategy_domain_guided_uncertainty"] = float(uncertainty)
            metadata["strategy_domain_guided_domain_confusion"] = float(
                domain_confusion
            )
            metadata["strategy_domain_guided_blend_mode"] = self._blend_mode
            metadata["strategy_domain_guided_normalization"] = (
                self._score_normalization
            )
            scored.append((combined_score, replace(candidate, metadata=metadata)))

        ranked = sorted(
            scored,
            key=lambda item: (
                item[0],
                _read_metadata_score(
                    item[1],
                    self._uncertainty_key,
                    fallback=item[1].score,
                ),
                item[1].score,
                item[1].sample_id,
            ),
            reverse=True,
        )
        selected = ranked[: min(k, len(ranked))]
        return [item[1] for item in selected]

    def _combined_score(self, uncertainty: float, domain_confusion: float) -> float:
        if self._blend_mode == "uncertainty_gated":
            return uncertainty * (1.0 + (self._domain_weight * domain_confusion))
        return ((1.0 - self._domain_weight) * uncertainty) + (
            self._domain_weight * domain_confusion
        )


def _read_metadata_score(
    candidate: SelectionCandidate, key: str, fallback: float
) -> float:
    raw_value = candidate.metadata.get(key, fallback)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return float(fallback)


def _normalize_scores(
    values: Sequence[float],
    sample_ids: Sequence[str],
    mode: str,
) -> list[float]:
    if mode == "none":
        return [float(value) for value in values]
    if mode == "minmax":
        return _minmax_normalize(values)
    if mode == "rank":
        return _rank_normalize(values, sample_ids)
    raise ValueError(f"Unsupported score normalization mode: {mode}")


def _minmax_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    data = [float(value) for value in values]
    minimum = min(data)
    maximum = max(data)
    if maximum <= minimum:
        return [0.5 for _ in data]
    scale = maximum - minimum
    return [(value - minimum) / scale for value in data]


def _rank_normalize(values: Sequence[float], sample_ids: Sequence[str]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    order = sorted(
        range(len(values)),
        key=lambda index: (float(values[index]), str(sample_ids[index])),
    )
    ranks = [0.0 for _ in values]
    denominator = float(len(values) - 1)
    for rank, index in enumerate(order):
        ranks[index] = float(rank) / denominator
    return ranks
