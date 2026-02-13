from __future__ import annotations

import pytest

from edge_al_pipeline.contracts import SelectionCandidate
from edge_al_pipeline.strategies import DomainGuidedStrategy, build_strategy


def test_build_strategy_domain_guided_uses_params():
    strategy = build_strategy(
        "domain_guided",
        {
            "cdgp_domain_guided_weight": 0.8,
            "uncertainty_key": "uncertainty_combined",
            "domain_confusion_key": "domain_confusion",
        },
    )
    assert isinstance(strategy, DomainGuidedStrategy)
    assert strategy.name == "domain_guided"


def test_domain_guided_prefers_domain_confusing_samples():
    strategy = build_strategy(
        "domain_guided",
        {"cdgp_domain_guided_weight": 0.8},
    )
    candidates = [
        SelectionCandidate(
            sample_id="sample_1",
            score=0.60,
            metadata={"uncertainty_combined": 0.60, "domain_confusion": 0.10},
        ),
        SelectionCandidate(
            sample_id="sample_2",
            score=0.55,
            metadata={"uncertainty_combined": 0.55, "domain_confusion": 0.90},
        ),
    ]
    selected = strategy.select(candidates, k=1)
    assert [item.sample_id for item in selected] == ["sample_2"]


def test_domain_guided_falls_back_to_entropy_when_domain_signal_missing():
    strategy = build_strategy("domain_guided", {"cdgp_domain_guided_weight": 0.9})
    candidates = [
        SelectionCandidate(sample_id="sample_1", score=0.40, metadata={}),
        SelectionCandidate(sample_id="sample_2", score=0.80, metadata={}),
    ]
    selected = strategy.select(candidates, k=2)
    assert [item.sample_id for item in selected] == ["sample_2", "sample_1"]


def test_domain_guided_supports_minmax_plus_uncertainty_gated_blend():
    strategy = build_strategy(
        "domain_guided",
        {
            "cdgp_domain_guided_weight": 0.8,
            "score_normalization": "minmax",
            "blend_mode": "uncertainty_gated",
        },
    )
    candidates = [
        SelectionCandidate(
            sample_id="sample_1",
            score=0.90,
            metadata={"uncertainty_combined": 0.90, "domain_confusion": 0.10},
        ),
        SelectionCandidate(
            sample_id="sample_2",
            score=0.20,
            metadata={"uncertainty_combined": 0.20, "domain_confusion": 0.90},
        ),
    ]
    selected = strategy.select(candidates, k=2)
    assert [item.sample_id for item in selected] == ["sample_1", "sample_2"]
    assert selected[0].metadata["strategy_domain_guided_blend_mode"] == "uncertainty_gated"
    assert selected[0].metadata["strategy_domain_guided_normalization"] == "minmax"
    assert "strategy_domain_guided_score" in selected[0].metadata


def test_domain_guided_rejects_unknown_score_normalization():
    with pytest.raises(ValueError, match="score_normalization must be one of"):
        build_strategy(
            "domain_guided",
            {"score_normalization": "not_supported"},
        )


def test_domain_guided_rejects_unknown_blend_mode():
    with pytest.raises(ValueError, match="blend_mode must be one of"):
        build_strategy(
            "domain_guided",
            {"blend_mode": "not_supported"},
        )
