from edge_al_pipeline.strategies.base import QueryStrategy
from edge_al_pipeline.strategies.domain_guided import DomainGuidedStrategy
from edge_al_pipeline.strategies.entropy import EntropyStrategy
from edge_al_pipeline.strategies.k_center_greedy import KCenterGreedyStrategy
from edge_al_pipeline.strategies.random_sampling import RandomStrategy


def build_strategy(
    name: str, params: dict[str, object] | None = None
) -> QueryStrategy:
    normalized = name.strip().lower()
    params = params or {}
    if normalized == "random":
        return RandomStrategy()
    if normalized == "entropy":
        return EntropyStrategy()
    if normalized == "k_center_greedy":
        return KCenterGreedyStrategy()
    if normalized == "domain_guided":
        return DomainGuidedStrategy(
            domain_weight=float(
                params.get(
                    "cdgp_domain_guided_weight",
                    params.get("domain_weight", 0.5),
                )
            ),
            uncertainty_key=str(
                params.get("uncertainty_key", "uncertainty_combined")
            ),
            domain_confusion_key=str(
                params.get("domain_confusion_key", "domain_confusion")
            ),
            score_normalization=str(
                params.get(
                    "cdgp_score_normalization",
                    params.get("score_normalization", "none"),
                )
            ),
            blend_mode=str(
                params.get(
                    "cdgp_blend_mode",
                    params.get("blend_mode", "linear"),
                )
            ),
        )
    raise ValueError(
        "Unknown strategy. Expected one of: random, entropy, "
        "k_center_greedy, domain_guided."
    )


__all__ = [
    "build_strategy",
    "DomainGuidedStrategy",
    "EntropyStrategy",
    "KCenterGreedyStrategy",
    "QueryStrategy",
    "RandomStrategy",
]
