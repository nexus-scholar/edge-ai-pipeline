from edge_al_pipeline.strategies.base import QueryStrategy
from edge_al_pipeline.strategies.entropy import EntropyStrategy
from edge_al_pipeline.strategies.k_center_greedy import KCenterGreedyStrategy
from edge_al_pipeline.strategies.random_sampling import RandomStrategy


def build_strategy(name: str) -> QueryStrategy:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomStrategy()
    if normalized == "entropy":
        return EntropyStrategy()
    if normalized == "k_center_greedy":
        return KCenterGreedyStrategy()
    raise ValueError(
        "Unknown strategy. Expected one of: random, entropy, k_center_greedy."
    )


__all__ = [
    "build_strategy",
    "EntropyStrategy",
    "KCenterGreedyStrategy",
    "QueryStrategy",
    "RandomStrategy",
]
