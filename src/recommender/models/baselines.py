"""Non-contextual bandit baselines: Random and popularity-greedy.

Both ignore the user profile and article content entirely -- exactly the
signal the context-aware bandits are expected to exploit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomPolicy:
    """Picks uniformly among the pool, ignoring context."""

    random_state: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)

    def select_arm(self, candidate_contexts: np.ndarray) -> int:
        return int(self._rng.integers(candidate_contexts.shape[0]))

    def update(self, context: np.ndarray, reward: float) -> None:  # noqa: ARG002
        return


@dataclass
class PopularityGreedy:
    """Always picks the pool candidate with the highest popularity feature.

    The popularity feature (second-to-last context column, see
    ``recommender.data.rounds.build_rounds``) is the same running,
    point-in-time statistic the contextual bandits see -- this baseline
    isolates how much the content/profile features add *beyond* popularity.
    """

    popularity_column: int = -2

    def select_arm(self, candidate_contexts: np.ndarray) -> int:
        return int(np.argmax(candidate_contexts[:, self.popularity_column]))

    def update(self, context: np.ndarray, reward: float) -> None:  # noqa: ARG002
        return
