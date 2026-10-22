"""Shared-parameter (hybrid) LinUCB for a dynamically changing arm set.

``crop-contextual-recommender`` keeps one ridge-regression posterior per
arm because its 22 crops are fixed and every round offers all of them.
Here the candidate pool is a *different* set of articles every round, so a
disjoint per-arm model would never accumulate enough data per article.
Instead a single shared posterior ``(A, b)`` is learned over the context
features themselves (content similarity, running popularity, user
profile), exactly the classical "hybrid" LinUCB formulation (Li et al.,
2010) collapsed to its shared-feature-only case: one linear model scores
any candidate's feature vector, regardless of which specific article it
represents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinUCBShared:
    """Single ridge-regression posterior shared across every candidate article.

    Attributes:
        n_features: Context dimensionality.
        alpha: Exploration strength.
    """

    n_features: int
    alpha: float = 1.0

    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.A = np.eye(self.n_features)
        self.b = np.zeros(self.n_features)

    def select_arm(self, candidate_contexts: np.ndarray) -> int:
        """Pick the pool position with the highest UCB score.

        Args:
            candidate_contexts: ``(pool_size, n_features)`` context matrix
                for this round's candidates.

        Returns:
            The chosen position within the pool.
        """
        a_inv = np.linalg.inv(self.A)
        theta = a_inv @ self.b
        means = candidate_contexts @ theta
        bonuses = self.alpha * np.sqrt(
            np.einsum("ij,jk,ik->i", candidate_contexts, a_inv, candidate_contexts)
        )
        return int(np.argmax(means + bonuses))

    def update(self, context: np.ndarray, reward: float) -> None:
        """Incorporate the chosen candidate's context and observed reward."""
        self.A += np.outer(context, context)
        self.b += reward * context
