"""Linear Thompson Sampling: a Bayesian alternative to LinUCB.

Where LinUCB scores each candidate with a point estimate plus an explicit
confidence bonus, Thompson Sampling instead keeps a full Gaussian posterior
over the shared coefficient vector theta and, at every round, draws one
sample theta_tilde from it to score every candidate with. Exploration falls
out naturally from posterior sampling rather than from a hand-tuned bonus
term. The posterior update is closed-form conjugate Bayesian linear
regression -- no gradient descent, no neural network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThompsonSamplingShared:
    """Bayesian linear regression bandit with a shared coefficient posterior.

    Attributes:
        n_features: Context dimensionality.
        noise_variance: Assumed observation noise variance (sigma^2).
        prior_precision: Prior precision (inverse variance) on theta; larger
            values pull theta towards zero more strongly before data arrives.
        random_state: Seed for posterior sampling.
    """

    n_features: int
    noise_variance: float = 0.25
    prior_precision: float = 1.0
    random_state: int = 0

    precision: np.ndarray = field(init=False)  # Sigma^{-1}
    mean_times_precision: np.ndarray = field(init=False)  # Sigma^{-1} @ mu
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.precision = self.prior_precision * np.eye(self.n_features)
        self.mean_times_precision = np.zeros(self.n_features)
        self._rng = np.random.default_rng(self.random_state)

    def _posterior_mean_and_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        covariance = np.linalg.inv(self.precision)
        mean = covariance @ self.mean_times_precision
        return mean, covariance

    def select_arm(self, candidate_contexts: np.ndarray) -> int:
        """Sample one posterior draw of theta and score every candidate with it.

        Args:
            candidate_contexts: ``(pool_size, n_features)`` context matrix.

        Returns:
            The chosen position within the pool.
        """
        mean, covariance = self._posterior_mean_and_covariance()
        theta_sample = self._rng.multivariate_normal(mean, covariance)
        scores = candidate_contexts @ theta_sample
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, reward: float) -> None:
        """Conjugate Bayesian linear regression update for one observation."""
        self.precision += np.outer(context, context) / self.noise_variance
        self.mean_times_precision += (reward / self.noise_variance) * context
