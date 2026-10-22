"""Online replay, frozen-policy evaluation, and IPS off-policy evaluation.

``replay_train``
    Streams training rounds in order, updating the policy after each one.
    This is direct online learning: the policy's own choices generate its
    own training signal (reward observed iff it happens to pick the same
    article the logging policy actually showed).

``evaluate_frozen``
    Streams held-out rounds through an already-trained, now-frozen policy,
    exactly as in ``crop-contextual-recommender``: this measures reward
    *conditional on* the policy happening to match the logged action, which
    discards every round where it did not (replay evaluation, Li et al.,
    2011).

``evaluate_ips``
    Off-policy evaluation via inverse propensity scoring: instead of
    discarding non-matching rounds, every round contributes
    ``I(match) * reward / propensity``, which is an unbiased estimator of
    the policy's expected reward under the (known) logging distribution,
    using *every* round rather than only the ones that happened to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from recommender.data.rounds import RoundsData

#: Floor applied to propensities before dividing, the standard stabiliser
#: for IPS: without it, a round with a near-zero logging probability could
#: dominate the estimate with a huge, high-variance weight.
PROPENSITY_FLOOR = 0.01


class BanditPolicy(Protocol):
    def select_arm(self, candidate_contexts: np.ndarray) -> int: ...
    def update(self, context: np.ndarray, reward: float) -> None: ...


def replay_train(policy: BanditPolicy, rounds: RoundsData, indices: np.ndarray) -> np.ndarray:
    """Stream rounds through ``policy``, updating it after each one.

    Args:
        policy: A bandit exposing ``select_arm`` and ``update``.
        rounds: The full rounds data.
        indices: Round indices to replay, in the order they are replayed.

    Returns:
        ``(len(indices),)`` binary reward obtained at each round.
    """
    rewards = np.empty(indices.shape[0])
    for i, round_idx in enumerate(indices):
        contexts = rounds.context[round_idx]
        position = policy.select_arm(contexts)
        reward = float(position == rounds.logged_position[round_idx])
        policy.update(contexts[position], reward)
        rewards[i] = reward
    return rewards


@dataclass(frozen=True)
class FrozenEvaluationReport:
    """Replay (match-conditional) generalisation metrics.

    Attributes:
        model: Name of the evaluated policy.
        match_rate: Fraction of rounds where the chosen position equals the
            logged position.
        regret: ``1 - match_rate``, against an oracle that always matches.
        n_rounds: Number of rounds evaluated.
    """

    model: str
    match_rate: float
    regret: float
    n_rounds: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "match_rate": round(self.match_rate, 4),
            "regret": round(self.regret, 4),
            "n_rounds": self.n_rounds,
        }


def evaluate_frozen(
    policy: BanditPolicy, rounds: RoundsData, indices: np.ndarray, model_name: str = "model"
) -> FrozenEvaluationReport:
    """Replay held-out rounds through ``policy`` without updating it.

    Args:
        policy: A fitted bandit exposing ``select_arm``.
        rounds: The full rounds data.
        indices: Held-out round indices.
        model_name: Label attached to the report.

    Returns:
        A :class:`FrozenEvaluationReport`.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    if indices.size == 0:
        raise ValueError("indices must not be empty.")

    matches = np.empty(indices.shape[0])
    for i, round_idx in enumerate(indices):
        position = policy.select_arm(rounds.context[round_idx])
        matches[i] = float(position == rounds.logged_position[round_idx])

    match_rate = float(matches.mean())
    return FrozenEvaluationReport(
        model=model_name, match_rate=match_rate, regret=1.0 - match_rate, n_rounds=indices.shape[0]
    )


@dataclass(frozen=True)
class IPSReport:
    """Inverse-propensity-scored off-policy value estimate.

    Attributes:
        model: Name of the evaluated policy.
        value_estimate: The IPS estimator itself -- an unbiased estimate of
            the policy's expected per-round reward under the logging
            distribution, assuming correctly specified propensities.
        match_rate: Raw (unweighted) match rate, reported alongside for
            comparison with :func:`evaluate_frozen`.
        effective_sample_size: ``(sum of weights)^2 / sum of weights^2``
            among matching rounds -- a diagnostic for how much a handful of
            low-propensity rounds dominate the estimate.
        n_rounds: Number of rounds the estimate is computed over.
    """

    model: str
    value_estimate: float
    match_rate: float
    effective_sample_size: float
    n_rounds: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "ips_value_estimate": round(self.value_estimate, 4),
            "match_rate": round(self.match_rate, 4),
            "effective_sample_size": round(self.effective_sample_size, 2),
            "n_rounds": self.n_rounds,
        }


def evaluate_ips(
    policy: BanditPolicy, rounds: RoundsData, indices: np.ndarray, model_name: str = "model"
) -> IPSReport:
    """Estimate a frozen policy's value via inverse propensity scoring.

    Args:
        policy: A fitted bandit exposing ``select_arm``.
        rounds: The full rounds data, including logged propensities.
        indices: Round indices to score (typically held-out).
        model_name: Label attached to the report.

    Returns:
        An :class:`IPSReport`.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    if indices.size == 0:
        raise ValueError("indices must not be empty.")

    weights = np.zeros(indices.shape[0])
    matches = np.zeros(indices.shape[0])
    for i, round_idx in enumerate(indices):
        position = policy.select_arm(rounds.context[round_idx])
        is_match = position == rounds.logged_position[round_idx]
        matches[i] = float(is_match)
        if is_match:
            propensity = max(rounds.propensity[round_idx], PROPENSITY_FLOOR)
            weights[i] = 1.0 / propensity  # reward is always 1 on a match

    value_estimate = float(weights.mean())
    matching_weights = weights[matches.astype(bool)]
    ess = (
        float(matching_weights.sum() ** 2 / (matching_weights**2).sum())
        if matching_weights.size
        else 0.0
    )
    return IPSReport(
        model=model_name,
        value_estimate=value_estimate,
        match_rate=float(matches.mean()),
        effective_sample_size=ess,
        n_rounds=indices.shape[0],
    )
