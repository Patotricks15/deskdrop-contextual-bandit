"""Tests for LinUCB, Thompson Sampling, and the baselines.

Learning-quality assertions use a hand-crafted, deterministic scenario
(two orthogonal content topics, one of them clearly preferred by the
user profile) rather than aggregate performance on a small random
fixture: with only a few dozen held-out rounds, a statistical "beats
random" comparison on noisy synthetic data is flaky across environments
(numpy/scikit-learn minor version differences shift the exact random
draws). A deterministic scenario with a known correct answer is immune to
that.
"""

from __future__ import annotations

import numpy as np

from recommender.models import LinUCBShared, PopularityGreedy, RandomPolicy, ThompsonSamplingShared


def _toy_contexts() -> tuple[np.ndarray, np.ndarray]:
    """Two candidates: one whose content matches the user profile, one that doesn't.

    Context layout matches ``recommender.data.rounds.build_rounds``:
    ``[profile ⊙ article, article, popularity, bias]``. Article A's vector
    equals the profile (interaction term large and positive); article B is
    orthogonal to it (interaction term zero).
    """
    profile = np.array([1.0, 0.0])
    article_a = np.array([1.0, 0.0])
    article_b = np.array([0.0, 1.0])

    def build(article: np.ndarray) -> np.ndarray:
        return np.concatenate([profile * article, article, [0.0, 1.0]])

    return build(article_a), build(article_b)


def test_linucb_learns_to_prefer_the_matching_article():
    context_a, context_b = _toy_contexts()
    policy = LinUCBShared(n_features=context_a.shape[0], alpha=0.0)
    for _ in range(50):
        policy.update(context_a, reward=1.0)
        policy.update(context_b, reward=0.0)

    chosen = policy.select_arm(np.stack([context_b, context_a]))
    assert chosen == 1  # position 1 is article A, the matching one


def test_thompson_sampling_posterior_mean_prefers_the_matching_article():
    context_a, context_b = _toy_contexts()
    policy = ThompsonSamplingShared(n_features=context_a.shape[0], random_state=0)
    for _ in range(50):
        policy.update(context_a, reward=1.0)
        policy.update(context_b, reward=0.0)

    mean, _ = policy._posterior_mean_and_covariance()
    assert context_a @ mean > context_b @ mean


def test_popularity_greedy_picks_highest_popularity_feature():
    policy = PopularityGreedy()
    contexts = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.2, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.9, 1.0],
        ]
    )
    assert policy.select_arm(contexts) == 1


def test_random_policy_always_returns_a_valid_position():
    policy = RandomPolicy(random_state=0)
    contexts = np.zeros((4, 3))
    for _ in range(20):
        assert 0 <= policy.select_arm(contexts) < 4


def test_linucb_update_changes_posterior():
    context_a, _ = _toy_contexts()
    policy = LinUCBShared(n_features=context_a.shape[0], alpha=1.0)
    a_before = policy.A.copy()
    policy.update(context_a, reward=1.0)
    assert not np.allclose(a_before, policy.A)


def test_thompson_sampling_update_increases_precision():
    context_a, _ = _toy_contexts()
    policy = ThompsonSamplingShared(n_features=context_a.shape[0])
    precision_before = policy.precision.copy()
    policy.update(context_a, reward=1.0)
    assert np.trace(policy.precision) > np.trace(precision_before)
