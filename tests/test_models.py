"""Tests for LinUCB, Thompson Sampling, and the baselines."""

from __future__ import annotations

import numpy as np

from recommender.evaluation import evaluate_frozen, replay_train
from recommender.models import LinUCBShared, PopularityGreedy, RandomPolicy, ThompsonSamplingShared


def test_linucb_learns_better_than_random(rounds, split):
    linucb = LinUCBShared(n_features=rounds.n_features, alpha=1.0)
    replay_train(linucb, rounds, split.train_indices)
    linucb_report = evaluate_frozen(linucb, rounds, split.test_indices, "LinUCB")

    random_policy = RandomPolicy(random_state=0)
    random_report = evaluate_frozen(random_policy, rounds, split.test_indices, "Random")

    assert linucb_report.match_rate >= random_report.match_rate


def test_thompson_sampling_learns_better_than_random(rounds, split):
    thompson = ThompsonSamplingShared(n_features=rounds.n_features, random_state=0)
    replay_train(thompson, rounds, split.train_indices)
    thompson_report = evaluate_frozen(thompson, rounds, split.test_indices, "ThompsonSampling")

    random_policy = RandomPolicy(random_state=1)
    random_report = evaluate_frozen(random_policy, rounds, split.test_indices, "Random")

    assert thompson_report.match_rate >= random_report.match_rate


def test_popularity_greedy_picks_highest_popularity_feature(rounds):
    policy = PopularityGreedy()
    contexts = rounds.context[0]
    chosen = policy.select_arm(contexts)
    assert chosen == int(np.argmax(contexts[:, -2]))


def test_linucb_update_changes_posterior(rounds):
    policy = LinUCBShared(n_features=rounds.n_features, alpha=1.0)
    a_before = policy.A.copy()
    policy.update(rounds.context[0, 0], reward=1.0)
    assert not np.allclose(a_before, policy.A)


def test_thompson_sampling_update_increases_precision(rounds):
    policy = ThompsonSamplingShared(n_features=rounds.n_features)
    precision_before = policy.precision.copy()
    policy.update(rounds.context[0, 0], reward=1.0)
    assert np.trace(policy.precision) > np.trace(precision_before)
