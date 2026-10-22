"""Leakage-prevention tests for the point-in-time feature layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recommender.features import load_feature_spec
from recommender.features.content import running_article_popularity, running_user_profile
from recommender.features.registry import assert_registry_matches_spec

ROOT = Path(__file__).resolve().parents[1]


def test_user_profile_uses_only_strictly_prior_interactions(log, feature_tables):
    # The very first interaction for any person must have the zero profile:
    # there is nothing strictly before it to average.
    interactions = log.interactions.reset_index(drop=True)
    first_per_person = interactions.groupby("person_index").head(1).index
    for idx in first_per_person:
        np.testing.assert_allclose(feature_tables.user_profile[idx], 0.0)


def test_corrupting_a_later_round_does_not_change_an_earlier_profile(log, feature_tables):
    corrupted_vectors = feature_tables.article_vectors.copy()
    corrupted_vectors[log.interactions["content_index"].iloc[-1]] = 999.0
    corrupted_profile = running_user_profile(log, corrupted_vectors)
    # Every round except possibly the very last few sharing that article
    # must be unaffected by corrupting the last round's own article vector.
    np.testing.assert_allclose(feature_tables.user_profile[0], corrupted_profile[0])


def test_article_popularity_is_zero_before_an_articles_first_interaction(log):
    popularity = running_article_popularity(log)
    first_round_content = log.interactions["content_index"].iloc[0]
    assert popularity[0, first_round_content] == 0.0


def test_registry_matches_spec_declaration():
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    assert_registry_matches_spec(spec)  # must not raise
