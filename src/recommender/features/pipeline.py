"""Build the content, user-profile, and popularity feature tables declaratively.

Validates that ``conf/features.yaml`` and the registry agree, then
materializes the three arrays that :mod:`recommender.data.rounds` combines
into each round's candidate context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.loader import DeskdropLog
from recommender.features.registry import TRANSFORMS, assert_registry_matches_spec
from recommender.features.spec import FeatureSpec


@dataclass(frozen=True)
class FeatureTables:
    """Materialized feature arrays."""

    article_vectors: np.ndarray  # (n_articles, n_terms)
    user_profile: np.ndarray  # (n_rounds, n_terms), point-in-time
    article_popularity: np.ndarray  # (n_rounds,), point-in-time


def build_feature_tables(log: DeskdropLog, spec: FeatureSpec) -> FeatureTables:
    """Materialize every declared feature.

    Args:
        log: The full deskdrop log.
        spec: The loaded declarative feature specification.

    Returns:
        The content and running behavioural feature tables.

    Raises:
        ValueError: If the registry and the spec disagree.
    """
    assert_registry_matches_spec(spec)

    article_vectors = TRANSFORMS["article_tfidf"](log)
    user_profile = TRANSFORMS["user_profile_running_mean"](log, article_vectors)
    article_popularity = TRANSFORMS["article_popularity_running"](log)

    return FeatureTables(
        article_vectors=article_vectors,
        user_profile=user_profile,
        article_popularity=article_popularity,
    )
