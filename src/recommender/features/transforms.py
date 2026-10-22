"""Thin, registry-visible wrappers around the content/behavioural transforms.

The actual implementations live in :mod:`recommender.features.content` so
that :mod:`recommender.data.rounds` (which needs the same quantities to
build candidate pools) can reuse them without importing through the
declarative registry layer.
"""

from __future__ import annotations

import numpy as np

from recommender.data.loader import DeskdropLog
from recommender.features.content import (
    fit_article_content,
    running_article_popularity,
    running_user_profile,
)


def article_tfidf(log: DeskdropLog) -> np.ndarray:
    """Dense TF-IDF content matrix, one row per cataloged article."""
    return fit_article_content(log).vectors


def user_profile_running_mean(log: DeskdropLog, article_vectors: np.ndarray) -> np.ndarray:
    """Point-in-time mean content vector of each person's prior interactions."""
    return running_user_profile(log, article_vectors)


def article_popularity_running(log: DeskdropLog) -> np.ndarray:
    """Point-in-time interaction count for the article logged in each round."""
    popularity = running_article_popularity(log)
    content_index = log.interactions["content_index"].to_numpy()
    return popularity[np.arange(log.n_rounds), content_index]
