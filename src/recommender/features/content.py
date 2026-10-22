"""Content and point-in-time behavioural features for the bandit context.

Every running (point-in-time) quantity here is computed with a strict
``<`` cut-off: round ``t``'s feature only ever reads interactions at
positions strictly before ``t`` in the chronologically-sorted log, never
round ``t`` itself or anything after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender.data.loader import DeskdropLog


@dataclass(frozen=True)
class ArticleContent:
    """Dense TF-IDF content vectors, one row per article (catalogue order)."""

    vectorizer: TfidfVectorizer
    vectors: np.ndarray  # (n_articles, n_terms)


def fit_article_content(log: DeskdropLog, max_features: int = 20) -> ArticleContent:
    """Fit a small TF-IDF vectorizer over every cataloged article's text.

    This reads the article catalogue only (title + body text), never an
    interaction label, so it carries no leakage risk regardless of which
    rounds end up in training versus held-out partitions.

    Args:
        log: The full deskdrop log.
        max_features: Vocabulary size cap, kept small so the resulting
            context vector stays low-dimensional.

    Returns:
        The fitted vectorizer and the dense ``(n_articles, max_features)``
        content matrix, row-aligned to ``content_index``.
    """
    corpus = (log.articles["title"].fillna("") + " " + log.articles["text"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    vectors = vectorizer.fit_transform(corpus).toarray()
    return ArticleContent(vectorizer=vectorizer, vectors=vectors)


def running_article_popularity(log: DeskdropLog) -> np.ndarray:
    """Strictly-before interaction count per article, snapshotted at every round.

    Returns:
        ``(n_rounds, n_articles)`` array; entry ``[t, a]`` is how many times
        article ``a`` had been interacted with before round ``t``.
    """
    n_rounds, n_articles = log.n_rounds, log.n_articles
    one_hot = np.zeros((n_rounds, n_articles))
    one_hot[np.arange(n_rounds), log.interactions["content_index"].to_numpy()] = 1.0
    inclusive = np.cumsum(one_hot, axis=0)
    return inclusive - one_hot  # exclude the current round's own interaction


def running_user_profile(log: DeskdropLog, article_vectors: np.ndarray) -> np.ndarray:
    """Mean content vector of each user's strictly-prior interactions.

    Args:
        log: The full deskdrop log, interactions sorted chronologically.
        article_vectors: ``(n_articles, n_terms)`` content matrix.

    Returns:
        ``(n_rounds, n_terms)`` array; a user's first-ever round (and any
        round for a never-before-seen user) is the zero vector.
    """
    own_vectors = article_vectors[log.interactions["content_index"].to_numpy()]
    frame = pd.DataFrame(own_vectors, columns=[f"v{i}" for i in range(own_vectors.shape[1])])
    frame["person_index"] = log.interactions["person_index"].to_numpy()

    value_cols = [c for c in frame.columns if c != "person_index"]
    grouped = frame.groupby("person_index")[value_cols]
    inclusive_sum = grouped.cumsum()
    inclusive_count = grouped.cumcount() + 1

    exclusive_sum = (inclusive_sum - frame[value_cols]).to_numpy()
    exclusive_count = (inclusive_count - 1).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = exclusive_sum / exclusive_count[:, None]
    profile[exclusive_count == 0] = 0.0
    return profile
