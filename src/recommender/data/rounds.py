"""Build bandit rounds: candidate pools, a synthetic logging policy, and context.

DeskDrop logs only the article a person actually engaged with, not what
else was shown. To evaluate a bandit policy -- online or via importance
sampling -- we need a plausible logging policy with *known* propensities.
This module assumes the historical system surfaced articles roughly
proportional to how popular they already were (a common, simple real-world
default), samples a candidate pool of that shape around each logged
article, and assigns the logged article its exact probability under that
explicit softmax policy. The resulting propensity is used both to simulate
the logging choice and, downstream, as the denominator of the IPS
estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.loader import DeskdropLog
from recommender.features.content import running_article_popularity, running_user_profile


@dataclass(frozen=True)
class RoundsData:
    """Per-round candidate pools, context vectors, and logging-policy data.

    Attributes:
        context: ``(n_rounds, pool_size, n_features)`` context vector per
            candidate in every round's pool.
        candidate_content_index: ``(n_rounds, pool_size)`` original content
            index of each pool member.
        logged_position: ``(n_rounds,)`` position within the pool of the
            article actually interacted with.
        propensity: ``(n_rounds,)`` logging policy's probability of having
            shown the logged article, given its pool.
        n_features: Context dimensionality.
    """

    context: np.ndarray
    candidate_content_index: np.ndarray
    logged_position: np.ndarray
    propensity: np.ndarray
    n_features: int


def build_rounds(
    log: DeskdropLog,
    article_vectors: np.ndarray,
    pool_size: int = 8,
    temperature: float = 1.0,
    popularity_smoothing: float = 1.0,
    random_state: int = 0,
) -> RoundsData:
    """Construct every round's candidate pool, context, and propensity.

    Args:
        log: The full deskdrop log.
        article_vectors: ``(n_articles, n_terms)`` TF-IDF content matrix.
        pool_size: Candidates offered per round, including the logged one.
        temperature: Softmax temperature for the logging policy; higher
            values flatten it towards uniform (more exploration in the
            simulated historical log).
        popularity_smoothing: Additive smoothing so a brand-new article
            with zero prior interactions still has non-zero probability.
        random_state: Seed controlling distractor sampling and pool order.

    Returns:
        A :class:`RoundsData`.

    Raises:
        ValueError: If ``pool_size`` exceeds the number of articles shared
            by the time of the earliest usable round.
    """
    rng = np.random.default_rng(random_state)
    n_rounds = log.n_rounds
    n_terms = article_vectors.shape[1]
    n_features = 2 * n_terms + 2  # profile-article interaction + article vector + popularity + bias

    popularity = running_article_popularity(log)
    user_profile = running_user_profile(log, article_vectors)

    article_timestamps = (
        log.articles.set_index("content_index")["timestamp"].sort_index().to_numpy()
    )
    interaction_timestamps = log.interactions["timestamp"].to_numpy()
    logged_content = log.interactions["content_index"].to_numpy()

    context = np.zeros((n_rounds, pool_size, n_features))
    candidate_content_index = np.zeros((n_rounds, pool_size), dtype=np.int64)
    logged_position = np.zeros(n_rounds, dtype=np.int64)
    propensity = np.zeros(n_rounds)

    for t in range(n_rounds):
        available = np.flatnonzero(article_timestamps <= interaction_timestamps[t])
        if available.size < pool_size:
            raise ValueError(
                f"Only {available.size} articles are available by round {t}, "
                f"fewer than pool_size={pool_size}."
            )
        logged = logged_content[t]
        distractor_candidates = available[available != logged]
        weights = popularity[t, distractor_candidates] + popularity_smoothing
        weights = weights / weights.sum()
        distractors = rng.choice(
            distractor_candidates, size=pool_size - 1, replace=False, p=weights
        )

        pool = np.concatenate([[logged], distractors])
        rng.shuffle(pool)
        position = int(np.flatnonzero(pool == logged)[0])

        scores = np.log(popularity[t, pool] + popularity_smoothing) / temperature
        scores -= scores.max()  # numerical stability
        exp_scores = np.exp(scores)
        pool_propensities = exp_scores / exp_scores.sum()

        candidate_content_index[t] = pool
        logged_position[t] = position
        propensity[t] = pool_propensities[position]

        pop_feature = np.log1p(popularity[t, pool])
        bias = np.ones(pool_size)
        # Elementwise product of the user's running content profile with each
        # candidate's own content vector: this is what lets a *linear* shared
        # model learn "article matches this user's topic", which a plain
        # concatenation of the two raw vectors cannot represent (a linear
        # function of concatenated features has no cross term between them).
        interaction = user_profile[t][None, :] * article_vectors[pool]
        context[t] = np.concatenate(
            [
                interaction,
                article_vectors[pool],
                pop_feature[:, None],
                bias[:, None],
            ],
            axis=1,
        )

    return RoundsData(
        context=context,
        candidate_content_index=candidate_content_index,
        logged_position=logged_position,
        propensity=propensity,
        n_features=n_features,
    )
