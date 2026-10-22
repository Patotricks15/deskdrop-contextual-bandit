"""Serialize a fitted shared-parameter bandit into a small JSON artifact.

Both ``LinUCBShared`` and ``ThompsonSamplingShared`` reduce, at serving
time, to a single linear scorer: their posterior mean ``theta`` over the
shared context features. Online, each draws its exploration differently
(a UCB bonus or a posterior sample); for a deployed scoring endpoint we
serve the posterior mean deterministically and leave further online
exploration to the next scheduled retraining cycle, which is a standard,
explicit simplification for this kind of batch-retrained bandit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from recommender.features.pipeline import FeatureTables


def serialize_policy(
    theta: np.ndarray,
    feature_tables: FeatureTables,
    content_ids: np.ndarray,
    person_ids: np.ndarray,
    final_user_profile: np.ndarray,
    article_popularity: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pack a policy's posterior mean and the feature tables needed to score it.

    Args:
        theta: The shared coefficient vector (posterior mean).
        feature_tables: Fitted content/popularity tables.
        content_ids: Original content ids, aligned to article index.
        person_ids: Original person ids, aligned to person index.
        final_user_profile: ``(n_persons, n_terms)`` last-known profile per
            person (the most recent point-in-time snapshot available).
        article_popularity: ``(n_articles,)`` raw (pre-log) interaction
            count snapshot, aligned to article index. Defaults to all
            zeros (treats every candidate as having no prior engagement).

    Returns:
        A dict safe to ``json.dumps`` directly.
    """
    if article_popularity is None:
        article_popularity = np.zeros(content_ids.shape[0])
    return {
        "theta": theta.tolist(),
        "content_ids": content_ids.tolist(),
        "article_vectors": feature_tables.article_vectors.tolist(),
        "person_ids": person_ids.tolist(),
        "final_user_profile": final_user_profile.tolist(),
        "article_popularity": article_popularity.tolist(),
    }


def recommend_from_payload(
    payload: dict[str, Any], person_id: int, candidate_content_ids: list[int], top_k: int = 1
) -> list[dict[str, Any]]:
    """Score a candidate list for one person using the serialized policy.

    Args:
        payload: Output of :func:`serialize_policy`, round-tripped through JSON.
        person_id: Original person id. Falls back to the zero profile
            (cold start) if unseen.
        candidate_content_ids: Original content ids to rank.
        top_k: Number of ranked results to return.

    Returns:
        ``[{"content_id": ..., "score": ...}, ...]``, best first.

    Raises:
        KeyError: If a candidate content id is not in the catalogue.
    """
    theta = np.array(payload["theta"])
    content_ids = payload["content_ids"]
    article_vectors = np.array(payload["article_vectors"])
    n_terms = article_vectors.shape[1]

    if person_id in payload["person_ids"]:
        person_pos = payload["person_ids"].index(person_id)
        profile = np.array(payload["final_user_profile"][person_pos])
    else:
        profile = np.zeros(n_terms)

    results = []
    for content_id in candidate_content_ids:
        if content_id not in content_ids:
            raise KeyError(f"content_id {content_id} is not in the catalogue.")
        article_index = content_ids.index(content_id)
        article_vector = article_vectors[article_index]
        interaction = profile * article_vector
        popularity_feature = np.log1p(payload["article_popularity"][article_index])
        context = np.concatenate([interaction, article_vector, [popularity_feature, 1.0]])
        results.append({"content_id": content_id, "score": float(context @ theta)})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
