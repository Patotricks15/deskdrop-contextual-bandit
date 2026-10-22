"""Offline/online parity: the served linear score must match the live
in-process ``context @ theta`` computation for the same context.
"""

from __future__ import annotations

import numpy as np

from recommender.evaluation import replay_train
from recommender.features.content import running_user_profile
from recommender.models import LinUCBShared
from recommender.serving.payload import recommend_from_payload, serialize_policy


def test_serialized_score_matches_live_theta_score(log, feature_tables, rounds, split):
    policy = LinUCBShared(n_features=rounds.n_features, alpha=1.0)
    replay_train(policy, rounds, split.train_indices)
    theta = np.linalg.inv(policy.A) @ policy.b

    # Pin every person's profile and every article's popularity snapshot to
    # one specific round's own point-in-time values, so the served context
    # is exactly the context the live policy actually scored -- isolating
    # a pure serialization round-trip check.
    round_idx = split.test_indices[0]
    position = 0
    content_index = int(rounds.candidate_content_index[round_idx, position])
    content_id = int(log.content_ids[content_index])

    raw_profile = running_user_profile(log, feature_tables.article_vectors)[round_idx]
    profile_table = np.tile(raw_profile, (log.person_ids.shape[0], 1))

    popularity_log1p = rounds.context[round_idx, position, -2]
    popularity_snapshot = np.zeros(log.content_ids.shape[0])
    popularity_snapshot[content_index] = np.expm1(popularity_log1p)

    payload = serialize_policy(
        theta, feature_tables, log.content_ids, log.person_ids, profile_table, popularity_snapshot
    )
    person_id = int(log.person_ids[0])

    served = recommend_from_payload(payload, person_id, [content_id], top_k=1)
    expected_score = float(rounds.context[round_idx, position] @ theta)

    assert abs(served[0]["score"] - expected_score) < 1e-6
