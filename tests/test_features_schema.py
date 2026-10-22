"""Schema-contract tests for the feature spec and the raw data loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recommender.data.loader import build_deskdrop_log, load_articles, load_interactions
from recommender.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[1]


def test_feature_spec_declares_known_families_only():
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    assert spec.names() == {
        "article_tfidf",
        "user_profile_running_mean",
        "article_popularity_running",
    }


def test_load_interactions_rejects_missing_columns(tmp_path):
    bad = tmp_path / "interactions.csv"
    pd.DataFrame({"person_id": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_interactions(bad)


def test_load_articles_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_articles(tmp_path / "missing.csv")


def test_build_deskdrop_log_drops_interactions_without_a_cataloged_article():
    articles = pd.DataFrame(
        {
            "content_id": [1, 2],
            "title": ["a", "b"],
            "text": ["x", "y"],
            "lang": ["en", "en"],
            "timestamp": [1, 2],
        }
    )
    interactions = pd.DataFrame(
        {
            "person_id": [1, 1],
            "content_id": [1, 999],  # 999 is not in the catalogue
            "event_type": ["VIEW", "VIEW"],
            "timestamp": [10, 11],
        }
    )
    log = build_deskdrop_log(interactions, articles)
    assert log.n_rounds == 1
