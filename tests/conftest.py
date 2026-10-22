"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_deskdrop_log, build_rounds, temporal_split  # noqa: E402
from recommender.data.loader import DeskdropLog  # noqa: E402
from recommender.features import build_feature_tables, load_feature_spec  # noqa: E402

TOPICS = {
    "ml": ["model", "training", "feature", "accuracy"],
    "cloud": ["lambda", "deploy", "scaling", "region"],
    "security": ["encryption", "firewall", "breach", "audit"],
}
EVENT_TYPES = ["VIEW", "LIKE", "BOOKMARK", "COMMENT"]


@pytest.fixture
def articles() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for content_id in range(1, 21):
        topic = rng.choice(list(TOPICS.keys()))
        words = rng.choice(TOPICS[topic], size=4, replace=True)
        rows.append(
            {
                "content_id": content_id,
                "title": f"{topic} notes {content_id}",
                "text": " ".join(words.tolist()),
                "lang": "en",
                "timestamp": content_id,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def interactions(articles: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    ts = 25
    for _ in range(200):
        rows.append(
            {
                "person_id": int(rng.integers(1, 21)),
                "content_id": int(rng.choice(articles["content_id"])),
                "event_type": rng.choice(EVENT_TYPES),
                "timestamp": ts,
            }
        )
        ts += int(rng.integers(1, 4))
    return pd.DataFrame(rows)


@pytest.fixture
def log(interactions: pd.DataFrame, articles: pd.DataFrame) -> DeskdropLog:
    return build_deskdrop_log(interactions, articles)


@pytest.fixture
def split(log: DeskdropLog):
    return temporal_split(log)


@pytest.fixture
def feature_tables(log: DeskdropLog):
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    return build_feature_tables(log, spec)


@pytest.fixture
def rounds(log: DeskdropLog, feature_tables):
    return build_rounds(log, feature_tables.article_vectors, pool_size=6, random_state=0)
