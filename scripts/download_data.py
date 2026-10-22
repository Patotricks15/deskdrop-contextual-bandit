"""Fetch the CI&T DeskDrop dataset, or generate a synthetic fixture.

Real data (requires ``KAGGLE_USERNAME``/``KAGGLE_KEY``):

    python scripts/download_data.py --kaggle

Synthetic fixture with the same schema, no Kaggle account needed:

    python scripts/download_data.py --synthetic
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KAGGLE_REF = "gspmoreira/articles-sharing-reading-from-cit-deskdrop"

# Five synthetic topics, each with its own vocabulary, so TF-IDF has real
# signal to separate articles by -- and so a person who mostly reads one
# topic has a genuinely informative running content profile.
TOPICS = {
    "data-engineering": ["pipeline", "etl", "warehouse", "airflow", "batch", "streaming", "spark"],
    "machine-learning": ["model", "training", "feature", "regression", "dataset", "accuracy"],
    "career": ["interview", "resume", "promotion", "mentor", "leadership", "salary"],
    "cloud": ["aws", "lambda", "kubernetes", "container", "deploy", "scaling", "region"],
    "security": ["vulnerability", "encryption", "firewall", "breach", "audit", "compliance"],
}
EVENT_TYPES = ["VIEW", "VIEW", "VIEW", "LIKE", "BOOKMARK", "COMMENT", "FOLLOW"]


def generate_synthetic(
    n_articles: int = 40, n_persons: int = 60, n_interactions: int = 700, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a topic-clustered article catalogue and a chronological log."""
    rng = np.random.default_rng(seed)
    topic_names = list(TOPICS.keys())

    articles = []
    for content_id in range(1, n_articles + 1):
        topic = rng.choice(topic_names)
        words = rng.choice(TOPICS[topic], size=6, replace=True)
        articles.append(
            {
                "content_id": content_id,
                "title": f"{topic.replace('-', ' ').title()} notes {content_id}",
                "text": " ".join(words.tolist()),
                "lang": "en",
                "timestamp": content_id,  # shared in catalogue order, well before any interaction
                "topic": topic,
            }
        )
    articles_df = pd.DataFrame(articles)

    # Each person has a Dirichlet preference over topics, so their
    # interaction history is genuinely topic-skewed.
    person_topic_pref = rng.dirichlet(np.ones(len(topic_names)) * 0.5, size=n_persons)

    rows = []
    timestamp = n_articles + 1
    for _ in range(n_interactions):
        person_id = rng.integers(1, n_persons + 1)
        preferred_topic = rng.choice(topic_names, p=person_topic_pref[person_id - 1])
        pool = articles_df[articles_df["topic"] == preferred_topic]["content_id"].to_numpy()
        content_id = int(rng.choice(pool))
        rows.append(
            {
                "person_id": int(person_id),
                "content_id": content_id,
                "event_type": rng.choice(EVENT_TYPES),
                "timestamp": timestamp,
            }
        )
        timestamp += rng.integers(1, 5)

    interactions_df = pd.DataFrame(rows)
    return interactions_df, articles_df.drop(columns=["topic"])


def download_kaggle() -> None:
    """Download the real dataset via the Kaggle CLI (requires credentials)."""
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    shared = DATA_DIR / "shared_articles.csv"
    interactions = DATA_DIR / "users_interactions.csv"
    if not shared.exists() or not interactions.exists():
        raise FileNotFoundError("Expected shared_articles.csv and users_interactions.csv.")

    articles_df = pd.read_csv(shared)
    articles_df = articles_df[articles_df["eventType"] == "CONTENT SHARED"].rename(
        columns={"contentId": "content_id", "publishTimestamp": "timestamp"}
    )
    articles_df["timestamp"] = articles_df.get("timestamp", articles_df.get("timestamp_x"))
    articles_df[["content_id", "title", "text", "lang", "timestamp"]].to_csv(
        DATA_DIR / "articles.csv", index=False
    )

    interactions_df = pd.read_csv(interactions).rename(
        columns={"contentId": "content_id", "personId": "person_id", "eventType": "event_type"}
    )
    interactions_df[["person_id", "content_id", "event_type", "timestamp"]].to_csv(
        DATA_DIR / "interactions.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if args.synthetic:
        interactions_df, articles_df = generate_synthetic()
        interactions_df.to_csv(DATA_DIR / "interactions.csv", index=False)
        articles_df.to_csv(DATA_DIR / "articles.csv", index=False)
        print(
            f"wrote {len(interactions_df)} interactions over {len(articles_df)} "
            f"articles to {DATA_DIR}"
        )
    else:
        download_kaggle()
        print(f"downloaded and normalized the real dataset into {DATA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
