"""Build and persist the content and point-in-time feature tables.

    python scripts/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_deskdrop_log, load_articles, load_interactions  # noqa: E402
from recommender.features import build_feature_tables, load_feature_spec  # noqa: E402


def main() -> None:
    interactions = load_interactions(ROOT / "data" / "interactions.csv")
    articles = load_articles(ROOT / "data" / "articles.csv")
    log = build_deskdrop_log(interactions, articles)
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")

    tables = build_feature_tables(log, spec)

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "article_vectors.npy", tables.article_vectors)
    np.save(out_dir / "user_profile.npy", tables.user_profile)
    np.save(out_dir / "article_popularity.npy", tables.article_popularity)
    print(
        f"wrote content vectors for {log.n_articles} articles and point-in-time "
        f"behavioural features for {log.n_rounds} rounds to {out_dir}"
    )


if __name__ == "__main__":
    main()
