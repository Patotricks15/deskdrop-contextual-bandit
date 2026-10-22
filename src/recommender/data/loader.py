"""Load the CI&T DeskDrop interaction log and article catalogue.

``users_interactions.csv`` is a chronological log of engagement events
(view, like, comment, follow, bookmark) between a person and a shared
article. ``shared_articles.csv`` is the article catalogue (title, text,
language, share timestamp). Framed as a contextual bandit: each interaction
event is one round where a person engaged with one particular article out
of whatever else could plausibly have been shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_INTERACTION_COLUMNS = {"person_id", "content_id", "event_type", "timestamp"}
REQUIRED_ARTICLE_COLUMNS = {"content_id", "title", "text", "lang", "timestamp"}

#: Relative engagement weight per event type, used only as a descriptive
#: popularity weight -- never as the bandit reward itself, which is always
#: 1 for the logged (person, article) pair (see ``recommender.data.rounds``).
EVENT_WEIGHTS = {
    "VIEW": 1.0,
    "BOOKMARK": 2.0,
    "LIKE": 2.0,
    "COMMENT": 3.0,
    "FOLLOW": 3.0,
}


@dataclass(frozen=True)
class DeskdropLog:
    """Interaction log and article catalogue with contiguous indices.

    Attributes:
        interactions: Columns ``person_index``, ``content_index``,
            ``event_type``, ``timestamp``, ``weight``, sorted by timestamp.
        articles: Columns ``content_index``, ``title``, ``text``, ``lang``,
            ``timestamp`` (share time), indexed by contiguous content index.
        person_ids: Original person ids, positionally aligned to person index.
        content_ids: Original content ids, positionally aligned to content index.
    """

    interactions: pd.DataFrame
    articles: pd.DataFrame
    person_ids: np.ndarray
    content_ids: np.ndarray

    @property
    def n_rounds(self) -> int:
        return int(self.interactions.shape[0])

    @property
    def n_articles(self) -> int:
        return int(self.content_ids.shape[0])


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Read ``users_interactions.csv`` and validate its contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py --synthetic"
        )
    frame = pd.read_csv(path)
    missing = REQUIRED_INTERACTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    return frame


def load_articles(path: str | Path) -> pd.DataFrame:
    """Read ``shared_articles.csv`` and validate its contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    frame = pd.read_csv(path)
    missing = REQUIRED_ARTICLE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def build_deskdrop_log(interactions: pd.DataFrame, articles: pd.DataFrame) -> DeskdropLog:
    """Reindex persons/articles to contiguous indices and sort chronologically.

    Only interactions whose article is present in the catalogue are kept:
    an interaction with no matching article has no content features to
    build a context from.

    Args:
        interactions: Raw frame from :func:`load_interactions`.
        articles: Raw frame from :func:`load_articles`.

    Returns:
        A :class:`DeskdropLog`.

    Raises:
        ValueError: If no interaction matches a cataloged article.
    """
    known_content = set(articles["content_id"].unique())
    frame = interactions[interactions["content_id"].isin(known_content)].copy()
    if frame.empty:
        raise ValueError("No interaction matches a cataloged article.")

    person_ids = np.sort(frame["person_id"].unique())
    content_ids = np.sort(articles["content_id"].unique())
    person_map = {pid: idx for idx, pid in enumerate(person_ids)}
    content_map = {cid: idx for idx, cid in enumerate(content_ids)}

    frame["person_index"] = frame["person_id"].map(person_map).astype(np.int64)
    frame["content_index"] = frame["content_id"].map(content_map).astype(np.int64)
    frame["weight"] = frame["event_type"].map(EVENT_WEIGHTS).fillna(1.0)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    article_frame = articles.copy()
    article_frame["content_index"] = article_frame["content_id"].map(content_map)
    article_frame = article_frame.dropna(subset=["content_index"])
    article_frame["content_index"] = article_frame["content_index"].astype(np.int64)
    article_frame = article_frame.sort_values("content_index").reset_index(drop=True)

    return DeskdropLog(
        interactions=frame[
            ["person_index", "content_index", "event_type", "timestamp", "weight"]
        ],
        articles=article_frame[["content_index", "title", "text", "lang", "timestamp"]],
        person_ids=person_ids,
        content_ids=content_ids,
    )
