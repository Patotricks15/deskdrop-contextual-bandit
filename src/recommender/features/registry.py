"""Registry mapping declared feature names to their transform implementations.

``assert_registry_matches_spec`` is the two-way check referenced by
``conf/features.yaml``: every declared feature must have a registered
transform, and every registered transform must be declared.
"""

from __future__ import annotations

from collections.abc import Callable

from recommender.features.spec import FeatureSpec
from recommender.features.transforms import (
    article_popularity_running,
    article_tfidf,
    user_profile_running_mean,
)

TRANSFORMS: dict[str, Callable] = {
    "article_tfidf": article_tfidf,
    "user_profile_running_mean": user_profile_running_mean,
    "article_popularity_running": article_popularity_running,
}


def assert_registry_matches_spec(spec: FeatureSpec) -> None:
    """Fail loudly if the spec and the registry disagree in either direction.

    Args:
        spec: The loaded declarative specification.

    Raises:
        ValueError: If a declared transform is unimplemented, or an
            implemented transform is undeclared.
    """
    declared = {f.transform for f in spec.features}
    implemented = set(TRANSFORMS)

    undeclared = implemented - declared
    unimplemented = declared - implemented
    if undeclared or unimplemented:
        raise ValueError(
            "Feature registry and conf/features.yaml disagree: "
            f"implemented-but-undeclared={sorted(undeclared)}, "
            f"declared-but-unimplemented={sorted(unimplemented)}."
        )
