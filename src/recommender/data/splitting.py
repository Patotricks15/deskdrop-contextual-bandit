"""Global temporal partitioning of the interaction stream.

Unlike ``crop-contextual-recommender`` and ``indonesia-tourism-diversity``,
DeskDrop interactions carry a real timestamp, so this project uses a
genuine temporal split: training rounds strictly precede validation rounds,
which strictly precede test rounds. The training partition doubles as the
online-learning replay stream, exactly as in ``crop-contextual-recommender``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.loader import DeskdropLog


@dataclass(frozen=True)
class Split:
    """Train (learning stream) / validation / test round indices, time-ordered."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.train_indices.size == 0:
            raise ValueError("The training partition is empty.")


def temporal_split(
    log: DeskdropLog,
    train_quantile: float = 0.70,
    validation_quantile: float = 0.85,
) -> Split:
    """Cut the chronologically-sorted interaction stream by timestamp quantiles.

    Args:
        log: The full interaction log, assumed sorted by timestamp.
        train_quantile: Quantile of the timestamp distribution ending training.
        validation_quantile: Quantile ending validation.

    Returns:
        A :class:`Split` over positional indices into ``log.interactions``.

    Raises:
        ValueError: If the quantiles are not ordered within ``(0, 1)``.
    """
    if not 0 < train_quantile < validation_quantile < 1:
        raise ValueError(
            "Expected 0 < train_quantile < validation_quantile < 1, got "
            f"{train_quantile} and {validation_quantile}."
        )

    timestamps = log.interactions["timestamp"].to_numpy()
    train_cutoff = np.quantile(timestamps, train_quantile)
    val_cutoff = np.quantile(timestamps, validation_quantile)

    # Strict `<` on the training side: an event exactly at the cut-off
    # belongs to the future, not to the history used to predict it.
    train_indices = np.flatnonzero(timestamps < train_cutoff)
    validation_indices = np.flatnonzero((timestamps >= train_cutoff) & (timestamps < val_cutoff))
    test_indices = np.flatnonzero(timestamps >= val_cutoff)

    return Split(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )
