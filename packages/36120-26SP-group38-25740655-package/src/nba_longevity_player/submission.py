"""Kaggle submission assembly and validation.

Because the model is already fitted at player level, there is no aggregation
step here: one input row is one submission row. That is the point of the
player-level design, and it removes the ``max``-over-seasons rule that the
earlier season-level pipeline needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import EXPECTED_SUBMISSION_ROWS, GROUP, TARGET


def ranks_to_unit_interval(scores: pd.Series | np.ndarray) -> np.ndarray:
    """Map any score vector onto ``(0, 1)`` preserving its exact ordering.

    Average precision reads only the ordering of the scores, so this transform
    leaves the metric unchanged while guaranteeing the submitted column sits in
    the ``[0, 1]`` range the competition expects. The output is a rank
    percentile, not a calibrated probability, and should not be read as one.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot rank an empty score vector.")
    return (rankdata(scores) - 0.5) / len(scores)


def build_submission(
    player_ids: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    as_ranks: bool = True,
    id_column: str = GROUP,
    prediction_column: str = TARGET,
) -> pd.DataFrame:
    """Assemble a submission frame from player-level scores.

    Args:
        player_ids: One id per player, in the order the scores were produced.
        scores: Model output for each player.
        as_ranks: When ``True`` the scores are converted to rank percentiles
            via :func:`ranks_to_unit_interval`. When ``False`` the scores are
            written through unchanged and must already lie in ``[0, 1]``.

    Raises:
        ValueError: on a length mismatch or duplicated player ids.
    """
    player_ids = np.asarray(player_ids)
    scores = np.asarray(scores, dtype=float)
    if len(player_ids) != len(scores):
        raise ValueError(
            f"Length mismatch: {len(player_ids)} ids vs {len(scores)} scores."
        )
    if pd.Series(player_ids).duplicated().any():
        raise ValueError(
            "Player ids must already be unique; the model predicts per player."
        )
    values = ranks_to_unit_interval(scores) if as_ranks else scores
    return pd.DataFrame({id_column: player_ids, prediction_column: values})


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame | None = None,
    expected_rows: int = EXPECTED_SUBMISSION_ROWS,
    id_column: str = GROUP,
    prediction_column: str = TARGET,
) -> None:
    """Check a submission for the faults that cause a rejected or zero upload.

    The id-set comparison against ``sample_submission`` is the check worth
    having: a file with the right shape but the wrong ids scores near zero and
    reports no error.

    Raises:
        ValueError: on the first problem found, describing it specifically.
    """
    for column in (id_column, prediction_column):
        if column not in submission.columns:
            raise ValueError(
                f"Submission is missing required column {column!r}. "
                f"Present: {list(submission.columns)}"
            )
    if len(submission) != expected_rows:
        raise ValueError(
            f"Submission has {len(submission)} rows; expected {expected_rows}."
        )
    if not submission[id_column].is_unique:
        duplicates = int(submission[id_column].duplicated().sum())
        raise ValueError(f"Submission contains {duplicates} duplicate id(s).")

    predictions = submission[prediction_column]
    if predictions.isna().any():
        raise ValueError(
            f"Submission has {int(predictions.isna().sum())} missing prediction(s)."
        )
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("Submission contains non-finite predictions.")
    if not predictions.between(0.0, 1.0).all():
        raise ValueError(
            f"Predictions must lie in [0, 1]; observed range "
            f"[{predictions.min():.4f}, {predictions.max():.4f}]."
        )
    if sample_submission is not None:
        expected = set(sample_submission[id_column])
        actual = set(submission[id_column])
        if expected != actual:
            raise ValueError(
                f"Submission ids do not match the sample submission: "
                f"{len(expected - actual)} expected id(s) missing, "
                f"{len(actual - expected)} unexpected id(s) present."
            )


def save_submission(
    submission: pd.DataFrame,
    path: str | Path,
    sample_submission: pd.DataFrame | None = None,
    expected_rows: int = EXPECTED_SUBMISSION_ROWS,
) -> Path:
    """Validate, then write a submission to CSV, creating parent directories.

    Validation runs first, so an invalid submission leaves no file behind to be
    uploaded by mistake.
    """
    validate_submission(
        submission, sample_submission=sample_submission, expected_rows=expected_rows
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path
