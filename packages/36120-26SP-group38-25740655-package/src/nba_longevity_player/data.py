"""Loading and integrity checks for the competition CSVs.

The competition files are distributed through Kaggle and are not committed to
the repository, so the loader resolves their location at runtime rather than
hard-coding one absolute path. The same notebook therefore runs unchanged on
Colab and on a marker's laptop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import GROUP, TARGET

#: Searched in order by :func:`resolve_data_dir`.
DEFAULT_SEARCH_PATHS = (
    ".",
    "data",
    "data/raw",
    "../data/raw",
    "/content/drive/MyDrive/AML_AT1/data",
)


class DataNotFoundError(FileNotFoundError):
    """Raised when the competition CSV files cannot be located."""


def resolve_data_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the first directory that actually contains ``train.csv``.

    Search order: the ``explicit`` argument, then the ``NBA_DATA_DIR``
    environment variable, then :data:`DEFAULT_SEARCH_PATHS`.

    Raises:
        DataNotFoundError: if no candidate directory contains ``train.csv``.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("NBA_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend(Path(p) for p in DEFAULT_SEARCH_PATHS)

    for candidate in candidates:
        if (candidate / "train.csv").is_file():
            return candidate.resolve()

    searched = "\n  ".join(str(c) for c in candidates)
    raise DataNotFoundError(
        "Could not find train.csv. Searched:\n  " + searched + "\n"
        "Set NBA_DATA_DIR or pass an explicit path."
    )


@dataclass(frozen=True)
class CompetitionData:
    """The competition tables, loaded together so they stay consistent."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    metadata: pd.DataFrame | None = None

    def describe(self) -> pd.DataFrame:
        """One row per table, for notebook display."""
        rows = [
            ("train", *self.train.shape),
            ("test", *self.test.shape),
            ("sample_submission", *self.sample_submission.shape),
        ]
        if self.metadata is not None:
            rows.append(("metadata", *self.metadata.shape))
        return pd.DataFrame(rows, columns=["table", "rows", "columns"])


def load_competition_data(
    data_dir: str | os.PathLike[str] | None = None,
) -> CompetitionData:
    """Load train/test/sample_submission, and metadata when present.

    ``metadata.csv`` ships latin-1 encoded; reading it as UTF-8 raises.
    """
    directory = resolve_data_dir(data_dir)

    train = pd.read_csv(directory / "train.csv", low_memory=False)
    test = pd.read_csv(directory / "test.csv", low_memory=False)
    sample_submission = pd.read_csv(directory / "sample_submission.csv")

    metadata_path = directory / "metadata.csv"
    metadata = (
        pd.read_csv(metadata_path, encoding="latin1")
        if metadata_path.is_file()
        else None
    )
    return CompetitionData(train, test, sample_submission, metadata)


def audit_player_structure(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group: str = GROUP,
    target: str = TARGET,
) -> pd.DataFrame:
    """Quantify the player-season structure that motivates player aggregation.

    If ``players_with_multiple_rows`` is large, a row-level model is answering
    a different question from the one the leaderboard asks, because the
    leaderboard scores one prediction per player.
    """
    rows_per_player = train[group].value_counts()
    labels_per_player = train.groupby(group)[target].nunique()

    checks = {
        "train_rows": len(train),
        "train_unique_players": int(train[group].nunique()),
        "players_with_multiple_rows": int((rows_per_player > 1).sum()),
        "max_rows_for_one_player": int(rows_per_player.max()),
        "test_rows": len(test),
        "test_unique_players": int(test[group].nunique()),
        "train_year_range": f"{train['year'].min()}-{train['year'].max()}",
        "test_year_range": f"{test['year'].min()}-{test['year'].max()}",
        "players_with_inconsistent_labels": int((labels_per_player > 1).sum()),
        "row_positive_rate": round(float(train[target].mean()), 4),
    }
    return pd.DataFrame(checks.items(), columns=["check", "value"])
