"""Season-row cleaning and player-level feature construction.

The raw data is one row per player-*season*; the competition scores one
prediction per *player*. Two designs are therefore possible, and which wins
depends on the model.

Measured under identical folds, features and hyperparameters:

===========================================  =======  ===========
representation                               XGBoost  Logistic
===========================================  =======  ===========
player rows (``build_player_table``)          0.5024      0.4430
season rows, predictions averaged             0.4801      0.4717
season rows, predictions collapsed by max     0.4654      0.4430
===========================================  =======  ===========

Aggregation is a trade rather than a free improvement: it exchanges ~41,000
training rows for ~17,900 while tripling the feature count. A tree ensemble
exploits the richer description and gains; a linear model cannot and loses.

Independently of that, ``max`` is a poor rule for collapsing predictions after
the fact: a four-season player gets four draws at the maximum while a
one-season player gets one, so long college careers are inflated for a reason
unrelated to NBA prospects. Use the mean if collapsing is unavoidable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AGGREGATION_STATS,
    CATEGORICAL_FEATURES,
    GROUP,
    NUISANCE_FEATURES,
    RAW_TEXT_COLUMNS,
    TARGET,
)

_VALID_STATS = ("latest", "mean", "max", "min", "std")


def height_to_inches(value) -> float:
    """Convert a ``feet-inches`` string such as ``'6-7'`` to total inches.

    Returns ``nan`` for missing or unparseable values rather than raising, so
    that one malformed row cannot abort a whole load. Both the ASCII hyphen and
    the en dash that appears in the source data are accepted.
    """
    if pd.isna(value):
        return np.nan
    parts = str(value).strip().replace("–", "-").split("-")
    if len(parts) != 2:
        return np.nan
    try:
        return 12.0 * float(parts[0]) + float(parts[1])
    except ValueError:
        return np.nan


def prepare_season_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one season table: parse text columns, normalise missing values.

    Three columns the raw file stores as text carry real signal for NBA
    longevity and are unusable until parsed:

    - ``ht`` (``'6-7'``) becomes ``height_inches``
    - ``dob`` becomes ``age_at_season`` (season year minus birth year)
    - ``num`` becomes numeric ``jersey_number``

    A pipeline that selects numeric columns only, as the earlier season-level
    experiment did, silently discards all three along with every categorical.

    Infinities are converted to missing values: XGBoost handles missing values
    natively by learning a default split direction, but an infinity breaks the
    fit outright.
    """
    out = df.copy()
    birth_year = pd.to_datetime(out["dob"], errors="coerce").dt.year
    out["height_inches"] = out["ht"].map(height_to_inches)
    out["age_at_season"] = out["year"] - birth_year
    out["jersey_number"] = pd.to_numeric(out["num"], errors="coerce")
    out = out.drop(columns=list(RAW_TEXT_COLUMNS), errors="ignore")

    numeric = out.select_dtypes(include=np.number).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_FEATURES:
        if column in out.columns:
            out[column] = out[column].astype("object")
    return out


def numeric_feature_columns(
    train_season: pd.DataFrame,
    test_season: pd.DataFrame,
) -> list[str]:
    """Numeric columns usable as predictors, present in both train and test.

    Requiring presence in the test frame converts a failure that would
    otherwise surface only at submission time into an immediate one. ``pid``,
    ``label`` and ``year`` are excluded: the first two are identifiers or the
    target, and ``year`` encodes an era that does not recur in the test set.
    """
    return [
        column
        for column in test_season.select_dtypes(include=np.number).columns
        if column not in {GROUP, TARGET, "year"} and column in train_season.columns
    ]


def build_player_table(
    season_df: pd.DataFrame,
    numeric_columns: list[str],
    stats: tuple[str, ...] = AGGREGATION_STATS,
    include_target: bool = False,
) -> pd.DataFrame:
    """Collapse season rows into exactly one row per player.

    Row order follows first appearance in ``season_df``, which keeps the output
    aligned with the order the competition supplies test players in.

    Aggregations, all computed within a player:

    - ``<stat>_latest``: value in the player's most recent season, taken as the
      last row after a stable sort on ``year``. The final college season is the
      one closest to an NBA transition.
    - ``<stat>_mean`` / ``<stat>_max``: level and peak across the career.
    - ``career_observations``, ``seasons_observed``, ``teams_observed``,
      ``career_span_years``: experience and continuity.

    The target is aggregated with ``max``, so a player with any positive season
    counts as positive. The label describes a career outcome that belongs to
    the player rather than to one season, and a handful of players carry
    inconsistent season labels in the source data.

    Raises:
        ValueError: if ``stats`` contains an unsupported aggregation.
    """
    unknown = [s for s in stats if s not in _VALID_STATS]
    if unknown:
        raise ValueError(
            f"Unsupported aggregation(s) {unknown}; expected any of {_VALID_STATS}."
        )

    present_categoricals = [
        c for c in CATEGORICAL_FEATURES if c in season_df.columns
    ]
    base = pd.DataFrame(
        {GROUP: season_df[GROUP].drop_duplicates().reset_index(drop=True)}
    )
    grouped = season_df.groupby(GROUP, sort=False)
    latest_rows = (
        season_df.sort_values([GROUP, "year"], kind="stable")
        .groupby(GROUP, sort=False)
        .tail(1)
    )

    player = base.merge(
        latest_rows[[GROUP, *present_categoricals]], on=GROUP, how="left"
    )
    if "latest" in stats:
        player = player.merge(
            latest_rows[[GROUP, *numeric_columns]].rename(
                columns={c: f"{c}_latest" for c in numeric_columns}
            ),
            on=GROUP,
            how="left",
        )

    frame_stats = [s for s in stats if s != "latest"]
    if frame_stats:
        aggregated = grouped[numeric_columns].agg(frame_stats)
        aggregated.columns = [f"{c}_{s}" for c, s in aggregated.columns]
        player = player.merge(aggregated.reset_index(), on=GROUP, how="left")

    summary = grouped.agg(
        career_observations=(GROUP, "size"),
        seasons_observed=("year", "nunique"),
        teams_observed=("team", "nunique"),
        _first_season=("year", "min"),
        _last_season=("year", "max"),
    ).reset_index()
    summary["career_span_years"] = summary["_last_season"] - summary["_first_season"]
    # Absolute seasons are dropped: only the span survives, never the era.
    summary = summary.drop(columns=["_first_season", "_last_season"])
    player = player.merge(summary, on=GROUP, how="left")

    if include_target:
        player = player.merge(
            grouped[TARGET].max().rename(TARGET).reset_index(), on=GROUP, how="left"
        )
    return player


def align_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give both frames identical ``category`` dtypes over the union of levels.

    XGBoost's native categorical support encodes by category code. If the two
    frames were converted independently, a level absent from one would shift
    every code after it and the test matrix would be silently mislabelled.
    """
    train_out, test_out = X_train.copy(), X_test.copy()
    for column in categorical_features:
        if column not in train_out.columns or column not in test_out.columns:
            continue
        categories = pd.api.types.union_categoricals(
            [pd.Categorical(train_out[column]), pd.Categorical(test_out[column])]
        ).categories
        train_out[column] = pd.Categorical(train_out[column], categories=categories)
        test_out[column] = pd.Categorical(test_out[column], categories=categories)
    return train_out, test_out


def assert_no_target_leakage(features: list[str], test: pd.DataFrame) -> None:
    """Fail loudly if a chosen feature is unusable at prediction time.

    Raises:
        ValueError: if the target or player id is among the features, or if a
            feature is absent from the test frame.
    """
    for forbidden in (TARGET, GROUP):
        if forbidden in features:
            raise ValueError(f"{forbidden!r} must not be used as a feature.")
    missing = [f for f in features if f not in test.columns]
    if missing:
        raise ValueError(
            f"These features are absent from the test set: {missing}"
        )


def within_season_zscore(
    season_df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Express each statistic relative to the players of the same season.

    Every value becomes ``(x - mean) / std`` computed within its own ``year``
    cohort, so the feature answers "how far above his contemporaries" rather
    than "what raw figure did he post".

    This exists because the training data covers 2008-2016 while the Kaggle test
    set covers 2020-2023. Raw levels drift with the era -- three-point volume
    above all -- so a model fitted on raw values partly learns the period. An
    adversarial classifier separates the two eras at 0.94 AUC on raw features,
    which is close to trivially. Standardising within season removes the shared
    per-year shift while preserving the ranking inside each cohort, which is the
    part that should transfer.

    A column whose within-year standard deviation is zero yields missing values
    rather than infinities, which XGBoost handles natively.

    Note that this transform is scored *worse* than raw features under random
    cross-validation folds (0.4564 against 0.4814), and better under era-shifted
    validation (0.4362 against 0.4294, winning at every cutoff tested). Random
    folds drawn inside the training years cannot measure resistance to a shift
    they do not contain, so they are the wrong instrument for judging it.

    Args:
        season_df: Cleaned season rows, containing a ``year`` column.
        numeric_columns: Columns to standardise; others pass through untouched.

    Returns:
        A copy of ``season_df`` with those columns replaced by their z-scores.
    """
    missing = [c for c in numeric_columns if c not in season_df.columns]
    if missing:
        raise KeyError(f"Cannot standardise absent columns: {missing}")

    out = season_df.copy()
    grouped = out.groupby("year")[numeric_columns]
    means = grouped.transform("mean")
    deviations = grouped.transform("std").replace(0, np.nan)
    out[numeric_columns] = (out[numeric_columns] - means) / deviations
    return out


def drop_nuisance_features(
    *frames: pd.DataFrame,
    nuisance: tuple[str, ...] = NUISANCE_FEATURES,
) -> tuple[pd.DataFrame, ...]:
    """Remove nuisance columns and every aggregate derived from them.

    Matching covers both the bare name and any ``<name>_<stat>`` aggregate, so
    passing ``jersey_number`` also removes ``jersey_number_mean`` and friends.

    Returns:
        The frames in the order supplied, each with the columns removed.
    """
    out = []
    for frame in frames:
        drop = [
            column for column in frame.columns
            if any(column == n or column.startswith(f"{n}_") for n in nuisance)
        ]
        out.append(frame.drop(columns=drop))
    return tuple(out)
