"""Shared configuration for the player-level NBA longevity model.

The constants here encode the two decisions that drove the score. First, the
model is built at *player* granularity rather than player-season granularity,
because that is the granularity the competition scores. Second, ``year`` never
becomes a feature: training data covers 2008-2016 while the Kaggle test set
covers 2020-2023, so a split on absolute season would key on an era that does
not recur in the test set.
"""

from __future__ import annotations

#: Binary target. ``1`` marks a player who went on to play at least three NBA
#: seasons; ``0`` marks a player who did not.
TARGET = "label"

#: Column identifying a player. The raw data holds one row per player-season,
#: so this is the key every aggregation groups on.
GROUP = "pid"

#: Columns that must never reach the model as predictors.
ID_COLUMNS = (TARGET, GROUP, "year")

#: Raw columns consumed by :func:`~nba_longevity_player.features.prepare_season_rows`
#: and dropped afterwards, having been converted into numeric equivalents.
RAW_TEXT_COLUMNS = ("ht", "dob", "num", "type")

#: Categorical predictors kept at player level, taken from the latest season.
#: ``team`` is deliberately excluded: with 353 levels it measurably hurt
#: cross-validated average precision (0.4545 against 0.4814 without it).
CATEGORICAL_FEATURES = ("conf", "yr", "role")

#: Per-statistic aggregations applied across a player's seasons. Chosen by
#: ablation: this trio scored 0.4853 against 0.4803 for the fuller
#: latest+mean+max+min+std set, which carries 297 features instead of 181.
#: With only ~600 positive players, the extra columns cost more than they pay.
AGGREGATION_STATS = ("latest", "mean", "max")

#: Features carrying no plausible causal link to NBA longevity that nonetheless
#: separate the 2008-2016 and 2020-2023 eras strongly. Adversarial validation
#: ranked ``jersey_number`` the single strongest era marker in the whole feature
#: set, which makes it a pure nuisance variable: it helps a model recognise
#: *when* a player played, never whether he lasted.
NUISANCE_FEATURES = ("jersey_number",)

#: Seed used for every deterministic operation.
RANDOM_STATE = 42

#: Seeds bagged in the final ensemble. Averaging ranks across several seeds
#: removes some of the variance a single XGBoost fit carries on a rare target.
BAGGING_SEEDS = (42, 7, 2024, 101, 555)

#: XGBoost hyperparameters selected by randomised search over 25 candidates,
#: scored by pooled out-of-fold average precision across three fold seeds.
BEST_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "min_child_weight": 20,
    "reg_lambda": 10,
    "reg_alpha": 0,
    "gamma": 0.5,
}

#: Rows the competition expects in a submission (one per test player).
EXPECTED_SUBMISSION_ROWS = 6261
