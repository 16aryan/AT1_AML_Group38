"""Player-level modelling of sustained NBA careers from college statistics.

The competition scores one prediction per player, so this package builds
features and fits models at player granularity. See :mod:`features` for why
that matters more than the choice of algorithm.

Typical use::

    import nba_longevity_player as nlp

    data = nlp.load_competition_data()
    train_season = nlp.prepare_season_rows(data.train)
    test_season = nlp.prepare_season_rows(data.test)
    numeric = nlp.numeric_feature_columns(train_season, test_season)

    player_train = nlp.build_player_table(train_season, numeric, include_target=True)
    player_test = nlp.build_player_table(test_season, numeric)
"""

from .config import (
    AGGREGATION_STATS,
    NUISANCE_FEATURES,
    BAGGING_SEEDS,
    BEST_PARAMS,
    CATEGORICAL_FEATURES,
    EXPECTED_SUBMISSION_ROWS,
    GROUP,
    ID_COLUMNS,
    RANDOM_STATE,
    RAW_TEXT_COLUMNS,
    TARGET,
)
from .data import (
    CompetitionData,
    DataNotFoundError,
    audit_player_structure,
    load_competition_data,
    resolve_data_dir,
)
from .evaluation import (
    ClassificationMetrics,
    adversarial_validation,
    era_shifted_scores,
    best_threshold_by_f1,
    era_holdout_indices,
    evaluate_predictions,
    metrics_table,
    out_of_fold_predictions,
    plot_precision_recall,
    summarise_cv,
    threshold_sweep,
)
from .features import (
    align_categoricals,
    drop_nuisance_features,
    within_season_zscore,
    assert_no_target_leakage,
    build_player_table,
    height_to_inches,
    numeric_feature_columns,
    prepare_season_rows,
)
from .modeling import (
    RankScorer,
    SeedBaggedLGBM,
    SeedBaggedXGB,
    blend_rank_scores,
    build_lgbm_classifier,
    build_logistic_pipeline,
    build_random_forest,
    build_xgb_classifier,
    feature_importance_table,
    imbalance_ratio,
    load_model,
    save_model,
)
from .submission import (
    build_submission,
    ranks_to_unit_interval,
    save_submission,
    validate_submission,
)

__version__ = "0.4.0"

__all__ = [
    "AGGREGATION_STATS", "NUISANCE_FEATURES", "adversarial_validation",
    "era_shifted_scores", "drop_nuisance_features", "within_season_zscore",
    "BAGGING_SEEDS", "BEST_PARAMS", "CATEGORICAL_FEATURES",
    "EXPECTED_SUBMISSION_ROWS", "GROUP", "ID_COLUMNS", "RANDOM_STATE",
    "RAW_TEXT_COLUMNS", "TARGET",
    "CompetitionData", "DataNotFoundError", "audit_player_structure",
    "load_competition_data", "resolve_data_dir",
    "ClassificationMetrics", "best_threshold_by_f1", "era_holdout_indices",
    "evaluate_predictions", "metrics_table", "out_of_fold_predictions",
    "plot_precision_recall", "summarise_cv", "threshold_sweep",
    "align_categoricals", "assert_no_target_leakage", "build_player_table",
    "height_to_inches", "numeric_feature_columns", "prepare_season_rows",
    "RankScorer", "SeedBaggedLGBM", "SeedBaggedXGB", "blend_rank_scores",
    "build_lgbm_classifier", "build_logistic_pipeline",
    "build_random_forest", "build_xgb_classifier", "feature_importance_table",
    "imbalance_ratio", "load_model", "save_model",
    "build_submission", "ranks_to_unit_interval", "save_submission",
    "validate_submission",
    "__version__",
]
