"""Model construction and the seed-bagged ensemble.

XGBoost is imported lazily so that splitting, evaluation and submission
assembly remain importable and testable where the optional ``xgboost`` extra
is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import BAGGING_SEEDS, BEST_PARAMS, RANDOM_STATE


def imbalance_ratio(y: pd.Series | np.ndarray) -> float:
    """Negatives divided by positives, the usual ``scale_pos_weight``.

    Raises:
        ValueError: if the target contains no positive cases.
    """
    y = np.asarray(y)
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("Target contains no positive cases.")
    return float((y == 0).sum() / positives)


def build_xgb_classifier(scale_pos_weight: float, seed: int = RANDOM_STATE, **overrides: Any):
    """Create an ``XGBClassifier`` on the tuned configuration.

    Notable defaults:

    - ``eval_metric='aucpr'`` matches the competition metric.
    - ``enable_categorical=True`` lets ``conf``, ``yr`` and ``role`` be used as
      genuine categoricals rather than as arbitrary integer codes.
    - No scaling is applied anywhere in this package. Tree splits depend only
      on the rank ordering of a feature, so a monotonic rescaling such as
      min-max leaves the fitted model unchanged.
    - Missing values are left in place, because absence is informative here: a
      statistic is often missing precisely because a player played too few
      minutes for it to be computed.

    Raises:
        ImportError: if the optional ``xgboost`` dependency is missing.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xgboost is required. Install with "
            "'pip install nba-longevity-player-25740655[xgboost]'."
        ) from exc

    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "enable_categorical": True,
        "n_jobs": -1,
        "random_state": seed,
        "scale_pos_weight": scale_pos_weight,
        **BEST_PARAMS,
    }
    params.update(overrides)
    return XGBClassifier(**params)


class SeedBaggedXGB:
    """Average the *ranks* of several XGBoost fits that differ only by seed.

    Bagging over seeds trims the variance a single fit carries when positives
    are rare (~600 of 17,938 players). Ranks are averaged rather than
    probabilities because average precision depends only on ordering, so the
    ensemble should combine what the metric actually reads; this also stops one
    over-confident member from dominating the mean.
    """

    def __init__(self, seeds: tuple[int, ...] = BAGGING_SEEDS, **overrides: Any):
        self.seeds = tuple(seeds)
        self.overrides = overrides
        self.models_: list[Any] = []
        self.scale_pos_weight_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "SeedBaggedXGB":
        """Fit one model per seed on the whole of ``X``."""
        y = np.asarray(y)
        self.scale_pos_weight_ = imbalance_ratio(y)
        self.models_ = []
        for seed in self.seeds:
            model = build_xgb_classifier(
                self.scale_pos_weight_, seed=seed, **self.overrides
            )
            model.fit(X, y)
            self.models_.append(model)
        return self

    def _check_fitted(self) -> None:
        if not self.models_:
            raise RuntimeError("SeedBaggedXGB must be fitted before predicting.")

    def predict_member_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Return an ``(n_samples, n_seeds)`` matrix of member probabilities."""
        self._check_fitted()
        return np.column_stack(
            [m.predict_proba(X)[:, 1] for m in self.models_]
        )

    def predict_rank_score(self, X: pd.DataFrame) -> np.ndarray:
        """Mean rank across members. Higher means more likely positive."""
        members = self.predict_member_probabilities(X)
        return np.mean(
            [rankdata(members[:, i]) for i in range(members.shape[1])], axis=0
        )

    def predict_proba_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Mean predicted probability across members."""
        return self.predict_member_probabilities(X).mean(axis=1)


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Descending gain-importance table for a fitted tree model.

    Gain importance is computed on training data and is biased towards
    high-cardinality continuous features, so it describes what the model used
    rather than what generalises.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise AttributeError("Model exposes no feature_importances_ attribute.")
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Model has {len(importances)} importances but "
            f"{len(feature_names)} feature names were supplied."
        )
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model(model, path: str | Path) -> Path:
    """Persist a fitted model with joblib, creating parent directories."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    """Load a model written by :func:`save_model`."""
    import joblib

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No saved model at {path}.")
    return joblib.load(path)


class RankScorer:
    """Adapt any scikit-learn classifier to the ranking interface used here.

    :func:`~nba_longevity_player.evaluation.out_of_fold_predictions` and
    :func:`~nba_longevity_player.evaluation.era_shifted_scores` both call
    ``predict_rank_score``. Wrapping an estimator in this class lets Logistic
    Regression, Random Forest and XGBoost be compared through exactly the same
    evaluation code, which is what makes the comparison across experiments fair.

    Categorical columns are ordinal-encoded because neither Logistic Regression
    nor Random Forest accepts pandas ``category`` dtype directly.
    """

    def __init__(self, estimator, categorical_features=None):
        self.estimator = estimator
        self.categorical_features = (
            tuple(categorical_features) if categorical_features is not None else None
        )

    @staticmethod
    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column in out.columns:
            if str(out[column].dtype) == "category":
                out[column] = out[column].cat.codes.astype(float)
        return out

    def fit(self, X: pd.DataFrame, y):
        from sklearn.base import clone

        self.model_ = clone(self.estimator)
        self.model_.fit(self._encode(X), np.asarray(y))
        return self

    def predict_proba_positive(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._encode(X))[:, 1]

    def predict_rank_score(self, X: pd.DataFrame) -> np.ndarray:
        return rankdata(self.predict_proba_positive(X))


def build_logistic_pipeline(class_weight="balanced", C: float = 0.1, **overrides):
    """Median-impute, standardise, then fit Logistic Regression.

    Scaling is required here and only here. A linear model's coefficients are
    expressed in the units of their features, so an unscaled matrix makes the
    penalty term punish large-unit features arbitrarily. The tree models in this
    package deliberately skip scaling, because tree splits depend only on rank
    ordering and a monotonic rescaling would leave them unchanged.

    ``class_weight='balanced'`` compensates for the ~3.4% positive rate.
    """
    from sklearn.compose import ColumnTransformer  # noqa: F401  (documented API)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = {"max_iter": 5000, "class_weight": class_weight, "C": C,
              "random_state": RANDOM_STATE}
    params.update(overrides)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(**params)),
    ])


def build_random_forest(class_weight="balanced_subsample", **overrides):
    """Median-impute, then fit a Random Forest.

    Imputation is required because scikit-learn's forest cannot consume missing
    values, unlike XGBoost which learns a default split direction. That is a
    real disadvantage on this dataset, where a statistic is often missing
    precisely because a player played too few minutes for it to be computed:
    imputing the median invents a figure the player never posted and erases the
    information carried by the absence itself.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    params = {
        "n_estimators": 600, "max_depth": None, "min_samples_leaf": 5,
        "max_features": "sqrt", "class_weight": class_weight,
        "n_jobs": -1, "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(**params)),
    ])


def build_lgbm_classifier(scale_pos_weight: float, seed: int = RANDOM_STATE, **overrides: Any):
    """Create a LightGBM classifier configured to match the XGBoost settings.

    LightGBM grows trees leaf-wise where XGBoost grows them level-wise, so the
    two make different errors on the same data even with comparable
    regularisation. That difference is the point: blending their ranks improved
    era-shifted AUPRC by 0.0060 and won at every cutoff tested, which is a
    larger and better-evidenced gain than any single-model change in this
    project.

    Raises:
        ImportError: if the optional ``lightgbm`` dependency is missing.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "lightgbm is required. Install with "
            "'pip install nba-longevity-player-25740655[lightgbm]'."
        ) from exc

    params: dict[str, Any] = {
        "n_estimators": 800, "learning_rate": 0.01, "num_leaves": 31,
        "min_child_samples": 20, "colsample_bytree": 0.3, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 10, "scale_pos_weight": scale_pos_weight,
        "random_state": seed, "n_jobs": -1, "verbose": -1,
    }
    params.update(overrides)
    return LGBMClassifier(**params)


class SeedBaggedLGBM(SeedBaggedXGB):
    """Seed-bagged LightGBM with the same interface as :class:`SeedBaggedXGB`.

    LightGBM does not accept pandas ``category`` dtype through this path, so
    categorical columns are ordinal-encoded to their category codes. Codes are
    aligned across train and test by :func:`~nba_longevity_player.features.align_categoricals`
    before fitting, so the same level maps to the same code on both sides.
    """

    @staticmethod
    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column in out.columns:
            if str(out[column].dtype) == "category":
                out[column] = out[column].cat.codes.astype(int)
        return out

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.scale_pos_weight_ = imbalance_ratio(y)
        encoded = self._encode(X)
        self.models_ = []
        for seed in self.seeds:
            model = build_lgbm_classifier(self.scale_pos_weight_, seed=seed,
                                          **self.overrides)
            model.fit(encoded, y)
            self.models_.append(model)
        return self

    def predict_member_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        encoded = self._encode(X)
        return np.column_stack([m.predict_proba(encoded)[:, 1] for m in self.models_])


def blend_rank_scores(*scored, weights=None) -> np.ndarray:
    """Combine several score vectors by averaging their normalised ranks.

    Average precision reads only ordering, so ranks are the natural quantity to
    average; probabilities from differently-conditioned models sit on different
    scales and are not directly comparable.

    Raises:
        ValueError: if the vectors differ in length or the weights do not match.
    """
    arrays = [np.asarray(s, dtype=float) for s in scored]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("All score vectors must have the same length.")
    if weights is None:
        weights = [1.0 / len(arrays)] * len(arrays)
    if len(weights) != len(arrays):
        raise ValueError(
            f"Got {len(arrays)} score vectors but {len(weights)} weights."
        )
    n = len(arrays[0])
    return sum(w * (rankdata(a) / n) for w, a in zip(weights, arrays))
