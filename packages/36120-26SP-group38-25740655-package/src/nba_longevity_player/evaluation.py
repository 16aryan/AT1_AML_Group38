"""Evaluation built around AUPRC, the competition metric.

With a ~3.4% positive class, accuracy is meaningless and ROC-AUC is optimistic:
the negative pool is so much larger that a great many false positives move the
false-positive rate only slightly. Average precision responds directly to that
failure, so it leads every report here.

A warning attaches to every number this module produces. Training data covers
2008-2016 and the Kaggle test set covers 2020-2023. Random folds drawn inside
the training years measure same-era ranking, which is an easier problem than
the leaderboard poses. Use :func:`era_holdout_indices` when a result needs to
survive the gap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_STATE


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metric bundle for one set of predictions at one threshold."""

    auprc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    baseline_auprc: float
    lift_over_baseline: float
    threshold: float

    def as_series(self) -> pd.Series:
        return pd.Series(asdict(self)).round(4)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Score predicted probabilities against the truth.

    ``baseline_auprc`` is the positive rate, which is what a random ranker
    scores; ``lift_over_baseline`` expresses AUPRC as a multiple of it, because
    a raw AUPRC of 0.5 means very different things at a 3% base rate than at a
    40% one.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}."
        )
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true contains a single class; AUPRC is undefined.")

    y_pred = (y_prob >= threshold).astype(int)
    baseline = float(np.mean(y_true))
    auprc = float(average_precision_score(y_true, y_prob))
    return ClassificationMetrics(
        auprc=auprc,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        baseline_auprc=baseline,
        lift_over_baseline=auprc / baseline if baseline > 0 else float("nan"),
        threshold=threshold,
    )


def metrics_table(results: dict[str, ClassificationMetrics]) -> pd.DataFrame:
    """Stack several metric bundles into one comparison table, AUPRC first."""
    frame = pd.DataFrame({k: v.as_series() for k, v in results.items()})
    ordered = [
        "auprc", "lift_over_baseline", "roc_auc", "precision",
        "recall", "f1", "baseline_auprc", "threshold",
    ]
    return frame.reindex([m for m in ordered if m in frame.index])


def out_of_fold_predictions(
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Player-level out-of-fold scores.

    ``estimator_factory`` is called with no arguments per fold and must return
    an object exposing ``fit`` and ``predict_rank_score``. Because the table is
    already one row per player, ordinary stratified folds cannot leak a player
    across the split; grouping is unnecessary by construction.

    Returns:
        ``(oof_scores, per_fold_table)``.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = estimator_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        scores = model.predict_rank_score(X.iloc[val_idx])
        oof[val_idx] = scores
        rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_positives": int(y[val_idx].sum()),
            "auprc": average_precision_score(y[val_idx], scores),
        })
    if np.isnan(oof).any():
        raise RuntimeError("Some rows received no out-of-fold prediction.")
    return oof, pd.DataFrame(rows).round(4)


def summarise_cv(per_fold: pd.DataFrame) -> pd.Series:
    """Reduce per-fold AUPRC to mean, spread and range.

    The standard deviation is the part that matters when comparing models: a
    0.01 gap is not evidence of anything if the fold-to-fold spread is 0.03.
    """
    scores = per_fold["auprc"]
    return pd.Series({
        "mean_auprc": scores.mean(),
        "std_auprc": scores.std(ddof=1),
        "min_auprc": scores.min(),
        "max_auprc": scores.max(),
        "n_folds": len(scores),
    }).round(4)


def era_holdout_indices(
    season_df: pd.DataFrame,
    player_ids: np.ndarray,
    split_year: int = 2014,
) -> tuple[np.ndarray, np.ndarray]:
    """Split players by era instead of at random.

    Players whose final observed season falls before ``split_year`` go to the
    training side; the rest are held out. This mimics the structure of the real
    task, where the model is fitted on 2008-2016 and scored on 2020-2023.

    Random folds inside the training years scored a tuned model 0.045 above a
    weaker one, while the true leaderboard gap was 0.002. An improvement that
    only appears under random folds has not been shown to transfer.

    Returns:
        ``(train_positions, holdout_positions)`` into ``player_ids``.
    """
    last_season = season_df.groupby("pid")["year"].max()
    aligned = last_season.reindex(player_ids).to_numpy()
    holdout = aligned >= split_year
    return np.flatnonzero(~holdout), np.flatnonzero(holdout)


def threshold_sweep(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision, recall and F1 across candidate decision thresholds."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "predicted_positive": int(y_pred.sum()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows).round(4)


def best_threshold_by_f1(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
) -> float:
    """Threshold maximising F1, taken from the precision-recall curve itself."""
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true), np.asarray(y_prob)
    )
    denominator = precision[:-1] + recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            denominator > 0, 2 * precision[:-1] * recall[:-1] / denominator, 0.0
        )
    if len(f1_scores) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1_scores))])


def plot_precision_recall(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    label: str = "model",
    ax=None,
):
    """Draw a precision-recall curve against the random-ranker baseline."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"{label} (AUPRC = {auprc:.4f})")
    ax.axhline(baseline, linestyle="--", color="grey",
               label=f"random baseline = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    return ax


def adversarial_validation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_splits: int = 4,
    random_state: int = RANDOM_STATE,
) -> tuple[float, pd.DataFrame]:
    """Measure how separable the training and test feature distributions are.

    A classifier is trained to predict which set a row came from. An AUC near
    0.5 means the two are indistinguishable and a model fitted on one should
    transfer to the other. An AUC near 1.0 means they are trivially separable,
    so the model can partly learn *which period* a row belongs to instead of the
    outcome, and cross-validation inside the training set will read optimistic.

    Its feature importances rank the columns that most betray the period. Those
    with high importance and no plausible causal link to the outcome are pure
    nuisance variables and are candidates for removal.

    Returns:
        ``(auc, importance_table)``, the table sorted most era-revealing first.

    Raises:
        ImportError: if the optional ``xgboost`` dependency is missing.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("xgboost is required for adversarial_validation.") from exc

    shared = [c for c in X_train.columns if c in X_test.columns]
    combined = pd.concat([X_train[shared], X_test[shared]], ignore_index=True)
    is_test = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]

    def fresh():
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             tree_method="hist", enable_categorical=True,
                             n_jobs=-1, random_state=random_state)

    oof = np.zeros(len(is_test))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    for train_idx, val_idx in splitter.split(combined, is_test):
        model = fresh().fit(combined.iloc[train_idx], is_test[train_idx])
        oof[val_idx] = model.predict_proba(combined.iloc[val_idx])[:, 1]

    auc = float(roc_auc_score(is_test, oof))
    full = fresh().fit(combined, is_test)
    importance = (
        pd.DataFrame({"feature": combined.columns,
                      "era_marker_importance": full.feature_importances_})
        .sort_values("era_marker_importance", ascending=False)
        .reset_index(drop=True)
    )
    return auc, importance


def era_shifted_scores(
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    last_season: np.ndarray,
    cutoffs: tuple[int, ...] = (2012, 2013, 2014, 2015),
    min_positives: int = 20,
) -> pd.DataFrame:
    """Score a model by fitting on earlier cohorts and evaluating on later ones.

    For each cutoff, players whose final college season falls before it form the
    training set and the rest form the holdout. This reproduces the structure of
    the real task, where the model is fitted on 2008-2016 and scored on
    2020-2023, which random folds inside the training years cannot do.

    Because the cutoffs are a paired comparison against a rival configuration
    rather than independent samples, the per-cutoff record matters more than the
    mean: a variant that wins at every cutoff is more trustworthy than one with
    a better average that loses at some.

    Returns:
        One row per usable cutoff, with the holdout size and its AUPRC.
    """
    y = np.asarray(y)
    rows = []
    for cutoff in cutoffs:
        train_idx = np.flatnonzero(last_season < cutoff)
        holdout_idx = np.flatnonzero(last_season >= cutoff)
        if (y[train_idx].sum() < min_positives
                or y[holdout_idx].sum() < min_positives):
            continue
        model = estimator_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        scores = model.predict_rank_score(X.iloc[holdout_idx])
        rows.append({
            "cutoff": cutoff,
            "n_train": len(train_idx),
            "n_holdout": len(holdout_idx),
            "holdout_positives": int(y[holdout_idx].sum()),
            "auprc": average_precision_score(y[holdout_idx], scores),
        })
    if not rows:
        raise ValueError("No cutoff produced a usable train/holdout split.")
    return pd.DataFrame(rows).round(4)
