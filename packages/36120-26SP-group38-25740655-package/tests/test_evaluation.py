"""Tests for the AUPRC-centred evaluation helpers."""

import numpy as np
import pandas as pd
import pytest

from nba_longevity_player import (
    best_threshold_by_f1,
    era_holdout_indices,
    evaluate_predictions,
    metrics_table,
    out_of_fold_predictions,
    prepare_season_rows,
    summarise_cv,
    threshold_sweep,
)


@pytest.fixture
def perfect_ranking():
    y = np.array([0, 0, 0, 0, 1, 1])
    return y, np.array([0.01, 0.02, 0.03, 0.04, 0.98, 0.99])


class TestEvaluatePredictions:
    def test_perfect_ranking_scores_one(self, perfect_ranking):
        y, p = perfect_ranking
        assert evaluate_predictions(y, p).auprc == pytest.approx(1.0)

    def test_baseline_is_the_positive_rate(self, perfect_ranking):
        y, p = perfect_ranking
        assert evaluate_predictions(y, p).baseline_auprc == pytest.approx(2 / 6)

    def test_lift_expresses_auprc_against_that_baseline(self, perfect_ranking):
        y, p = perfect_ranking
        m = evaluate_predictions(y, p)
        assert m.lift_over_baseline == pytest.approx(m.auprc / m.baseline_auprc)

    def test_threshold_moves_the_point_metrics_only(self, perfect_ranking):
        y, p = perfect_ranking
        low, high = evaluate_predictions(y, p, 0.5), evaluate_predictions(y, p, 0.995)
        assert low.auprc == pytest.approx(high.auprc)  # ranking is unchanged
        assert high.recall < low.recall

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate_predictions(np.array([0, 1]), np.array([0.5]))

    def test_rejects_single_class_truth(self):
        with pytest.raises(ValueError, match="single class"):
            evaluate_predictions(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]))

    def test_as_series_is_rounded_for_display(self, perfect_ranking):
        y, p = perfect_ranking
        assert "auprc" in evaluate_predictions(y, p).as_series().index


class TestMetricsTable:
    def test_puts_auprc_first(self, perfect_ranking):
        y, p = perfect_ranking
        table = metrics_table({"a": evaluate_predictions(y, p)})
        assert table.index[0] == "auprc"
        assert list(table.columns) == ["a"]


class TestThresholdSweep:
    def test_covers_the_requested_thresholds(self, perfect_ranking):
        y, p = perfect_ranking
        sweep = threshold_sweep(y, p, thresholds=np.array([0.1, 0.5, 0.9]))
        assert len(sweep) == 3
        assert set(["threshold", "precision", "recall", "f1"]).issubset(sweep.columns)

    def test_recall_falls_as_the_threshold_rises(self, perfect_ranking):
        y, p = perfect_ranking
        sweep = threshold_sweep(y, p, thresholds=np.array([0.01, 0.5, 0.99]))
        assert sweep["recall"].is_monotonic_decreasing


class TestBestThresholdByF1:
    def test_separates_the_classes_on_a_clean_problem(self, perfect_ranking):
        y, p = perfect_ranking
        threshold = best_threshold_by_f1(y, p)
        assert (p >= threshold).astype(int).tolist() == y.tolist()


class TestSummariseCv:
    def test_reports_mean_and_spread(self):
        per_fold = pd.DataFrame({"auprc": [0.40, 0.50, 0.60]})
        summary = summarise_cv(per_fold)
        assert summary["mean_auprc"] == pytest.approx(0.50)
        assert summary["n_folds"] == 3
        assert summary["std_auprc"] > 0


class TestOutOfFoldPredictions:
    def test_every_row_receives_a_prediction(self):
        rng = np.random.default_rng(0)
        n = 200
        y = (rng.random(n) < 0.25).astype(int)
        X = pd.DataFrame({"signal": y + rng.normal(0, 0.5, n)})

        class Ranker:
            def fit(self, X, y):
                self.threshold = np.asarray(X["signal"]).mean()
                return self

            def predict_rank_score(self, X):
                return np.asarray(X["signal"])

        oof, per_fold = out_of_fold_predictions(Ranker, X, y, n_splits=4)
        assert not np.isnan(oof).any()
        assert len(per_fold) == 4
        assert per_fold["n_val"].sum() == n


class TestEraHoldoutIndices:
    def test_splits_players_by_their_final_season(self, season_frame):
        season = prepare_season_rows(season_frame)
        players = np.array([1, 2, 3, 4])
        train_idx, holdout_idx = era_holdout_indices(season, players, split_year=2014)
        # pid 1 ends 2012 and pid 2 ends 2011 -> train.
        # pid 3 ends 2014 and pid 4 ends 2016 -> holdout.
        assert sorted(players[train_idx]) == [1, 2]
        assert sorted(players[holdout_idx]) == [3, 4]

    def test_partitions_are_disjoint_and_complete(self, season_frame):
        season = prepare_season_rows(season_frame)
        players = np.array([1, 2, 3, 4])
        train_idx, holdout_idx = era_holdout_indices(season, players, split_year=2013)
        assert set(train_idx).isdisjoint(holdout_idx)
        assert len(train_idx) + len(holdout_idx) == len(players)
