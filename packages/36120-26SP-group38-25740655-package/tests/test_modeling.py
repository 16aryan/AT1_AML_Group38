"""Tests for model construction and the seed-bagged ensemble."""

import numpy as np
import pandas as pd
import pytest

from nba_longevity_player import (
    BEST_PARAMS,
    SeedBaggedXGB,
    build_xgb_classifier,
    feature_importance_table,
    imbalance_ratio,
    load_model,
    save_model,
)

xgboost = pytest.importorskip("xgboost", reason="optional xgboost extra")


@pytest.fixture
def imbalanced_dataset():
    """400 rows at a ~5% positive rate with one genuinely predictive column."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.05).astype(int)
    return pd.DataFrame({
        "signal": y * 2.0 + rng.normal(0, 1.0, n),
        "noise": rng.normal(0, 1, n),
    }), y


class TestImbalanceRatio:
    def test_counts_negatives_per_positive(self):
        assert imbalance_ratio(np.array([0, 0, 0, 1])) == pytest.approx(3.0)

    def test_accepts_a_series(self):
        assert imbalance_ratio(pd.Series([0, 0, 1, 1])) == pytest.approx(1.0)

    def test_raises_when_no_positives(self):
        with pytest.raises(ValueError, match="no positive cases"):
            imbalance_ratio(np.zeros(5, dtype=int))


class TestBuildXgbClassifier:
    def test_applies_the_tuned_configuration(self):
        model = build_xgb_classifier(scale_pos_weight=10.0)
        params = model.get_params()
        for key, value in BEST_PARAMS.items():
            assert params[key] == value

    def test_optimises_the_competition_metric(self):
        assert build_xgb_classifier(1.0).get_params()["eval_metric"] == "aucpr"

    def test_enables_native_categorical_support(self):
        assert build_xgb_classifier(1.0).get_params()["enable_categorical"] is True

    def test_overrides_win_over_defaults(self):
        model = build_xgb_classifier(1.0, max_depth=2)
        assert model.get_params()["max_depth"] == 2


class TestSeedBaggedXGB:
    def test_fits_one_model_per_seed(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = SeedBaggedXGB(seeds=(1, 2, 3), n_estimators=20).fit(X, y)
        assert len(model.models_) == 3

    def test_derives_scale_pos_weight_from_the_training_target(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = SeedBaggedXGB(seeds=(1,), n_estimators=20).fit(X, y)
        assert model.scale_pos_weight_ == pytest.approx(imbalance_ratio(y))

    def test_predicting_before_fitting_raises(self, imbalanced_dataset):
        X, _ = imbalanced_dataset
        with pytest.raises(RuntimeError, match="must be fitted"):
            SeedBaggedXGB().predict_rank_score(X)

    def test_member_matrix_has_one_column_per_seed(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = SeedBaggedXGB(seeds=(1, 2), n_estimators=20).fit(X, y)
        assert model.predict_member_probabilities(X).shape == (len(X), 2)

    def test_rank_scores_span_the_sample(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = SeedBaggedXGB(seeds=(1, 2), n_estimators=20).fit(X, y)
        scores = model.predict_rank_score(X)
        assert len(scores) == len(X)
        assert scores.min() >= 1.0 and scores.max() <= len(X)

    def test_mean_probabilities_are_probabilities(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = SeedBaggedXGB(seeds=(1, 2), n_estimators=20).fit(X, y)
        probabilities = model.predict_proba_mean(X)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    def test_learns_the_signal_column(self, imbalanced_dataset):
        from sklearn.metrics import average_precision_score

        X, y = imbalanced_dataset
        model = SeedBaggedXGB(seeds=(1, 2), n_estimators=60).fit(X, y)
        scores = model.predict_rank_score(X)
        # Well above the ~0.05 base rate a random ranker would achieve.
        assert average_precision_score(y, scores) > 0.30

    def test_is_reproducible_for_a_fixed_seed_set(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        a = SeedBaggedXGB(seeds=(1, 2), n_estimators=20).fit(X, y).predict_rank_score(X)
        b = SeedBaggedXGB(seeds=(1, 2), n_estimators=20).fit(X, y).predict_rank_score(X)
        np.testing.assert_array_equal(a, b)

    def test_handles_categorical_columns(self):
        rng = np.random.default_rng(1)
        n = 200
        y = (rng.random(n) < 0.2).astype(int)
        X = pd.DataFrame({
            "num": rng.normal(0, 1, n),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        })
        model = SeedBaggedXGB(seeds=(1,), n_estimators=20).fit(X, y)
        assert len(model.predict_rank_score(X)) == n


class TestFeatureImportanceTable:
    def test_sorted_descending_by_importance(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = build_xgb_classifier(imbalance_ratio(y), n_estimators=30).fit(X, y)
        table = feature_importance_table(model, list(X.columns))
        assert list(table.columns) == ["feature", "importance"]
        assert table["importance"].is_monotonic_decreasing

    def test_rejects_mismatched_name_count(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = build_xgb_classifier(imbalance_ratio(y), n_estimators=10).fit(X, y)
        with pytest.raises(ValueError, match="feature names"):
            feature_importance_table(model, ["only_one"])

    def test_rejects_a_model_without_importances(self):
        with pytest.raises(AttributeError):
            feature_importance_table(object(), ["a"])


class TestModelPersistence:
    def test_round_trips_through_disk(self, tmp_path, imbalanced_dataset):
        X, y = imbalanced_dataset
        model = SeedBaggedXGB(seeds=(1,), n_estimators=20).fit(X, y)
        path = save_model(model, tmp_path / "nested" / "model.joblib")
        assert path.is_file()
        np.testing.assert_allclose(
            load_model(path).predict_rank_score(X), model.predict_rank_score(X)
        )

    def test_loading_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "absent.joblib")


class TestRankScorer:
    def test_wraps_any_sklearn_estimator(self, imbalanced_dataset):
        from nba_longevity_player import RankScorer, build_logistic_pipeline
        X, y = imbalanced_dataset
        model = RankScorer(build_logistic_pipeline()).fit(X, y)
        scores = model.predict_rank_score(X)
        assert len(scores) == len(X)
        assert scores.min() >= 1.0 and scores.max() <= len(X)

    def test_encodes_category_columns_for_estimators_that_reject_them(self):
        from nba_longevity_player import RankScorer, build_random_forest
        rng = np.random.default_rng(3)
        n = 150
        y = (rng.random(n) < 0.25).astype(int)
        X = pd.DataFrame({
            "num": rng.normal(0, 1, n),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        })
        # A bare RandomForest would raise on the category dtype.
        model = RankScorer(build_random_forest(n_estimators=20)).fit(X, y)
        assert len(model.predict_rank_score(X)) == n

    def test_probabilities_stay_in_range(self, imbalanced_dataset):
        from nba_longevity_player import RankScorer, build_logistic_pipeline
        X, y = imbalanced_dataset
        p = RankScorer(build_logistic_pipeline()).fit(X, y).predict_proba_positive(X)
        assert ((p >= 0) & (p <= 1)).all()

    def test_ranks_agree_with_probability_ordering(self, imbalanced_dataset):
        from nba_longevity_player import RankScorer, build_logistic_pipeline
        X, y = imbalanced_dataset
        m = RankScorer(build_logistic_pipeline()).fit(X, y)
        p, r = m.predict_proba_positive(X), m.predict_rank_score(X)
        assert list(np.argsort(p)) == list(np.argsort(r))


class TestLinearAndForestBuilders:
    def test_logistic_pipeline_scales_and_imputes(self):
        from nba_longevity_player import build_logistic_pipeline
        steps = dict(build_logistic_pipeline().named_steps)
        assert "impute" in steps and "scale" in steps and "model" in steps

    def test_logistic_handles_missing_values_via_imputation(self):
        from nba_longevity_player import build_logistic_pipeline
        rng = np.random.default_rng(4)
        X = pd.DataFrame({"a": rng.normal(0, 1, 80), "b": rng.normal(0, 1, 80)})
        X.loc[:10, "a"] = np.nan
        y = (rng.random(80) < 0.3).astype(int)
        build_logistic_pipeline().fit(X, y)  # must not raise

    def test_random_forest_defaults_address_imbalance(self):
        from nba_longevity_player import build_random_forest
        rf = build_random_forest().named_steps["model"]
        assert rf.class_weight == "balanced_subsample"

    def test_builder_overrides_are_applied(self):
        from nba_longevity_player import build_logistic_pipeline, build_random_forest
        assert build_logistic_pipeline(C=5.0).named_steps["model"].C == 5.0
        assert build_random_forest(n_estimators=7).named_steps["model"].n_estimators == 7


class TestSeedBaggedLGBM:
    def test_fits_and_ranks(self, imbalanced_dataset):
        pytest.importorskip("lightgbm")
        from nba_longevity_player import SeedBaggedLGBM
        X, y = imbalanced_dataset
        m = SeedBaggedLGBM(seeds=(1, 2), n_estimators=30).fit(X, y)
        assert len(m.models_) == 2
        assert len(m.predict_rank_score(X)) == len(X)

    def test_encodes_categoricals_lightgbm_rejects(self):
        pytest.importorskip("lightgbm")
        from nba_longevity_player import SeedBaggedLGBM
        rng = np.random.default_rng(5)
        n = 200
        y = (rng.random(n) < 0.25).astype(int)
        X = pd.DataFrame({"num": rng.normal(0, 1, n),
                          "cat": pd.Categorical(rng.choice(list("abc"), n))})
        m = SeedBaggedLGBM(seeds=(1,), n_estimators=30).fit(X, y)
        assert len(m.predict_rank_score(X)) == n

    def test_learns_the_signal(self, imbalanced_dataset):
        pytest.importorskip("lightgbm")
        from sklearn.metrics import average_precision_score
        from nba_longevity_player import SeedBaggedLGBM
        X, y = imbalanced_dataset
        m = SeedBaggedLGBM(seeds=(1, 2), n_estimators=120).fit(X, y)
        assert average_precision_score(y, m.predict_rank_score(X)) > 0.30


class TestBlendRankScores:
    def test_equal_weights_average_the_ranks(self):
        from nba_longevity_player import blend_rank_scores
        a = np.array([1.0, 2.0, 3.0, 4.0])
        out = blend_rank_scores(a, a)
        np.testing.assert_allclose(out, np.array([0.25, 0.5, 0.75, 1.0]))

    def test_weights_shift_the_result_toward_one_member(self):
        from nba_longevity_player import blend_rank_scores
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])   # exactly reversed
        biased = blend_rank_scores(a, b, weights=[0.9, 0.1])
        assert list(np.argsort(biased)) == list(np.argsort(a))

    def test_output_is_scale_invariant(self):
        from nba_longevity_player import blend_rank_scores
        a = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(blend_rank_scores(a), blend_rank_scores(a * 1000))

    def test_rejects_length_mismatch(self):
        from nba_longevity_player import blend_rank_scores
        with pytest.raises(ValueError, match="same length"):
            blend_rank_scores(np.array([1.0, 2.0]), np.array([1.0]))

    def test_rejects_weight_count_mismatch(self):
        from nba_longevity_player import blend_rank_scores
        with pytest.raises(ValueError, match="weights"):
            blend_rank_scores(np.array([1.0, 2.0]), weights=[0.5, 0.5])
