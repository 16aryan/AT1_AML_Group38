"""Tests for season cleaning and player-level aggregation."""

import numpy as np
import pandas as pd
import pytest

from nba_longevity_player import (
    assert_no_target_leakage,
    drop_nuisance_features,
    within_season_zscore,
    align_categoricals,
    build_player_table,
    height_to_inches,
    numeric_feature_columns,
    prepare_season_rows,
)


class TestHeightToInches:
    @pytest.mark.parametrize("value,expected", [
        ("6-0", 72.0), ("6-7", 79.0), ("7-0", 84.0), ("5-11", 71.0),
    ])
    def test_parses_feet_inches(self, value, expected):
        assert height_to_inches(value) == expected

    def test_accepts_en_dash(self):
        # The source data mixes '-' and '–'; both must parse identically.
        assert height_to_inches("6–7") == height_to_inches("6-7") == 79.0

    @pytest.mark.parametrize("value", [None, np.nan, "", "six-seven", "6", "6-7-2"])
    def test_unparseable_becomes_nan_rather_than_raising(self, value):
        assert np.isnan(height_to_inches(value))


class TestPrepareSeasonRows:
    def test_derives_numeric_columns_from_text(self, season_frame):
        out = prepare_season_rows(season_frame)
        assert out.loc[0, "height_inches"] == 72.0
        assert out.loc[0, "age_at_season"] == 2010 - 1991
        assert out.loc[0, "jersey_number"] == 23

    def test_drops_the_raw_text_columns(self, season_frame):
        out = prepare_season_rows(season_frame)
        for column in ("ht", "dob", "num", "type"):
            assert column not in out.columns

    def test_unparseable_jersey_becomes_nan(self, season_frame):
        out = prepare_season_rows(season_frame)
        assert np.isnan(out.loc[4, "jersey_number"])  # 'abc'

    def test_infinities_become_missing(self, season_frame):
        # An infinity breaks an XGBoost fit outright, unlike a missing value.
        out = prepare_season_rows(season_frame)
        assert not np.isinf(out.select_dtypes(include=np.number).to_numpy()).any()
        assert np.isnan(out.loc[6, "bpm"])

    def test_does_not_mutate_its_input(self, season_frame):
        before = season_frame.copy()
        prepare_season_rows(season_frame)
        pd.testing.assert_frame_equal(season_frame, before)


class TestNumericFeatureColumns:
    def test_excludes_identifiers_target_and_year(self, season_frame):
        train = prepare_season_rows(season_frame)
        test = prepare_season_rows(season_frame.drop(columns=["label"]))
        columns = numeric_feature_columns(train, test)
        for excluded in ("pid", "label", "year"):
            assert excluded not in columns

    def test_includes_the_derived_columns(self, season_frame):
        train = prepare_season_rows(season_frame)
        test = prepare_season_rows(season_frame.drop(columns=["label"]))
        columns = numeric_feature_columns(train, test)
        for derived in ("height_inches", "age_at_season", "jersey_number"):
            assert derived in columns

    def test_omits_columns_absent_from_test(self, season_frame):
        train = prepare_season_rows(season_frame)
        train["train_only_stat"] = 1.0
        test = prepare_season_rows(season_frame.drop(columns=["label"]))
        assert "train_only_stat" not in numeric_feature_columns(train, test)


class TestBuildPlayerTable:
    @pytest.fixture
    def player_table(self, season_frame):
        season = prepare_season_rows(season_frame)
        numeric = ["pts", "bpm"]
        return build_player_table(season, numeric, include_target=True)

    def test_one_row_per_player(self, player_table):
        assert len(player_table) == 4
        assert player_table["pid"].is_unique

    def test_preserves_first_seen_player_order(self, player_table):
        # Submission rows must line up with the ids the competition supplies.
        assert player_table["pid"].tolist() == [1, 2, 3, 4]

    def test_latest_takes_the_most_recent_season(self, player_table):
        row = player_table.set_index("pid").loc[1]
        assert row["pts_latest"] == 6.0   # 2012, not 2010
        assert row["yr"] == "Jr"

    def test_mean_and_max_computed_within_player(self, player_table):
        row = player_table.set_index("pid").loc[1]
        assert row["pts_mean"] == pytest.approx(3.0)  # (1+2+6)/3
        assert row["pts_max"] == 6.0

    def test_career_summary_counts(self, player_table):
        table = player_table.set_index("pid")
        assert table.loc[1, "career_observations"] == 3
        assert table.loc[1, "seasons_observed"] == 3
        assert table.loc[1, "career_span_years"] == 2   # 2012 - 2010
        assert table.loc[2, "career_observations"] == 1
        assert table.loc[2, "career_span_years"] == 0
        assert table.loc[4, "teams_observed"] == 2      # changed team

    def test_absolute_season_never_becomes_a_feature(self, player_table):
        # Train is 2008-2016 and test is 2020-2023; a split on absolute year
        # would key on an era that does not recur in the test set.
        for leaked in ("year", "_first_season", "_last_season",
                       "first_season", "last_season"):
            assert leaked not in player_table.columns

    def test_target_aggregates_with_max(self, player_table):
        table = player_table.set_index("pid")
        # pid 3 has labels [0, 1]; a career outcome belongs to the player.
        assert table.loc[3, "label"] == 1
        assert table.loc[1, "label"] == 0

    def test_target_absent_unless_requested(self, season_frame):
        season = prepare_season_rows(season_frame)
        table = build_player_table(season, ["pts"], include_target=False)
        assert "label" not in table.columns

    def test_nan_input_does_not_poison_the_mean(self, season_frame):
        season = prepare_season_rows(season_frame)
        table = build_player_table(season, ["pts"]).set_index("pid")
        # pid 4 has pts [nan, 5.0]; pandas skips the nan.
        assert table.loc[4, "pts_mean"] == pytest.approx(5.0)

    def test_rejects_unknown_aggregation(self, season_frame):
        season = prepare_season_rows(season_frame)
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            build_player_table(season, ["pts"], stats=("latest", "median"))

    def test_optional_stats_are_honoured(self, season_frame):
        season = prepare_season_rows(season_frame)
        table = build_player_table(season, ["pts"], stats=("mean",))
        assert "pts_mean" in table.columns
        assert "pts_latest" not in table.columns
        assert "pts_max" not in table.columns


class TestAlignCategoricals:
    def test_shared_category_codes_across_frames(self):
        train = pd.DataFrame({"conf": ["X", "Y"]})
        test = pd.DataFrame({"conf": ["Y", "Z"]})
        a, b = align_categoricals(train, test, ("conf",))
        assert list(a["conf"].cat.categories) == list(b["conf"].cat.categories)
        # 'Y' must encode identically on both sides.
        assert a["conf"].cat.codes[1] == b["conf"].cat.codes[0]

    def test_ignores_columns_not_present(self):
        train = pd.DataFrame({"a": [1]})
        test = pd.DataFrame({"a": [2]})
        a, b = align_categoricals(train, test, ("conf",))
        pd.testing.assert_frame_equal(a, train)
        pd.testing.assert_frame_equal(b, test)


class TestAssertNoTargetLeakage:
    def test_passes_for_clean_features(self):
        test = pd.DataFrame({"pts_mean": [1.0], "bpm_max": [2.0]})
        assert_no_target_leakage(["pts_mean", "bpm_max"], test)

    @pytest.mark.parametrize("forbidden", ["label", "pid"])
    def test_rejects_target_or_identifier(self, forbidden):
        test = pd.DataFrame({"pts_mean": [1.0], forbidden: [0]})
        with pytest.raises(ValueError, match="must not be used as a feature"):
            assert_no_target_leakage(["pts_mean", forbidden], test)

    def test_rejects_feature_absent_from_test(self):
        test = pd.DataFrame({"pts_mean": [1.0]})
        with pytest.raises(ValueError, match="absent from the test set"):
            assert_no_target_leakage(["pts_mean", "ghost_stat"], test)


class TestWithinSeasonZscore:
    @pytest.fixture
    def two_cohorts(self):
        """Two seasons on deliberately different scales.

        2010 is centred near 10 and 2020 near 100. The *ranking inside* each
        season is identical, so a correct transform must map them onto the same
        values despite the tenfold level difference.
        """
        return pd.DataFrame({
            "pid": [1, 2, 3, 4, 5, 6],
            "year": [2010, 2010, 2010, 2020, 2020, 2020],
            "pts": [8.0, 10.0, 12.0, 80.0, 100.0, 120.0],
        })

    def test_removes_the_level_difference_between_eras(self, two_cohorts):
        out = within_season_zscore(two_cohorts, ["pts"])
        early = out.loc[out["year"] == 2010, "pts"].to_numpy()
        late = out.loc[out["year"] == 2020, "pts"].to_numpy()
        # Identical within-season ranking must produce identical z-scores.
        np.testing.assert_allclose(early, late)

    def test_each_season_is_centred(self, two_cohorts):
        out = within_season_zscore(two_cohorts, ["pts"])
        for year in (2010, 2020):
            assert out.loc[out["year"] == year, "pts"].mean() == pytest.approx(0.0)

    def test_preserves_ordering_inside_a_season(self, two_cohorts):
        out = within_season_zscore(two_cohorts, ["pts"])
        early = out.loc[out["year"] == 2010, "pts"].tolist()
        assert early == sorted(early)

    def test_zero_variance_season_yields_nan_not_infinity(self):
        # An infinity would break the model fit; a nan is handled natively.
        frame = pd.DataFrame({"pid": [1, 2], "year": [2010, 2010], "pts": [5.0, 5.0]})
        out = within_season_zscore(frame, ["pts"])
        assert out["pts"].isna().all()
        assert not np.isinf(out["pts"].to_numpy()).any()

    def test_leaves_other_columns_untouched(self, two_cohorts):
        out = within_season_zscore(two_cohorts, ["pts"])
        pd.testing.assert_series_equal(out["pid"], two_cohorts["pid"])

    def test_does_not_mutate_its_input(self, two_cohorts):
        before = two_cohorts.copy()
        within_season_zscore(two_cohorts, ["pts"])
        pd.testing.assert_frame_equal(two_cohorts, before)

    def test_raises_on_absent_column(self, two_cohorts):
        with pytest.raises(KeyError, match="absent columns"):
            within_season_zscore(two_cohorts, ["not_a_column"])


class TestDropNuisanceFeatures:
    def test_removes_the_column_and_its_aggregates(self):
        frame = pd.DataFrame({
            "jersey_number": [1], "jersey_number_mean": [1.0],
            "jersey_number_max": [2.0], "pts_mean": [3.0],
        })
        (out,) = drop_nuisance_features(frame, nuisance=("jersey_number",))
        assert list(out.columns) == ["pts_mean"]

    def test_does_not_remove_merely_similar_names(self):
        frame = pd.DataFrame({"jersey_numberish": [1.0], "pts_mean": [2.0]})
        (out,) = drop_nuisance_features(frame, nuisance=("jersey_number",))
        assert "jersey_numberish" in out.columns

    def test_applies_the_same_removal_to_every_frame(self):
        a = pd.DataFrame({"jersey_number_mean": [1.0], "pts_mean": [2.0]})
        b = pd.DataFrame({"jersey_number_mean": [3.0], "pts_mean": [4.0]})
        out_a, out_b = drop_nuisance_features(a, b, nuisance=("jersey_number",))
        assert list(out_a.columns) == list(out_b.columns) == ["pts_mean"]

    def test_absent_nuisance_column_is_harmless(self):
        frame = pd.DataFrame({"pts_mean": [1.0]})
        (out,) = drop_nuisance_features(frame, nuisance=("jersey_number",))
        assert list(out.columns) == ["pts_mean"]
