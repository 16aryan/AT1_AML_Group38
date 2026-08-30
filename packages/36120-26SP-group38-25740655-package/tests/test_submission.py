"""Tests for submission assembly and the checks that catch a zero-scoring upload."""

import numpy as np
import pandas as pd
import pytest

from nba_longevity_player import (
    build_submission,
    ranks_to_unit_interval,
    save_submission,
    validate_submission,
)


class TestRanksToUnitInterval:
    def test_output_lies_strictly_inside_zero_one(self):
        out = ranks_to_unit_interval([0.1, 5.0, -3.0, 2.2])
        assert (out > 0).all() and (out < 1).all()

    def test_preserves_ordering_exactly(self):
        # Average precision reads only the ordering, so this must be exact.
        scores = np.array([0.9, 0.1, 0.5, 0.3])
        out = ranks_to_unit_interval(scores)
        assert list(np.argsort(out)) == list(np.argsort(scores))

    def test_is_monotonic_in_the_input(self):
        out = ranks_to_unit_interval([1.0, 2.0, 3.0, 4.0])
        assert list(out) == sorted(out)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            ranks_to_unit_interval([])


class TestBuildSubmission:
    def test_produces_one_row_per_player(self):
        sub = build_submission([1, 2, 3], [0.2, 0.9, 0.5])
        assert list(sub.columns) == ["pid", "label"]
        assert len(sub) == 3

    def test_rank_mode_preserves_score_ordering(self):
        sub = build_submission([1, 2, 3], [0.2, 0.9, 0.5])
        assert sub.loc[1, "label"] > sub.loc[2, "label"] > sub.loc[0, "label"]

    def test_raw_mode_passes_scores_through(self):
        sub = build_submission([1, 2], [0.25, 0.75], as_ranks=False)
        assert sub["label"].tolist() == [0.25, 0.75]

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            build_submission([1, 2, 3], [0.5, 0.5])

    def test_rejects_duplicate_players(self):
        # The model predicts per player; duplicates mean the caller passed
        # season rows and skipped aggregation.
        with pytest.raises(ValueError, match="already be unique"):
            build_submission([1, 1, 2], [0.1, 0.2, 0.3])


class TestValidateSubmission:
    @pytest.fixture
    def good(self):
        return pd.DataFrame({"pid": [1, 2, 3, 4], "label": [0.1, 0.2, 0.3, 0.4]})

    def test_accepts_a_valid_submission(self, good, sample_submission_frame):
        validate_submission(good, sample_submission_frame, expected_rows=4)

    def test_rejects_missing_column(self, good):
        with pytest.raises(ValueError, match="missing required column"):
            validate_submission(good.rename(columns={"label": "pred"}), expected_rows=4)

    def test_rejects_wrong_row_count(self, good):
        with pytest.raises(ValueError, match="expected 6261"):
            validate_submission(good)

    def test_rejects_duplicate_ids(self):
        bad = pd.DataFrame({"pid": [1, 1], "label": [0.1, 0.2]})
        with pytest.raises(ValueError, match="duplicate id"):
            validate_submission(bad, expected_rows=2)

    def test_rejects_missing_predictions(self):
        bad = pd.DataFrame({"pid": [1, 2], "label": [0.1, np.nan]})
        with pytest.raises(ValueError, match="missing prediction"):
            validate_submission(bad, expected_rows=2)

    def test_rejects_out_of_range_predictions(self):
        bad = pd.DataFrame({"pid": [1, 2], "label": [0.1, 1.4]})
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_submission(bad, expected_rows=2)

    def test_rejects_mismatched_id_set(self, sample_submission_frame):
        # The check worth having: right shape, wrong ids, scores near zero
        # with no error message from Kaggle.
        bad = pd.DataFrame({"pid": [1, 2, 3, 99], "label": [0.1, 0.2, 0.3, 0.4]})
        with pytest.raises(ValueError, match="do not match the sample submission"):
            validate_submission(bad, sample_submission_frame, expected_rows=4)


class TestSaveSubmission:
    def test_writes_a_validated_file(self, tmp_path, sample_submission_frame):
        sub = pd.DataFrame({"pid": [1, 2, 3, 4], "label": [0.1, 0.2, 0.3, 0.4]})
        path = save_submission(sub, tmp_path / "nested" / "sub.csv",
                               sample_submission=sample_submission_frame,
                               expected_rows=4)
        assert path.is_file()
        pd.testing.assert_frame_equal(pd.read_csv(path), sub)

    def test_refuses_to_write_an_invalid_file(self, tmp_path):
        bad = pd.DataFrame({"pid": [1, 1], "label": [0.1, 0.2]})
        path = tmp_path / "sub.csv"
        with pytest.raises(ValueError):
            save_submission(bad, path, expected_rows=2)
        assert not path.exists()
