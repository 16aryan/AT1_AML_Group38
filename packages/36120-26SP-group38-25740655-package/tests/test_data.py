"""Tests for data resolution, loading and the player-structure audit."""

import pandas as pd
import pytest

from nba_longevity_player import (
    CompetitionData,
    DataNotFoundError,
    audit_player_structure,
    load_competition_data,
    resolve_data_dir,
)


@pytest.fixture
def data_dir(tmp_path, season_frame):
    """A miniature competition folder on disk."""
    train = season_frame
    test = season_frame.drop(columns=["label"]).assign(year=lambda d: d["year"] + 10)
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"pid": [1, 2, 3, 4], "label": [0.5] * 4}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    return tmp_path


class TestResolveDataDir:
    def test_accepts_an_explicit_directory(self, data_dir):
        assert resolve_data_dir(data_dir) == data_dir.resolve()

    def test_reads_the_environment_variable(self, data_dir, monkeypatch):
        monkeypatch.chdir(data_dir.parent)
        monkeypatch.setenv("NBA_DATA_DIR", str(data_dir))
        assert resolve_data_dir() == data_dir.resolve()

    def test_explicit_argument_wins_over_environment(self, data_dir, monkeypatch):
        monkeypatch.setenv("NBA_DATA_DIR", str(data_dir.parent))
        assert resolve_data_dir(data_dir) == data_dir.resolve()

    def test_reports_where_it_looked_when_nothing_is_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NBA_DATA_DIR", raising=False)
        with pytest.raises(DataNotFoundError, match="Could not find train.csv"):
            resolve_data_dir(tmp_path / "nowhere")


class TestLoadCompetitionData:
    def test_loads_the_expected_tables(self, data_dir):
        data = load_competition_data(data_dir)
        assert isinstance(data, CompetitionData)
        assert len(data.train) == 8
        assert len(data.sample_submission) == 4

    def test_metadata_is_optional(self, data_dir):
        assert load_competition_data(data_dir).metadata is None

    def test_metadata_is_read_when_present(self, data_dir):
        pd.DataFrame({"feature": [1], "name": ["pts"], "description": ["Points"]}).to_csv(
            data_dir / "metadata.csv", index=False
        )
        assert load_competition_data(data_dir).metadata is not None

    def test_describe_summarises_every_table(self, data_dir):
        describe = load_competition_data(data_dir).describe()
        assert set(describe["table"]) == {"train", "test", "sample_submission"}


class TestAuditPlayerStructure:
    def test_counts_the_player_season_structure(self, season_frame):
        test = season_frame.drop(columns=["label"])
        audit = audit_player_structure(season_frame, test).set_index("check")["value"]
        assert audit["train_rows"] == 8
        assert audit["train_unique_players"] == 4
        # pids 1, 3 and 4 each appear more than once.
        assert audit["players_with_multiple_rows"] == 3
        assert audit["max_rows_for_one_player"] == 3

    def test_flags_inconsistent_player_labels(self, season_frame):
        test = season_frame.drop(columns=["label"])
        audit = audit_player_structure(season_frame, test).set_index("check")["value"]
        # pid 3 carries labels [0, 1] across its two seasons.
        assert audit["players_with_inconsistent_labels"] == 1

    def test_reports_the_year_ranges_that_motivate_era_validation(self, season_frame):
        test = season_frame.drop(columns=["label"]).assign(
            year=lambda d: d["year"] + 10
        )
        audit = audit_player_structure(season_frame, test).set_index("check")["value"]
        assert audit["train_year_range"] == "2010-2016"
        assert audit["test_year_range"] == "2020-2026"
