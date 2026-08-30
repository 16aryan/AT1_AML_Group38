"""Fixtures: a small hand-built season table with known correct answers.

The frame is written so that every aggregation has an answer that can be
checked by hand, rather than by re-running the code under test.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def season_frame():
    """Four players with deliberately awkward structure.

    - pid 1: three seasons, improving, label 0 in every row
    - pid 2: one season only (std would be undefined), label 1
    - pid 3: two seasons with *inconsistent* labels (0 then 1)
    - pid 4: two seasons, changes team, has a missing stat and an infinity
    """
    return pd.DataFrame({
        "pid":  [1, 1, 1, 2, 3, 3, 4, 4],
        "year": [2010, 2011, 2012, 2011, 2013, 2014, 2015, 2016],
        "team": ["A", "A", "A", "B", "C", "C", "D", "E"],
        "conf": ["X", "X", "X", "Y", "Z", "Z", "X", "X"],
        "yr":   ["Fr", "So", "Jr", "Fr", "Fr", "So", "Fr", "So"],
        "role": ["G", "G", "G", "F", "C", "C", "G", "G"],
        "pts":  [1.0, 2.0, 6.0, 10.0, 4.0, 8.0, np.nan, 5.0],
        "bpm":  [0.0, 1.0, 2.0, 5.0, 1.0, 3.0, np.inf, 2.0],
        "ht":   ["6-0", "6-0", "6-1", "6-10", "7-0", "7-0", "5-11", "5-11"],
        "dob":  ["1991-01-01"] * 3 + ["1992-06-01"] + ["1994-03-01"] * 2 + ["1996-02-01"] * 2,
        "num":  ["23", "23", "23", "0", "abc", "12", "7", "7"],
        "type": ["t"] * 8,
        "label": [0, 0, 0, 1, 0, 1, 0, 0],
    })


@pytest.fixture
def sample_submission_frame():
    return pd.DataFrame({"pid": [1, 2, 3, 4], "label": [0.1, 0.2, 0.3, 0.4]})
