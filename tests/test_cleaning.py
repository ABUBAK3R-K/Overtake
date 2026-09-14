"""Tests for the Lap-table filters in src/preprocessing/cleaning.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaning import (
    clean_gap_to_leader, is_clean_lap, is_neutralised, is_racing_lap,
)
from src.preprocessing.leakage import as_of_lap


@pytest.fixture
def laps() -> pd.DataFrame:
    """Lap table rows: 4 drivers on laps 1-3, with one of each problem."""
    n = 12
    df = pd.DataFrame({
        "race_id": "r",
        "driver": ["A", "B", "C", "D"] * 3,
        "lap_number": [1] * 4 + [2] * 4 + [3] * 4,
        "lap_time": [100.0, 101, 102, 103,
                     90.0, 90.5, 91, 99.0,     # D lap 2: 10% off the pace
                     90.0, np.nan, 91, 120],   # B lap 3: no time; D lap 3: in-lap
        "pit_flag": [False] * n,
        "pit_out_flag": [False] * n,
        "track_status": ["1"] * n,
        "is_accurate": [True] * n,
        "gap_to_leader": [0.0, 1, 2, 3, 0, 1.5, 3, 12, 0, 2, 4, 3100],
    })
    df.loc[11, "pit_flag"] = True
    return df


def test_neutralised_codes():
    status = pd.Series(["1", "12", "4", "671", "25", None])
    assert is_neutralised(status).tolist() == [False, False, True, True, True, False]


def test_racing_lap_excludes_lap_one_pit_laps_and_missing_times(laps):
    racing = is_racing_lap(laps)
    assert not racing[laps.lap_number == 1].any()
    assert not racing[11]      # in-lap
    assert not racing[9]       # no lap time
    assert racing[[4, 5, 6, 7, 8, 10]].all()


def test_racing_lap_excludes_safety_car_and_inaccurate_laps(laps):
    laps.loc[4, "track_status"] = "14"
    laps.loc[5, "is_accurate"] = False
    racing = is_racing_lap(laps)
    assert not racing[4] and not racing[5]


def test_clean_lap_drops_slow_outlier_against_same_lap_field(laps):
    clean = is_clean_lap(laps)
    assert not clean[7]                 # 99.0 vs lap-2 median 90.75
    assert clean[[4, 5, 6, 8, 10]].all()


def test_clean_lap_on_lap_n_ignores_later_laps(laps):
    full = is_clean_lap(laps)
    for n in (2, 3):
        visible = as_of_lap(laps, "lap_number", n)
        pd.testing.assert_series_equal(is_clean_lap(visible), full[visible.index])


def test_gap_masked_on_red_flag_and_impossible_values(laps):
    laps.loc[6, "track_status"] = "15"
    gap = clean_gap_to_leader(laps)
    assert np.isnan(gap[6])            # red flag
    assert np.isnan(gap[11])           # 3100 s
    assert gap[7] == 12.0
    assert laps.loc[11, "gap_to_leader"] == 3100   # input untouched
