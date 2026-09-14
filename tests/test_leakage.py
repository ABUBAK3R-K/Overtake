"""Tests for the shared no-leakage guard (Design.md Section 7: the single
highest-value test in the repo)."""

import pandas as pd
import pytest

from src.preprocessing.leakage import LeakageError, as_of_lap, assert_as_of_lap


@pytest.fixture
def laps() -> pd.DataFrame:
    """Two drivers, laps 1-4, shaped like the Lap table."""
    return pd.DataFrame({
        "race_id": "r",
        "driver": ["VER", "HAM"] * 4,
        "lap_number": [1, 1, 2, 2, 3, 3, 4, 4],
        "lap_time": [95.0, 96.0, 90.0, 91.0, 90.5, 92.0, 90.2, 115.0],
    })


def test_keeps_current_lap_and_drops_future_laps(laps):
    out = as_of_lap(laps, "lap_number", 2)
    assert out["lap_number"].tolist() == [1, 1, 2, 2]


def test_lap_zero_is_empty_and_past_the_end_is_everything(laps):
    assert as_of_lap(laps, "lap_number", 0).empty
    pd.testing.assert_frame_equal(as_of_lap(laps, "lap_number", 99), laps)


def test_every_lap_returns_nothing_beyond_it(laps):
    for n in range(0, 6):
        out = as_of_lap(laps, "lap_number", n)
        assert (out["lap_number"] <= n).all()
        assert len(out) == (laps["lap_number"] <= n).sum()


def test_does_not_modify_input_and_result_is_independent(laps):
    before = laps.copy()
    out = as_of_lap(laps, "lap_number", 2)
    out["lap_time"] = 0.0
    out["new_feature"] = 1
    pd.testing.assert_frame_equal(laps, before)


def test_rows_with_missing_lap_are_dropped():
    df = pd.DataFrame({"lap": pd.array([1, None, 3], dtype="Int64"), "x": [1, 2, 3]})
    assert as_of_lap(df, "lap", 5)["x"].tolist() == [1, 3]

    df_float = pd.DataFrame({"lap": [1.0, float("nan"), 3.0], "x": [1, 2, 3]})
    assert as_of_lap(df_float, "lap", 5)["x"].tolist() == [1, 3]


def test_works_with_any_lap_column_name():
    # e.g. the partner's PitStop / RaceControlEvent tables key on "lap".
    pit_stops = pd.DataFrame({"driver": ["HAM", "VER"], "lap": [14, 30]})
    assert as_of_lap(pit_stops, "lap", 20)["driver"].tolist() == ["HAM"]


def test_missing_column_raises(laps):
    with pytest.raises(KeyError):
        as_of_lap(laps, "lap", 2)


@pytest.mark.parametrize("bad", [2.5, "2", None, True])
def test_non_integer_current_lap_raises(laps, bad):
    with pytest.raises(TypeError):
        as_of_lap(laps, "lap_number", bad)


def test_assert_as_of_lap_passes_on_filtered_frame(laps):
    assert_as_of_lap(as_of_lap(laps, "lap_number", 3), "lap_number", 3)


def test_assert_as_of_lap_catches_future_and_missing_laps(laps):
    with pytest.raises(LeakageError, match="future laps: \\[3, 4\\]"):
        assert_as_of_lap(laps, "lap_number", 2)

    df = pd.DataFrame({"lap": [1.0, float("nan")]})
    with pytest.raises(LeakageError, match="missing laps: 1"):
        assert_as_of_lap(df, "lap", 5)
