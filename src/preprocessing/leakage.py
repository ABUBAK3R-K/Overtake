"""No-leakage guard shared by both lanes (Design.md Section 5).

"As of lap N" means the moment lap N has just been completed: every row for
laps 1..N is known, nothing from lap N+1 onwards is. Replay state, features,
model inputs and strategy calls at lap N must all be built from data filtered
through as_of_lap(..., current_lap=N).

Rows whose lap is missing are dropped. A row with an unknown lap can't be
shown to be in the past, so it is treated as possible future data.
"""

import numbers

import pandas as pd


class LeakageError(AssertionError):
    """Raised when a frame contains rows from after the current lap."""


def _check_current_lap(current_lap) -> None:
    # bool is an int subclass; as_of_lap(df, "lap", True) is almost surely a bug.
    if isinstance(current_lap, bool) or not isinstance(current_lap, numbers.Integral):
        raise TypeError(f"current_lap must be an int, got {current_lap!r}")


def as_of_lap(df: pd.DataFrame, lap_col: str, current_lap: int) -> pd.DataFrame:
    """Returns only rows with lap_col <= current_lap. Every feature or
    state lookup anywhere in the project routes through this.

    Returns a copy, so callers can add columns without touching the full
    race frame. The input is never modified.
    """
    if lap_col not in df.columns:
        raise KeyError(f"lap column {lap_col!r} not in frame columns {list(df.columns)}")
    _check_current_lap(current_lap)
    # NaN/NA <= N is False, so rows with a missing lap drop out here.
    known_past = (df[lap_col] <= current_lap).fillna(False).astype(bool)
    return df.loc[known_past].copy()


def assert_as_of_lap(df: pd.DataFrame, lap_col: str, current_lap: int) -> None:
    """Raise LeakageError if df holds any row not visible at current_lap.

    For checking the output of feature/state builders, e.g.
        features = build_features(laps, current_lap=20)
        assert_as_of_lap(features, "lap_number", 20)
    """
    if lap_col not in df.columns:
        raise KeyError(f"lap column {lap_col!r} not in frame columns {list(df.columns)}")
    _check_current_lap(current_lap)
    laps = df[lap_col]
    leaked = laps.isna() | (laps > current_lap)
    if leaked.any():
        bad = sorted(laps[leaked].dropna().unique().tolist())
        raise LeakageError(
            f"{int(leaked.sum())} row(s) not visible as of lap {current_lap} "
            f"(future laps: {bad}, missing laps: {int(laps[leaked].isna().sum())})"
        )
