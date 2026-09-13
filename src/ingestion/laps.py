"""Lap table: one row per (race_id, driver, lap_number).

Schema (Design.md Section 4, "Lap"), all times in float seconds:
    race_id, driver, lap_number, lap_time, sector_1_time, sector_2_time,
    sector_3_time, position, gap_to_leader, pit_flag
Extra columns kept because the models need them:
    team, pit_out_flag, track_status, is_accurate

Ingestion stays faithful to the source: nulls (e.g. lap 1 has no sector 1
time because of the standing start) are kept, not cleaned. Cleaning and
feature engineering happen in src/preprocessing/, behind as_of_lap().
"""

import pandas as pd

LAP_COLUMNS = [
    "race_id", "driver", "team", "lap_number",
    "lap_time", "sector_1_time", "sector_2_time", "sector_3_time",
    "position", "gap_to_leader", "pit_flag", "pit_out_flag",
    "track_status", "is_accurate",
]


def _seconds(td: pd.Series) -> pd.Series:
    """Convert a FastF1 timedelta column to float seconds (NaT -> NaN)."""
    return td.dt.total_seconds()


def compute_gap_to_leader(laps: pd.DataFrame) -> pd.Series:
    """Gap (s) between each driver and whoever finished that same lap first.

    FastF1's `Time` column is the session clock when a driver *completed*
    the lap. The leader on lap N is the first driver to complete lap N, so:
        gap = my lap-N finish time - earliest lap-N finish time.

    No leakage: this only compares finish times of lap N itself, all of
    which exist once lap N is complete.
    """
    finish_s = _seconds(laps["Time"])
    leader_finish_s = finish_s.groupby(laps["LapNumber"]).transform("min")
    return finish_s - leader_finish_s


def build_lap_table(laps: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Turn a FastF1 `session.laps` frame into the Lap table."""
    table = pd.DataFrame({
        "race_id": race_id,
        "driver": laps["Driver"],
        "team": laps["Team"],
        "lap_number": laps["LapNumber"].astype("int64"),
        "lap_time": _seconds(laps["LapTime"]),
        "sector_1_time": _seconds(laps["Sector1Time"]),
        "sector_2_time": _seconds(laps["Sector2Time"]),
        "sector_3_time": _seconds(laps["Sector3Time"]),
        "position": laps["Position"].astype("Int64"),  # nullable int
        "gap_to_leader": compute_gap_to_leader(laps),
        # pit_flag = driver entered the pit lane at the end of this lap (in-lap).
        "pit_flag": laps["PitInTime"].notna(),
        # pit_out_flag = lap started from the pit lane (out-lap). Both kinds of
        # lap are slow for non-pace reasons, so the lap-time model needs both.
        "pit_out_flag": laps["PitOutTime"].notna(),
        # Raw FastF1 track status codes seen during the lap, e.g. "12" means
        # green then yellow. 4 = SC, 5 = red flag, 6/7 = VSC.
        "track_status": laps["TrackStatus"].astype("string"),
        "is_accurate": laps["IsAccurate"].astype(bool),
    })
    return table[LAP_COLUMNS].sort_values(["lap_number", "driver"]).reset_index(drop=True)


def load_laps(session, race_id: str) -> pd.DataFrame:
    """Lap table for a loaded FastF1 race session."""
    return build_lap_table(session.laps, race_id)
