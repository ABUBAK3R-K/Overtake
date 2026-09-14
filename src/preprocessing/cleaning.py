"""Row filters and fixes for the Lap table, applied before any modeling.

Every rule here uses only the row itself or other rows from the *same lap*,
so each is safe to apply to a frame that has already been through
as_of_lap(): the result for lap N never depends on laps after N.

Rules come from notebooks/01_eda.ipynb.
"""

import numpy as np
import pandas as pd

# FastF1 track status codes: 4 = safety car, 5 = red flag, 6/7 = VSC.
NEUTRALISED_CODES = "[4567]"
RED_FLAG_CODE = "5"

# A clean lap slower than this multiple of the same-lap field median is
# treated as an outlier (damage, off-track, unflagged yellow, traffic).
SLOW_LAP_RATIO = 1.07

# Largest believable gap to the leader. Lapped cars reach ~330 s (Monaco);
# the 600-4000 s gaps in the data are red-flag stoppages and retirements.
MAX_GAP_TO_LEADER_S = 600.0

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def is_neutralised(track_status: pd.Series) -> pd.Series:
    """True where the lap saw a safety car, VSC or red flag."""
    return track_status.astype("string").fillna("").str.contains(NEUTRALISED_CODES)


def is_racing_lap(laps: pd.DataFrame) -> pd.Series:
    """Laps that reflect car + tyre pace: not lap 1 (standing start), not an
    in- or out-lap, not neutralised, with an accurate recorded time."""
    return (
        (laps["lap_number"] > 1)
        & ~laps["pit_flag"].astype(bool)
        & ~laps["pit_out_flag"].astype(bool)
        & ~is_neutralised(laps["track_status"])
        & laps["lap_time"].notna()
        & laps["is_accurate"].astype(bool)
    )


def is_clean_lap(laps: pd.DataFrame, slow_ratio: float = SLOW_LAP_RATIO) -> pd.Series:
    """Racing laps that aren't slow outliers against the same-lap field median.

    The median is taken over racing laps of the same race and lap number, so
    it uses nothing from later laps.
    """
    racing = is_racing_lap(laps)
    field_median = (
        laps["lap_time"].where(racing)
        .groupby([laps["race_id"], laps["lap_number"]])
        .transform("median")
    )
    return racing & (laps["lap_time"] < slow_ratio * field_median)


def clean_gap_to_leader(laps: pd.DataFrame,
                        max_gap_s: float = MAX_GAP_TO_LEADER_S) -> pd.Series:
    """gap_to_leader with red-flag laps and impossible gaps set to NaN.

    Ingestion keeps the raw gap; use this series for features instead.
    """
    gap = laps["gap_to_leader"].astype(float)
    red_flag = laps["track_status"].astype("string").fillna("").str.contains(RED_FLAG_CODE)
    return gap.mask(red_flag | (gap > max_gap_s), np.nan)
