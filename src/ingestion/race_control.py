"""RaceControlEvent table: one row per race control message.

Schema (Design.md Section 4, "RaceControlEvent"):
    race_id, lap, event_type (SC / VSC / RedFlag / YellowFlag)
Extended:
    time_seconds, category, message, status, flag, scope, sector

LEAKAGE SAFETY:
Race control events are stamped with the `lap` they occurred on so that
`as_of_lap()` can strictly exclude future race control events.
"""

from __future__ import annotations

import re
import pandas as pd

RACE_CONTROL_COLUMNS = [
    "race_id", "lap", "event_type", "message",
    "time_seconds", "category", "status", "flag", "scope", "sector",
]


def _classify_event_type(message: str, category: str, flag: str) -> str:
    """Classify race control message into standard event types."""
    msg = str(message).upper() if message is not None else ""
    cat = str(category).upper() if category is not None else ""
    flg = str(flag).upper() if flag is not None else ""

    if "SAFETY CAR DEPLOYED" in msg or "SAFETY CAR IN THIS LAP" in msg or (cat == "SAFETYCAR" and "VIRTUAL" not in msg):
        return "SAFETY_CAR"
    elif "VIRTUAL SAFETY CAR" in msg or "VSC" in msg or (cat == "SAFETYCAR" and "VIRTUAL" in msg):
        return "VIRTUAL_SAFETY_CAR"
    elif "RED FLAG" in msg or flg == "RED":
        return "RED_FLAG"
    elif "YELLOW" in msg or flg == "YELLOW" or flg == "DOUBLE YELLOW":
        return "YELLOW_FLAG"
    elif "CLEAR" in msg or "GREEN" in msg or flg == "GREEN" or "TRACK CLEAR" in msg:
        return "TRACK_CLEAR"
    elif "CHEQUERED" in msg or flg == "CHEQUERED":
        return "CHEQUERED_FLAG"
    return "OTHER"


def _seconds(td: pd.Series) -> pd.Series:
    """Convert timedelta series to float seconds."""
    if td is None:
        return pd.Series(dtype=float)
    return td.dt.total_seconds()


def _elapsed_seconds(ts: pd.Series, t0) -> pd.Series:
    """Convert absolute UTC timestamps to seconds elapsed since session start.

    `race_control_messages.Time` is an absolute UTC timestamp (FastF1
    `to_datetime(entry['Utc'])`), unlike `laps.Time` which is already a
    session-relative Timedelta. `session.t0_date` is the same reference point
    FastF1 uses internally to convert between the two.
    """
    if ts is None:
        return pd.Series(dtype=float)
    if t0 is None:
        # Source has no session start time (e.g. 2018 Bahrain): no elapsed time.
        return pd.Series(float("nan"), index=ts.index)
    return (ts - t0).dt.total_seconds()


def build_race_control_table(session, race_id: str) -> pd.DataFrame:
    """Build the RaceControlEvent table from a loaded FastF1 session."""
    rc_messages = session.race_control_messages
    laps = session.laps

    if rc_messages is None or rc_messages.empty:
        return pd.DataFrame(columns=RACE_CONTROL_COLUMNS)

    df = rc_messages.copy()
    time_s = _elapsed_seconds(df["Time"], session.t0_date)

    # Map message time to lap if Lap is missing/null in the message
    lap_times = []
    if laps is not None and not laps.empty:
        valid_laps = laps.dropna(subset=["LapNumber", "Time"])
        if not valid_laps.empty:
            finish_s = _seconds(valid_laps["Time"])
            leader_times = finish_s.groupby(valid_laps["LapNumber"]).min().sort_index()
            lap_times = list(zip(leader_times.index.astype(int), leader_times.values))

    resolved_laps = []
    for idx, row in df.iterrows():
        raw_lap = row.get("Lap")
        if pd.notna(raw_lap) and float(raw_lap) > 0:
            resolved_laps.append(int(raw_lap))
        else:
            t = time_s.iloc[idx]
            # Match against leader lap times
            assigned_lap = 1
            if pd.notna(t) and lap_times:
                for lap_num, finish_t in lap_times:
                    if t <= finish_t:
                        assigned_lap = lap_num
                        break
                else:
                    assigned_lap = lap_times[-1][0]
            resolved_laps.append(assigned_lap)

    event_types = [
        _classify_event_type(
            row.get("Message"),
            row.get("Category"),
            row.get("Flag"),
        )
        for _, row in df.iterrows()
    ]

    table = pd.DataFrame({
        "race_id": race_id,
        "lap": resolved_laps,
        "event_type": event_types,
        "message": df["Message"].astype("string"),
        "time_seconds": time_s,
        "category": df["Category"].astype("string") if "Category" in df else "",
        "status": df["Status"].astype("string") if "Status" in df else "",
        "flag": df["Flag"].astype("string") if "Flag" in df else "",
        "scope": df["Scope"].astype("string") if "Scope" in df else "",
        "sector": df["Sector"].astype("string") if "Sector" in df else "",
    })

    return table[RACE_CONTROL_COLUMNS].sort_values(["lap", "time_seconds"]).reset_index(drop=True)


def load_race_control(session, race_id: str) -> pd.DataFrame:
    """RaceControlEvent table for a loaded FastF1 race session."""
    return build_race_control_table(session, race_id)
