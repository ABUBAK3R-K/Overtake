"""Race table: one row per race.

Schema (Design.md Section 4, "Race"):
    race_id, circuit, season, total_laps, weather_summary
Added:
    round, event_name  - for display and for re-loading the session.

LEAKAGE WARNING: weather_summary describes the *whole* race (e.g. "wet"
if it rained at any point), so at lap 10 it can reveal rain on lap 60.
It is for display and race selection only; it must never be used as a
model feature. Per-lap weather comes from the WeatherSnapshot table.
"""

import pandas as pd

RACE_COLUMNS = [
    "race_id", "season", "round", "event_name", "circuit",
    "total_laps", "weather_summary",
]


def summarise_weather(weather: pd.DataFrame) -> str:
    """Whole-race weather label, e.g. "dry, track 25-31C"."""
    if weather is None or weather.empty:
        return "unknown"
    condition = "wet" if weather["Rainfall"].astype(bool).any() else "dry"
    low, high = weather["TrackTemp"].min(), weather["TrackTemp"].max()
    return f"{condition}, track {low:.0f}-{high:.0f}C"


def build_race_table(session, race_id: str, season: int, round_number: int) -> pd.DataFrame:
    """Build the one-row Race table from a loaded FastF1 session."""
    # total_laps = scheduled race distance. Fall back to laps actually run if
    # FastF1 has no lap-count data for this session.
    total_laps = session.total_laps
    if total_laps is None:
        total_laps = int(session.laps["LapNumber"].max())

    table = pd.DataFrame([{
        "race_id": race_id,
        "season": season,
        "round": round_number,
        "event_name": session.event["EventName"],
        "circuit": session.event["Location"],
        "total_laps": int(total_laps),
        "weather_summary": summarise_weather(session.weather_data),
    }])
    return table[RACE_COLUMNS]
