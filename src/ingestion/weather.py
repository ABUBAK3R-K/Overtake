"""WeatherSnapshot table: one row per (race_id, lap).

Schema (Design.md Section 4, "WeatherSnapshot"):
    race_id, lap, track_temp, air_temp, is_wet
Extended:
    humidity, pressure, wind_speed, rainfall

LEAKAGE SAFETY:
Weather at lap N describes conditions during or up to lap N.
Rows are keyed by lap so that `as_of_lap()` can filter them strictly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLUMNS = [
    "race_id", "lap", "track_temp", "air_temp",
    "humidity", "pressure", "wind_speed", "rainfall", "is_wet",
]


def _seconds(td: pd.Series) -> pd.Series:
    """Convert timedelta series to float seconds."""
    if td is None:
        return pd.Series(dtype=float)
    return td.dt.total_seconds()


def build_weather_table(session, race_id: str) -> pd.DataFrame:
    """Build the per-lap Weather table from a loaded FastF1 session."""
    weather = session.weather_data
    laps = session.laps

    total_laps = getattr(session, "total_laps", None)
    if total_laps is None and laps is not None and not laps.empty:
        total_laps = int(laps["LapNumber"].max())
    elif total_laps is None:
        total_laps = 1
    else:
        total_laps = int(total_laps)

    if weather is None or weather.empty:
        # Fallback default values if no weather data exists
        rows = [{
            "race_id": race_id,
            "lap": lap,
            "track_temp": 30.0,
            "air_temp": 25.0,
            "humidity": 50.0,
            "pressure": 1013.0,
            "wind_speed": 5.0,
            "rainfall": False,
            "is_wet": False,
        } for lap in range(1, total_laps + 1)]
        return pd.DataFrame(rows)[WEATHER_COLUMNS]

    weather_df = weather.copy()
    weather_df["time_s"] = _seconds(weather_df["Time"])

    # Determine lap time boundaries from the race leader's finish times
    lap_times = {}
    if laps is not None and not laps.empty:
        valid_laps = laps.dropna(subset=["LapNumber", "Time"])
        if not valid_laps.empty:
            finish_s = _seconds(valid_laps["Time"])
            leader_times = finish_s.groupby(valid_laps["LapNumber"]).min()
            for lap_num, finish_time in leader_times.items():
                lap_times[int(lap_num)] = float(finish_time)

    rows = []
    prev_time = 0.0

    for lap in range(1, total_laps + 1):
        lap_end_time = lap_times.get(lap)
        if lap_end_time is not None:
            mask = (weather_df["time_s"] >= prev_time) & (weather_df["time_s"] <= lap_end_time)
            lap_weather = weather_df[mask]
            if lap_weather.empty:
                # Nearest prior observation
                prior = weather_df[weather_df["time_s"] <= lap_end_time]
                lap_weather = prior.tail(1) if not prior.empty else weather_df.head(1)
            prev_time = lap_end_time
        else:
            # If lap timing is unavailable, slice evenly across weather data
            frac = lap / total_laps
            idx = min(int(frac * len(weather_df)), len(weather_df) - 1)
            lap_weather = weather_df.iloc[[idx]]

        track_temp = float(lap_weather["TrackTemp"].mean()) if "TrackTemp" in lap_weather else 30.0
        air_temp = float(lap_weather["AirTemp"].mean()) if "AirTemp" in lap_weather else 25.0
        humidity = float(lap_weather["Humidity"].mean()) if "Humidity" in lap_weather else 50.0
        pressure = float(lap_weather["Pressure"].mean()) if "Pressure" in lap_weather else 1013.0
        wind_speed = float(lap_weather["WindSpeed"].mean()) if "WindSpeed" in lap_weather else 5.0
        rainfall = bool(lap_weather["Rainfall"].any()) if "Rainfall" in lap_weather else False
        is_wet = rainfall

        rows.append({
            "race_id": race_id,
            "lap": lap,
            "track_temp": track_temp,
            "air_temp": air_temp,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "rainfall": rainfall,
            "is_wet": is_wet,
        })

    table = pd.DataFrame(rows)
    return table[WEATHER_COLUMNS].sort_values("lap").reset_index(drop=True)


def load_weather(session, race_id: str) -> pd.DataFrame:
    """Weather table for a loaded FastF1 race session."""
    return build_weather_table(session, race_id)
