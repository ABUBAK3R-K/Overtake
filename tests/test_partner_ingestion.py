"""Offline tests for the Partner ingestion tables: weather, race_control, and pit_stops."""

from unittest.mock import MagicMock
import pandas as pd
import pytest

from src.ingestion.pit_stops import build_pit_stops_table
from src.ingestion.race_control import build_race_control_table
from src.ingestion.weather import build_weather_table

s = pd.to_timedelta


@pytest.fixture
def mock_session() -> MagicMock:
    """Mock session with laps, weather, and race control data."""
    session = MagicMock()
    session.total_laps = 3

    # Laps
    session.laps = pd.DataFrame({
        "Driver": ["VER", "HAM"] * 3,
        "DriverNumber": ["1", "44"] * 3,
        "Team": ["Red Bull", "Mercedes"] * 3,
        "LapNumber": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "LapStartTime": s([0, 0, 90, 91, 180, 185], unit="s"),
        "Time": s([90, 91, 180, 185, 270, 300], unit="s"),
        "LapTime": s([90, 91, 90, 94, 90, 115], unit="s"),
        "Position": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        "PitInTime": [pd.NaT, pd.NaT, pd.NaT, s(184, unit="s"), pd.NaT, pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, s(208, unit="s")],
        "TrackStatus": ["1"] * 6,
        "Stint": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "Compound": ["MEDIUM", "SOFT", "MEDIUM", "SOFT", "MEDIUM", "HARD"],
        "TyreLife": [1.0, 4.0, 2.0, 5.0, 3.0, 1.0],
        "FreshTyre": [True, False, True, False, True, True],
    })

    # Weather
    session.weather_data = pd.DataFrame({
        "Time": s([30, 120, 220], unit="s"),
        "AirTemp": [24.0, 25.0, 26.0],
        "TrackTemp": [32.0, 34.0, 35.0],
        "Humidity": [55.0, 52.0, 50.0],
        "Pressure": [1012.0, 1012.5, 1013.0],
        "WindSpeed": [4.5, 5.0, 5.5],
        "Rainfall": [False, False, False],
    })

    # Race control messages
    session.race_control_messages = pd.DataFrame({
        "Time": s([40, 182, 250], unit="s"),
        "Lap": [1.0, 2.0, 3.0],
        "Category": ["Flag", "SafetyCar", "Flag"],
        "Message": ["YELLOW FLAG IN SECTOR 2", "SAFETY CAR DEPLOYED", "TRACK CLEAR"],
        "Status": ["DEPLOYED", "DEPLOYED", "CLEAR"],
        "Flag": ["YELLOW", "", "CLEAR"],
        "Scope": ["Sector", "Track", "Track"],
        "Sector": ["2", "", ""],
    })

    return session


# --- Weather table tests ---

def test_weather_table_builds_lap_rows(mock_session):
    weather = build_weather_table(mock_session, "test_race")
    assert len(weather) == 3
    assert list(weather["lap"]) == [1, 2, 3]
    assert weather.loc[0, "track_temp"] == 32.0
    assert weather.loc[1, "track_temp"] == 34.0
    assert not weather["is_wet"].any()


def test_weather_table_handles_empty_weather():
    session = MagicMock()
    session.total_laps = 2
    session.laps = pd.DataFrame({"LapNumber": [1, 2], "Time": s([90, 180], unit="s")})
    session.weather_data = None
    weather = build_weather_table(session, "test_race")
    assert len(weather) == 2
    assert "track_temp" in weather.columns
    assert weather.loc[0, "track_temp"] == 30.0


# --- Race Control table tests ---

def test_race_control_classifies_events(mock_session):
    rc = build_race_control_table(mock_session, "test_race")
    assert len(rc) == 3
    assert rc.loc[0, "event_type"] == "YELLOW_FLAG"
    assert rc.loc[1, "event_type"] == "SAFETY_CAR"
    assert rc.loc[2, "event_type"] == "TRACK_CLEAR"


def test_race_control_resolves_missing_lap(mock_session):
    # Set Lap to NaN for the SC message
    mock_session.race_control_messages.loc[1, "Lap"] = None
    rc = build_race_control_table(mock_session, "test_race")
    # Time is 182s, Leader Lap 2 finishes at 180s, Leader Lap 3 finishes at 270s -> Lap 3 (or resolved to 3)
    assert rc.loc[1, "lap"] == 3 or rc.loc[1, "lap"] == 2


# --- Pit Stop table tests ---

def test_pit_stops_table_extracts_stop(mock_session):
    pit_stops = build_pit_stops_table(mock_session.laps, "test_race")
    assert len(pit_stops) == 1
    stop = pit_stops.iloc[0]
    assert stop["driver"] == "HAM"
    assert stop["lap"] == 2
    assert stop["compound_before"] == "SOFT"
    assert stop["compound_after"] == "HARD"
    assert stop["pit_duration"] == 24.0  # 208s - 184s
