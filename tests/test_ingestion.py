"""Offline tests for the ingestion tables.

Each test builds a tiny hand-made frame shaped like FastF1's output, so the
suite runs in seconds with no network and no cache.
"""

import pandas as pd
import pytest

from src.ingestion.laps import build_lap_table
from src.ingestion.storage import connect, write_table
from src.ingestion.telemetry import assign_samples_to_laps, build_telemetry_table
from src.ingestion.tyres import build_tyre_table

s = pd.to_timedelta  # shorthand: s(90, "s") -> 90 seconds


@pytest.fixture
def fastf1_laps() -> pd.DataFrame:
    """Two drivers, three laps. VER leads; HAM pits at the end of lap 2."""
    return pd.DataFrame({
        "Driver": ["VER", "HAM"] * 3,
        "DriverNumber": ["1", "44"] * 3,
        "Team": ["Red Bull", "Mercedes"] * 3,
        "LapNumber": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        # Session clock at lap start / lap end.
        "LapStartTime": s([0, 0, 90, 91, 180, 185], unit="s"),
        "Time": s([90, 91, 180, 185, 270, 300], unit="s"),
        "LapTime": s([90, 91, 90, 94, 90, 115], unit="s"),
        "Sector1Time": [pd.NaT, pd.NaT] + list(s([30, 31, 30, 40], unit="s")),
        "Sector2Time": s([30, 30, 30, 31, 30, 40], unit="s"),
        "Sector3Time": s([30, 31, 30, 32, 30, 35], unit="s"),
        "Position": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        "PitInTime": [pd.NaT, pd.NaT, pd.NaT, s(184, unit="s"), pd.NaT, pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, s(208, unit="s")],
        "TrackStatus": ["1"] * 6,
        "IsAccurate": [False, False, True, True, True, False],
        "Stint": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "Compound": ["MEDIUM", "SOFT", "MEDIUM", "SOFT", "MEDIUM", "HARD"],
        "TyreLife": [1.0, 4.0, 2.0, 5.0, 3.0, 1.0],
        "FreshTyre": [True, False, True, False, True, True],
    })


# --- Lap table -------------------------------------------------------------

def test_lap_table_converts_times_to_seconds(fastf1_laps):
    laps = build_lap_table(fastf1_laps, "test_race")
    ver_lap1 = laps[(laps.driver == "VER") & (laps.lap_number == 1)].iloc[0]
    assert ver_lap1.lap_time == 90.0
    assert pd.isna(ver_lap1.sector_1_time)  # nulls kept, not cleaned


def test_gap_to_leader_is_zero_for_leader_and_positive_behind(fastf1_laps):
    laps = build_lap_table(fastf1_laps, "test_race").set_index(["driver", "lap_number"])
    assert laps.loc[("VER", 3), "gap_to_leader"] == 0.0
    assert laps.loc[("HAM", 2), "gap_to_leader"] == 5.0
    assert laps.loc[("HAM", 3), "gap_to_leader"] == 30.0


def test_pit_flags_mark_in_lap_and_out_lap(fastf1_laps):
    laps = build_lap_table(fastf1_laps, "test_race").set_index(["driver", "lap_number"])
    assert laps.loc[("HAM", 2), "pit_flag"] and not laps.loc[("HAM", 2), "pit_out_flag"]
    assert laps.loc[("HAM", 3), "pit_out_flag"] and not laps.loc[("HAM", 3), "pit_flag"]
    assert not laps.loc[("VER", 2), "pit_flag"]


def test_gap_to_leader_does_not_change_when_future_laps_are_removed(fastf1_laps):
    """No-leakage check: lap-2 gaps must be identical whether or not lap 3
    exists in the input, i.e. they don't depend on later laps."""
    full = build_lap_table(fastf1_laps, "r")
    truncated = build_lap_table(fastf1_laps[fastf1_laps.LapNumber <= 2], "r")
    pd.testing.assert_series_equal(
        full[full.lap_number <= 2]["gap_to_leader"].reset_index(drop=True),
        truncated["gap_to_leader"].reset_index(drop=True),
    )


# --- Tyre table ------------------------------------------------------------

def test_tyre_table_tracks_stint_start_and_used_tyre_age(fastf1_laps):
    tyres = build_tyre_table(fastf1_laps, "test_race").set_index(["driver", "lap_number"])
    # HAM started on used softs: age 4 on lap 1, not 1.
    assert tyres.loc[("HAM", 1), "tyre_age_at_lap"] == 4
    assert tyres.loc[("HAM", 2), "stint_start_lap"] == 1
    # New stint after the stop.
    assert tyres.loc[("HAM", 3), "compound"] == "HARD"
    assert tyres.loc[("HAM", 3), "stint_start_lap"] == 3


def test_tyre_table_does_not_change_when_future_laps_are_removed(fastf1_laps):
    full = build_tyre_table(fastf1_laps, "r")
    truncated = build_tyre_table(fastf1_laps[fastf1_laps.LapNumber <= 2], "r")
    pd.testing.assert_frame_equal(
        full[full.lap_number <= 2].reset_index(drop=True), truncated
    )


# --- Telemetry table -------------------------------------------------------

def _car_samples(times_s, speeds, throttle=100, brake=False, drs=0):
    return pd.DataFrame({
        "SessionTime": s(times_s, unit="s"),
        "Speed": speeds,
        "Throttle": throttle,
        "Brake": brake,
        "DRS": drs,
    })


def test_samples_are_assigned_to_the_lap_they_were_recorded_on(fastf1_laps):
    ver_laps = fastf1_laps[fastf1_laps.Driver == "VER"]
    # t=10 -> lap 1, t=100 -> lap 2, t=200 -> lap 3, t=400 -> after the race.
    samples = _car_samples([10, 100, 200, 400], [300, 310, 320, 50])
    tagged = assign_samples_to_laps(samples, ver_laps)
    assert tagged["LapNumber"].tolist() == [1.0, 2.0, 3.0]  # t=400 dropped


def test_telemetry_table_summarises_each_lap(fastf1_laps):
    car_data = {
        "1": _car_samples([10, 20, 100, 110], [300, 200, 310, 290],
                          throttle=[100, 50, 100, 100], brake=[False, True, False, False],
                          drs=[0, 0, 12, 8]),
        "44": _car_samples([10], [250]),
    }
    table = build_telemetry_table(car_data, fastf1_laps, "test_race")
    ver = table[table.driver == "VER"].set_index("lap_number")
    assert ver.loc[1, "mean_speed"] == 250.0
    assert ver.loc[1, "max_speed"] == 300.0
    assert ver.loc[1, "full_throttle_pct"] == 0.5
    assert ver.loc[1, "brake_pct"] == 0.5
    assert ver.loc[2, "drs_open_pct"] == 0.5  # 12 = open, 8 = only eligible
    assert set(table.driver) == {"VER", "HAM"}


def test_telemetry_skips_drivers_without_car_data(fastf1_laps):
    table = build_telemetry_table({"1": _car_samples([10], [300])}, fastf1_laps, "r")
    assert set(table.driver) == {"VER"}


# --- Storage ---------------------------------------------------------------

def test_tables_round_trip_through_parquet_and_duckdb(fastf1_laps, tmp_path):
    for race_id in ["race_a", "race_b"]:
        write_table(build_lap_table(fastf1_laps, race_id), "laps", race_id, base_dir=tmp_path)

    con = connect(base_dir=tmp_path)
    counts = con.sql(
        "SELECT race_id, COUNT(*) AS n FROM laps GROUP BY race_id ORDER BY race_id"
    ).fetchall()
    assert counts == [("race_a", 6), ("race_b", 6)]
