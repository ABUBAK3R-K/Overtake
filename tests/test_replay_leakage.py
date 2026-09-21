"""Tests for FR-4: Race Replay Engine and strict lap-by-lap no-leakage guarantees."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.leakage import LeakageError, as_of_lap
from src.simulation.replay import ReplaySession, _build_from_dfs, build_state_at_lap
from src.simulation.state import RaceState
from tests.test_lap_time_model import with_race_state
from tests.test_tyre_model import synthetic_race


def _create_synthetic_multi_table_race(race_id="2023_test", n_drivers=6, n_laps=30):
    """Create comprehensive multi-table synthetic race data."""
    raw_race = synthetic_race(race_id, "TestCirc", wear_scale=1.0, seed=123, n_drivers=n_drivers, n_laps=n_laps)
    laps_df = with_race_state(raw_race)

    # Tyres table
    tyres_rows = []
    for _, row in laps_df.iterrows():
        tyres_rows.append({
            "race_id": race_id,
            "driver": row["driver"],
            "lap_number": row["lap_number"],
            "compound": row["compound"],
            "tyre_age_at_lap": row["tyre_age_at_lap"],
        })
    tyres_df = pd.DataFrame(tyres_rows)

    # Pit stops table: driver D0 pits on lap 12, driver D1 pits on lap 22
    pit_stops_df = pd.DataFrame([
        {
            "race_id": race_id,
            "driver": "D00",
            "lap": 12,
            "stint": 1,
            "pit_duration": 22.4,
            "compound_before": "SOFT",
            "compound_after": "HARD",
        },
        {
            "race_id": race_id,
            "driver": "D01",
            "lap": 22,
            "stint": 1,
            "pit_duration": 21.8,
            "compound_before": "MEDIUM",
            "compound_after": "HARD",
        },
    ])

    # Weather table: dry until lap 20, rain starts on lap 21
    weather_rows = []
    for l in range(1, n_laps + 1):
        is_wet = l >= 21
        weather_rows.append({
            "race_id": race_id,
            "lap": l,
            "track_temp": 18.0 if is_wet else 32.0,
            "air_temp": 16.0 if is_wet else 26.0,
            "is_wet": is_wet,
        })
    weather_df = pd.DataFrame(weather_rows)

    # Race control: SC deployed on lap 25
    race_control_df = pd.DataFrame([
        {
            "race_id": race_id,
            "lap": 25,
            "event_type": "SAFETY_CAR",
            "message": "SAFETY CAR DEPLOYED",
            "category": "SAFETYCAR",
        }
    ])

    meta = {"circuit": "TestCirc", "total_laps": n_laps}
    return meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df


def test_replay_state_schema_and_intervals():
    meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df = _create_synthetic_multi_table_race()
    state = _build_from_dfs(
        race_id="2023_test",
        lap=10,
        meta=meta,
        all_laps=laps_df,
        all_tyres=tyres_df,
        all_pit_stops=pit_stops_df,
        all_weather=weather_df,
        all_race_control=race_control_df,
    )

    assert state.lap == 10
    assert state.circuit == "TestCirc"
    assert state.total_laps == 30
    assert len(state.positions) == 6

    # Intervals check: leader interval is 0.0, follower intervals equal delta to car ahead
    sorted_pos = sorted(state.positions.keys(), key=lambda d: state.positions[d])
    p1 = sorted_pos[0]
    p2 = sorted_pos[1]
    assert state.intervals[p1] == 0.0
    assert state.intervals[p2] == pytest.approx(state.gaps[p2] - state.gaps[p1], abs=1e-3)


def test_future_data_mutation_does_not_affect_past_state():
    """HARD CONSTRAINT: Modifying future laps cannot alter the state of any past lap."""
    meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df = _create_synthetic_multi_table_race()

    # Original state at lap 15
    state_orig = _build_from_dfs(
        race_id="2023_test",
        lap=15,
        meta=meta,
        all_laps=laps_df,
        all_tyres=tyres_df,
        all_pit_stops=pit_stops_df,
        all_weather=weather_df,
        all_race_control=race_control_df,
    )

    # Mutate future laps (laps 16+) severely
    laps_mutated = laps_df.copy()
    future_mask = laps_mutated["lap_number"] > 15
    laps_mutated.loc[future_mask, "lap_time"] = 45.0  # Impossible lap time
    laps_mutated.loc[future_mask, "track_status"] = "4"  # Safety car in future

    tyres_mutated = tyres_df.copy()
    tyres_mutated.loc[tyres_mutated["lap_number"] > 15, "compound"] = "WET"

    weather_mutated = weather_df.copy()
    weather_mutated.loc[weather_mutated["lap"] > 15, "is_wet"] = True
    weather_mutated.loc[weather_mutated["lap"] > 15, "track_temp"] = 5.0

    state_after_mutation = _build_from_dfs(
        race_id="2023_test",
        lap=15,
        meta=meta,
        all_laps=laps_mutated,
        all_tyres=tyres_mutated,
        all_pit_stops=pit_stops_df,
        all_weather=weather_mutated,
        all_race_control=race_control_df,
    )

    # State dictionaries at lap 15 must be 100% identical
    assert state_orig.to_dict() == state_after_mutation.to_dict()
    assert state_after_mutation.weather["is_wet"] is False
    assert state_after_mutation.safety_car is False


def test_pit_stop_timeline_leakage():
    meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df = _create_synthetic_multi_table_race()

    # D00 pits on lap 12, D01 pits on lap 22
    state_lap_11 = _build_from_dfs("2023_test", 11, meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df)
    state_lap_12 = _build_from_dfs("2023_test", 12, meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df)
    state_lap_21 = _build_from_dfs("2023_test", 21, meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df)

    # Lap 11: 0 pit stops completed
    assert state_lap_11.pit_stops_count["D00"] == 0
    assert len(state_lap_11.pit_stops_history) == 0

    # Lap 12: D00 pit stop visible, D01 not visible
    assert state_lap_12.pit_stops_count["D00"] == 1
    assert len(state_lap_12.pit_stops_history) == 1
    assert state_lap_12.pit_stops_history[0]["driver"] == "D00"

    # Lap 21: D00 pit stop visible, D01 still 0
    assert state_lap_21.pit_stops_count["D01"] == 0
    assert len(state_lap_21.pit_stops_history) == 1


def test_safety_car_and_weather_timeline():
    meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df = _create_synthetic_multi_table_race()

    # Rain starts lap 21, SC deployed lap 25
    state_dry = _build_from_dfs("2023_test", 20, meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df)
    assert state_dry.weather["is_wet"] is False
    assert state_dry.safety_car is False
    assert len(state_dry.race_control_events) == 0

    state_wet = _build_from_dfs("2023_test", 22, meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df)
    assert state_wet.weather["is_wet"] is True
    assert state_wet.safety_car is False
    assert len(state_wet.race_control_events) == 0

    state_sc = _build_from_dfs("2023_test", 25, meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df)
    assert len(state_sc.race_control_events) == 1
    assert state_sc.race_control_events[0]["event_type"] == "SAFETY_CAR"


def test_fastest_lap_no_leakage():
    meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df = _create_synthetic_multi_table_race()

    # Artificially set a lap 10 time to 85.0s, and lap 28 time to 80.0s
    laps_mod = laps_df.copy()
    idx_10 = laps_mod[(laps_mod["driver"] == "D02") & (laps_mod["lap_number"] == 10)].index
    idx_28 = laps_mod[(laps_mod["driver"] == "D03") & (laps_mod["lap_number"] == 28)].index

    laps_mod.loc[idx_10, "lap_time"] = 85.0
    laps_mod.loc[idx_28, "lap_time"] = 80.0

    state_at_15 = _build_from_dfs("2023_test", 15, meta, laps_mod, tyres_df, pit_stops_df, weather_df, race_control_df)
    state_at_29 = _build_from_dfs("2023_test", 29, meta, laps_mod, tyres_df, pit_stops_df, weather_df, race_control_df)

    # At lap 15, D02's lap 10 is fastest (D03's lap 28 is future)
    assert state_at_15.fastest_lap["driver"] == "D02"
    assert state_at_15.fastest_lap["lap"] == 10
    assert state_at_15.fastest_lap["lap_time"] == 85.0

    # At lap 29, D03's lap 28 is now visible and fastest
    assert state_at_29.fastest_lap["driver"] == "D03"
    assert state_at_29.fastest_lap["lap"] == 28
    assert state_at_29.fastest_lap["lap_time"] == 80.0


def test_retirement_tracking():
    meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df = _create_synthetic_multi_table_race()

    # Drop driver D05 after lap 8 (simulating retirement on lap 8)
    laps_ret = laps_df[~((laps_df["driver"] == "D05") & (laps_df["lap_number"] > 8))].copy()

    state_before_ret = _build_from_dfs("2023_test", 8, meta, laps_ret, tyres_df, pit_stops_df, weather_df, race_control_df)
    state_after_ret = _build_from_dfs("2023_test", 14, meta, laps_ret, tyres_df, pit_stops_df, weather_df, race_control_df)

    assert "D05" in state_before_ret.positions
    assert "D05" not in state_before_ret.retired

    assert "D05" not in state_after_ret.positions
    assert "D05" in state_after_ret.retired
    assert "Retired (Lap 8)" in state_after_ret.retired["D05"]


def test_replay_session_seek_and_step():
    meta, laps_df, tyres_df, pit_stops_df, weather_df, race_control_df = _create_synthetic_multi_table_race()

    session = ReplaySession("2023_test")
    # Inject tables directly into session
    session.meta = meta
    session.total_laps = meta["total_laps"]
    session.laps_df = laps_df
    session.tyres_df = tyres_df
    session.pit_stops_df = pit_stops_df
    session.weather_df = weather_df
    session.race_control_df = race_control_df

    st_5 = session.seek(5)
    assert st_5.lap == 5

    st_6 = session.step()
    assert st_6.lap == 6

    st_5_back = session.prev()
    assert st_5_back.lap == 5

    # Generator
    gen_laps = list(session.iter_laps(start_lap=1, end_lap=4))
    assert len(gen_laps) == 4
    assert [s.lap for s in gen_laps] == [1, 2, 3, 4]
