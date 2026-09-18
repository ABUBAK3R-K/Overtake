"""Race Replay Engine (Design.md Section 6.5).

Reconstructs race state lap-by-lap from historical Parquet/DuckDB data,
strictly respecting the as_of_lap() no-leakage guard.
"""

from __future__ import annotations

import logging
from typing import Any
import pandas as pd

from src.ingestion.storage import connect
from src.preprocessing.leakage import as_of_lap, assert_as_of_lap
from src.simulation.state import RaceState

log = logging.getLogger("overtake.simulation.replay")


def get_race_metadata(race_id: str, con=None) -> dict[str, Any]:
    """Fetch race metadata for total_laps and circuit."""
    if con is None:
        con = connect()
    df = con.sql("SELECT * FROM races WHERE race_id = $race_id", params={"race_id": race_id}).df()
    if df.empty:
        return {"circuit": "Unknown", "total_laps": 57}
    row = df.iloc[0]
    return {
        "circuit": str(row.get("circuit", "Unknown")),
        "total_laps": int(row.get("total_laps", 57)),
    }


def build_state_at_lap(race_id: str, lap: int, con=None) -> RaceState:
    """Reconstruct the RaceState at the completion of `lap` with no leakage."""
    if con is None:
        con = connect()

    # Load race info
    meta = get_race_metadata(race_id, con=con)
    circuit = meta["circuit"]
    total_laps = meta["total_laps"]

    # Load laps strictly up to current lap
    all_laps = con.sql("SELECT * FROM laps WHERE race_id = $race_id", params={"race_id": race_id}).df()
    if all_laps.empty:
        return RaceState(
            race_id=race_id, lap=lap, positions={}, gaps={}, tyres={},
            safety_car=False, total_laps=total_laps, circuit=circuit,
        )

    laps_as_of = as_of_lap(all_laps, "lap_number", lap)
    assert_as_of_lap(laps_as_of, "lap_number", lap)

    # Current lap slice
    current_laps = laps_as_of[laps_as_of["lap_number"] == lap].copy()
    if current_laps.empty and lap > 1:
        # Fallback to the latest available lap <= lap
        latest_lap = int(laps_as_of["lap_number"].max())
        current_laps = laps_as_of[laps_as_of["lap_number"] == latest_lap].copy()

    # Positions and gaps
    positions = {}
    gaps = {}
    last_lap_times = {}
    for _, row in current_laps.iterrows():
        driver = str(row["driver"])
        pos = row["position"]
        positions[driver] = int(pos) if pd.notna(pos) else 20
        gap = row["gap_to_leader"]
        gaps[driver] = float(gap) if pd.notna(gap) else 0.0
        lt = row["lap_time"]
        if pd.notna(lt):
            last_lap_times[driver] = float(lt)

    # Tyres
    all_tyres = con.sql("SELECT * FROM tyres WHERE race_id = $race_id", params={"race_id": race_id}).df()
    tyres = {}
    if not all_tyres.empty:
        tyres_as_of = as_of_lap(all_tyres, "lap_number", lap)
        assert_as_of_lap(tyres_as_of, "lap_number", lap)
        cur_tyres = tyres_as_of[tyres_as_of["lap_number"] == lap]
        for _, row in cur_tyres.iterrows():
            d = str(row["driver"])
            comp = str(row.get("compound", "MEDIUM"))
            age = int(row.get("tyre_age_at_lap", 1)) if pd.notna(row.get("tyre_age_at_lap")) else 1
            tyres[d] = (comp, age)

    # For any driver missing in tyres, default to MEDIUM, age=lap
    for d in positions:
        if d not in tyres:
            tyres[d] = ("MEDIUM", lap)

    # Pit stops count
    pit_stops_count = {}
    pit_stops = laps_as_of[laps_as_of["pit_flag"] == True]
    if not pit_stops.empty:
        counts = pit_stops.groupby("driver")["lap_number"].count()
        pit_stops_count = {str(d): int(c) for d, c in counts.items()}
    for d in positions:
        pit_stops_count.setdefault(d, 0)

    # Safety car status on current lap
    safety_car = False
    status = "RACING"
    if not current_laps.empty:
        statuses = current_laps["track_status"].dropna().astype(str)
        # FastF1 track statuses: 4 = SC, 5 = Red Flag, 6/7 = VSC
        if any("4" in s for s in statuses):
            safety_car = True
            status = "SAFETY_CAR"
        elif any("6" in s or "7" in s for s in statuses):
            safety_car = True
            status = "VIRTUAL_SAFETY_CAR"
        elif any("5" in s for s in statuses):
            status = "RED_FLAG"

    # Weather
    weather = {"track_temp": 30.0, "air_temp": 25.0, "is_wet": False}
    tables = {row[0] for row in con.sql("SHOW TABLES").fetchall()}
    if "weather" in tables:
        w_df = con.sql("SELECT * FROM weather WHERE race_id = $race_id", params={"race_id": race_id}).df()
        if not w_df.empty:
            w_as_of = as_of_lap(w_df, "lap", lap)
            cur_w = w_as_of[w_as_of["lap"] == lap]
            if not cur_w.empty:
                row = cur_w.iloc[0]
                weather = {
                    "track_temp": float(row.get("track_temp", 30.0)),
                    "air_temp": float(row.get("air_temp", 25.0)),
                    "is_wet": bool(row.get("is_wet", False)),
                }

    return RaceState(
        race_id=race_id,
        lap=lap,
        positions=positions,
        gaps=gaps,
        tyres=tyres,
        safety_car=safety_car,
        total_laps=total_laps,
        circuit=circuit,
        status=status,
        weather=weather,
        last_lap_times=last_lap_times,
        pit_stops_count=pit_stops_count,
    )


def init_state(race_id: str, lap: int = 1, con=None) -> RaceState:
    """Initialize race state at the given lap (default: lap 1)."""
    return build_state_at_lap(race_id, lap, con=con)


def apply_pace(
    state: RaceState,
    driver: str,
    pace: float,
    degradation: float = 0.0,
) -> RaceState:
    """Apply predicted pace and tyre degradation delta to a driver in RaceState."""
    effective_lap_time = pace + degradation
    state.last_lap_times[driver] = round(effective_lap_time, 3)

    # Increment tyre age
    if driver in state.tyres:
        comp, age = state.tyres[driver]
        state.tyres[driver] = (comp, age + 1)

    return state


def advance_lap_with_predictions(
    state: RaceState,
    predict_lap_time_fn=None,
    predict_degradation_fn=None,
) -> RaceState:
    """Advance the state by one lap using model predictions (Sprint Plan Day 6).

    Iterates over active drivers, predicts next lap pace and degradation,
    and updates the race state accordingly.
    """
    if state.lap >= state.total_laps:
        state.status = "FINISHED"
        return state

    import copy
    new_state = copy.deepcopy(state)
    new_state.lap += 1

    accumulated_times = {}
    for driver in list(new_state.positions.keys()):
        # 1. Predict pace
        if predict_lap_time_fn is not None:
            pace = float(predict_lap_time_fn(new_state, driver))
        else:
            pace = new_state.last_lap_times.get(driver, 90.0)

        # 2. Predict degradation
        comp, age = new_state.tyres.get(driver, ("MEDIUM", 1))
        if predict_degradation_fn is not None:
            try:
                degradation = float(predict_degradation_fn(comp, age, new_state.circuit, new_state.weather.get("track_temp", 30.0)))
            except Exception:
                degradation = 0.0
        else:
            degradation = 0.0

        apply_pace(new_state, driver, pace, degradation)
        current_gap = new_state.gaps.get(driver, 0.0)
        accumulated_times[driver] = current_gap + (pace + degradation)

    # Update relative positions and gaps
    min_time = min(accumulated_times.values()) if accumulated_times else 0.0
    sorted_drivers = sorted(accumulated_times.keys(), key=lambda d: accumulated_times[d])

    new_positions = {}
    new_gaps = {}
    for rank, d in enumerate(sorted_drivers, start=1):
        new_positions[d] = rank
        new_gaps[d] = round(accumulated_times[d] - min_time, 2)

    new_state.positions = new_positions
    new_state.gaps = new_gaps
    return new_state


def advance_lap(
    state: RaceState,
    con=None,
    predict_lap_time_fn=None,
    predict_degradation_fn=None,
) -> RaceState:
    """Advance state to the next lap (historical or predictive)."""
    if predict_lap_time_fn is not None:
        return advance_lap_with_predictions(
            state,
            predict_lap_time_fn=predict_lap_time_fn,
            predict_degradation_fn=predict_degradation_fn,
        )

    if state.lap >= state.total_laps:
        state.status = "FINISHED"
        return state
    return build_state_at_lap(state.race_id, state.lap + 1, con=con)


def run_replay(race_id: str, start_lap: int = 1, end_lap: int | None = None, con=None) -> list[RaceState]:
    """Replay race lap-by-lap from start_lap to end_lap."""
    if con is None:
        con = connect()
    meta = get_race_metadata(race_id, con=con)
    total_laps = meta["total_laps"]
    target_end = min(end_lap or total_laps, total_laps)

    history = []
    for lap in range(start_lap, target_end + 1):
        st = build_state_at_lap(race_id, lap, con=con)
        history.append(st)
    return history

