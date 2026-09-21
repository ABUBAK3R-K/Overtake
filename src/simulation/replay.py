"""Race Replay Engine (Design.md Section 6.6, PRD FR-4).

Reconstructs race state lap-by-lap from historical Parquet/DuckDB data,
strictly respecting the as_of_lap() no-leakage guard at every step.

Key Capabilities:
- build_state_at_lap(race_id, lap): Reconstructs full RaceState at completion of `lap`.
- ReplaySession(race_id): Stateful, fast in-memory replay session with seek/step/iter
  for instant scrubbing without repeated database scans.
- Strict No-Leakage: All lookups route through as_of_lap() and assert_as_of_lap().
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Iterator
import numpy as np
import pandas as pd

from src.ingestion.session import load_race_list
from src.ingestion.storage import connect
from src.preprocessing.leakage import as_of_lap, assert_as_of_lap
from src.simulation.state import RaceState

log = logging.getLogger("overtake.simulation.replay")


def get_race_metadata(race_id: str, con=None) -> dict[str, Any]:
    """Fetch race metadata for total_laps and circuit with safe fallback."""
    if con is None:
        con = connect()

    tables = {row[0] for row in con.sql("SHOW TABLES").fetchall()}
    if "races" in tables:
        df = con.sql("SELECT * FROM races WHERE race_id = $race_id", params={"race_id": race_id}).df()
        if not df.empty:
            row = df.iloc[0]
            return {
                "circuit": str(row.get("circuit", "Unknown")),
                "total_laps": int(row.get("total_laps", 57)),
            }

    # Fallback to configured race list
    try:
        for r in load_race_list():
            if r.get("race_id") == race_id:
                return {
                    "circuit": str(r.get("circuit", "Unknown")),
                    "total_laps": int(r.get("total_laps", 57)),
                }
    except Exception:
        pass

    return {"circuit": "Unknown", "total_laps": 57}


def _build_from_dfs(
    race_id: str,
    lap: int,
    meta: dict[str, Any],
    all_laps: pd.DataFrame,
    all_tyres: pd.DataFrame,
    all_pit_stops: pd.DataFrame,
    all_weather: pd.DataFrame,
    all_race_control: pd.DataFrame,
) -> RaceState:
    """Core reconstruction function strictly filtered via as_of_lap."""
    circuit = meta.get("circuit", "Unknown")
    total_laps = meta.get("total_laps", 57)

    if all_laps.empty:
        return RaceState(
            race_id=race_id,
            lap=lap,
            positions={},
            gaps={},
            intervals={},
            tyres={},
            safety_car=False,
            total_laps=total_laps,
            circuit=circuit,
        )

    # 1. Laps filtered strictly as of lap
    laps_as_of = as_of_lap(all_laps, "lap_number", lap)
    assert_as_of_lap(laps_as_of, "lap_number", lap)

    # Slice for current lap
    current_laps = laps_as_of[laps_as_of["lap_number"] == lap].copy()
    if current_laps.empty and lap > 1 and not laps_as_of.empty:
        latest_lap = int(laps_as_of["lap_number"].max())
        current_laps = laps_as_of[laps_as_of["lap_number"] == latest_lap].copy()

    # 2. Positions, gaps to leader, and intervals to car ahead
    positions: dict[str, int] = {}
    gaps: dict[str, float] = {}
    intervals: dict[str, float] = {}
    last_lap_times: dict[str, float] = {}

    if not current_laps.empty:
        # Sort by position (1st, 2nd, ...)
        current_laps = current_laps.sort_values("position")
        prev_gap = 0.0

        for idx, (_, row) in enumerate(current_laps.iterrows()):
            driver = str(row["driver"])
            pos = int(row["position"]) if pd.notna(row.get("position")) else idx + 1
            gap = float(row["gap_to_leader"]) if pd.notna(row.get("gap_to_leader")) else 0.0

            positions[driver] = pos
            gaps[driver] = round(gap, 3)

            # Interval: 0.0 for leader, delta to previous position for others
            if idx == 0:
                intervals[driver] = 0.0
            else:
                intervals[driver] = round(max(0.0, gap - prev_gap), 3)
            prev_gap = gap

            lt = row.get("lap_time")
            if pd.notna(lt) and lt is not None:
                last_lap_times[driver] = round(float(lt), 3)

    # 3. Overall fastest lap as of current lap
    fastest_lap: dict[str, Any] = {}
    if not laps_as_of.empty:
        valid_laps = laps_as_of[
            laps_as_of["lap_time"].notna()
            & (laps_as_of["lap_time"] > 0)
            & ~laps_as_of["pit_flag"].astype(bool)
        ]
        if not valid_laps.empty:
            best_idx = valid_laps["lap_time"].idxmin()
            best_row = valid_laps.loc[best_idx]
            fastest_lap = {
                "driver": str(best_row["driver"]),
                "lap": int(best_row["lap_number"]),
                "lap_time": round(float(best_row["lap_time"]), 3),
            }

    # 4. Retired drivers up to this lap
    retired: dict[str, str] = {}
    if not laps_as_of.empty:
        seen_drivers = set(laps_as_of["driver"].unique())
        active_drivers = set(positions.keys())
        missing = seen_drivers - active_drivers
        for d in sorted(missing):
            d_laps = laps_as_of[laps_as_of["driver"] == d]
            last_l = int(d_laps["lap_number"].max())
            retired[d] = f"Retired (Lap {last_l})"

    # 5. Tyres strictly as of lap
    tyres: dict[str, tuple[str, int]] = {}
    if not all_tyres.empty:
        tyres_as_of = as_of_lap(all_tyres, "lap_number", lap)
        assert_as_of_lap(tyres_as_of, "lap_number", lap)
        cur_tyres = tyres_as_of[tyres_as_of["lap_number"] == lap]
        for _, row in cur_tyres.iterrows():
            d = str(row["driver"])
            comp = str(row.get("compound", "MEDIUM")).upper()
            age = int(row.get("tyre_age_at_lap", 1)) if pd.notna(row.get("tyre_age_at_lap")) else 1
            tyres[d] = (comp, age)

    for d in positions:
        if d not in tyres:
            tyres[d] = ("MEDIUM", lap)

    # 6. Pit stops strictly as of lap
    pit_stops_count: dict[str, int] = {d: 0 for d in positions}
    pit_stops_history: list[dict[str, Any]] = []

    if not all_pit_stops.empty:
        pit_col = "lap" if "lap" in all_pit_stops else "lap_number"
        pit_as_of = as_of_lap(all_pit_stops, pit_col, lap)
        assert_as_of_lap(pit_as_of, pit_col, lap)
        for _, row in pit_as_of.iterrows():
            d = str(row["driver"])
            pit_stops_count[d] = pit_stops_count.get(d, 0) + 1
            pit_stops_history.append({
                "driver": d,
                "lap": int(row[pit_col]),
                "duration": round(float(row.get("pit_duration", 22.0)), 2) if pd.notna(row.get("pit_duration")) else None,
                "compound_before": str(row.get("compound_before", "")),
                "compound_after": str(row.get("compound_after", "")),
            })
    else:
        # Fallback to pit_flag in laps_as_of
        flagged = laps_as_of[laps_as_of["pit_flag"].astype(bool)]
        for _, row in flagged.iterrows():
            d = str(row["driver"])
            pit_stops_count[d] = pit_stops_count.get(d, 0) + 1
            pit_stops_history.append({
                "driver": d,
                "lap": int(row["lap_number"]),
            })

    # 7. Safety Car & Track Status
    safety_car = False
    status = "RACING"
    if not current_laps.empty:
        statuses = current_laps["track_status"].dropna().astype(str)
        if any("4" in s for s in statuses):
            safety_car = True
            status = "SAFETY_CAR"
        elif any("6" in s or "7" in s for s in statuses):
            safety_car = True
            status = "VIRTUAL_SAFETY_CAR"
        elif any("5" in s for s in statuses):
            status = "RED_FLAG"

    # 8. Race control events strictly as of lap
    race_control_events: list[dict[str, Any]] = []
    if not all_race_control.empty:
        rc_col = "lap" if "lap" in all_race_control else "lap_number"
        rc_as_of = as_of_lap(all_race_control, rc_col, lap)
        assert_as_of_lap(rc_as_of, rc_col, lap)
        for _, row in rc_as_of.iterrows():
            race_control_events.append({
                "lap": int(row[rc_col]),
                "event_type": str(row.get("event_type", "")),
                "message": str(row.get("message", "")),
                "category": str(row.get("category", "")),
            })

    # 9. Weather strictly as of lap
    weather = {"track_temp": 30.0, "air_temp": 25.0, "is_wet": False}
    if not all_weather.empty:
        w_col = "lap" if "lap" in all_weather else "lap_number"
        w_as_of = as_of_lap(all_weather, w_col, lap)
        assert_as_of_lap(w_as_of, w_col, lap)
        cur_w = w_as_of[w_as_of[w_col] == lap]
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
        intervals=intervals,
        tyres=tyres,
        safety_car=safety_car,
        total_laps=total_laps,
        circuit=circuit,
        status=status,
        weather=weather,
        last_lap_times=last_lap_times,
        pit_stops_count=pit_stops_count,
        retired=retired,
        fastest_lap=fastest_lap,
        pit_stops_history=pit_stops_history,
        race_control_events=race_control_events,
    )


def build_state_at_lap(race_id: str, lap: int, con=None) -> RaceState:
    """Reconstruct the RaceState at the completion of `lap` with no leakage."""
    if con is None:
        con = connect()

    meta = get_race_metadata(race_id, con=con)
    tables = {row[0] for row in con.sql("SHOW TABLES").fetchall()}

    def _get_df(table_name: str) -> pd.DataFrame:
        if table_name in tables:
            try:
                return con.sql(f"SELECT * FROM {table_name} WHERE race_id = $race_id", params={"race_id": race_id}).df()
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    all_laps = _get_df("laps")
    all_tyres = _get_df("tyres")
    all_pit_stops = _get_df("pit_stops")
    all_weather = _get_df("weather")
    all_race_control = _get_df("race_control")

    return _build_from_dfs(
        race_id=race_id,
        lap=lap,
        meta=meta,
        all_laps=all_laps,
        all_tyres=all_tyres,
        all_pit_stops=all_pit_stops,
        all_weather=all_weather,
        all_race_control=all_race_control,
    )


class ReplaySession:
    """In-memory, leakage-safe race replay session for high-speed scrubbing."""

    def __init__(self, race_id: str, con=None):
        self.race_id = race_id
        if con is None:
            con = connect()
        self.con = con
        self.meta = get_race_metadata(race_id, con=con)
        self.total_laps = self.meta.get("total_laps", 57)
        self.circuit = self.meta.get("circuit", "Unknown")
        self.current_lap = 1

        tables = {row[0] for row in con.sql("SHOW TABLES").fetchall()}
        self.laps_df = con.sql("SELECT * FROM laps WHERE race_id = $race_id", params={"race_id": race_id}).df() if "laps" in tables else pd.DataFrame()
        self.tyres_df = con.sql("SELECT * FROM tyres WHERE race_id = $race_id", params={"race_id": race_id}).df() if "tyres" in tables else pd.DataFrame()
        self.pit_stops_df = con.sql("SELECT * FROM pit_stops WHERE race_id = $race_id", params={"race_id": race_id}).df() if "pit_stops" in tables else pd.DataFrame()
        self.weather_df = con.sql("SELECT * FROM weather WHERE race_id = $race_id", params={"race_id": race_id}).df() if "weather" in tables else pd.DataFrame()
        self.race_control_df = con.sql("SELECT * FROM race_control WHERE race_id = $race_id", params={"race_id": race_id}).df() if "race_control" in tables else pd.DataFrame()

        self._cache: dict[int, RaceState] = {}

    def seek(self, lap: int) -> RaceState:
        """Seek directly to lap and return RaceState with zero leakage."""
        lap = max(1, min(lap, self.total_laps))
        self.current_lap = lap
        if lap not in self._cache:
            self._cache[lap] = _build_from_dfs(
                race_id=self.race_id,
                lap=lap,
                meta=self.meta,
                all_laps=self.laps_df,
                all_tyres=self.tyres_df,
                all_pit_stops=self.pit_stops_df,
                all_weather=self.weather_df,
                all_race_control=self.race_control_df,
            )
        return self._cache[lap]

    def step(self) -> RaceState:
        """Advance replay by one lap."""
        if self.current_lap < self.total_laps:
            self.current_lap += 1
        return self.seek(self.current_lap)

    def prev(self) -> RaceState:
        """Step back replay by one lap."""
        if self.current_lap > 1:
            self.current_lap -= 1
        return self.seek(self.current_lap)

    def iter_laps(self, start_lap: int = 1, end_lap: int | None = None) -> Iterator[RaceState]:
        """Generator yielding RaceState lap-by-lap."""
        target_end = min(end_lap or self.total_laps, self.total_laps)
        for l in range(start_lap, target_end + 1):
            yield self.seek(l)


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

    if driver in state.tyres:
        comp, age = state.tyres[driver]
        state.tyres[driver] = (comp, age + 1)

    return state


def advance_lap_with_predictions(
    state: RaceState,
    predict_lap_time_fn=None,
    predict_degradation_fn=None,
) -> RaceState:
    """Advance the state by one lap using model predictions."""
    if state.lap >= state.total_laps:
        state.status = "FINISHED"
        return state

    new_state = copy.deepcopy(state)
    new_state.lap += 1

    accumulated_times = {}
    for driver in list(new_state.positions.keys()):
        if predict_lap_time_fn is not None:
            pace = float(predict_lap_time_fn(new_state, driver))
        else:
            pace = new_state.last_lap_times.get(driver, 90.0)

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

    min_time = min(accumulated_times.values()) if accumulated_times else 0.0
    sorted_drivers = sorted(accumulated_times.keys(), key=lambda d: accumulated_times[d])

    new_positions = {}
    new_gaps = {}
    new_intervals = {}
    prev_g = 0.0
    for rank, d in enumerate(sorted_drivers, start=1):
        g = round(accumulated_times[d] - min_time, 2)
        new_positions[d] = rank
        new_gaps[d] = g
        new_intervals[d] = round(g - prev_g, 2) if rank > 1 else 0.0
        prev_g = g

    new_state.positions = new_positions
    new_state.gaps = new_gaps
    new_state.intervals = new_intervals
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
    session = ReplaySession(race_id, con=con)
    return list(session.iter_laps(start_lap=start_lap, end_lap=end_lap))
