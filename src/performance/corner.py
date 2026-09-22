"""Corner/mini-sector driver-performance module (Design.md Section 6.11 / PRD FR-9).

Segments a lap's raw telemetry into corner windows (using FastF1's circuit
corner markers) and compares one driver's lap against a benchmark lap,
corner by corner, to estimate time gained or lost in each one.

This is a read-only, retrospective analysis feature, not part of the
prediction/decision pipeline — it does not go through as_of_lap(). The
benchmark lap can legitimately be any lap in the session (e.g. the race's
fastest lap), before or after the lap being analysed, the same way a real
telemetry-overlay tool works; there is no "future" to leak into a decision
here because no decision is being made.

Note this operates on raw per-sample FastF1 telemetry (Distance/Speed/
Throttle/Brake/nGear), not the ingested `telemetry` Parquet table — that
table only stores per-lap summaries (mean/max speed etc., see
src/ingestion/telemetry.py), which aren't enough to segment by corner. A
FastF1 session is loaded on demand and cached in-process instead of adding a
new per-corner Parquet table for all 112 races, since only a handful of
laps are ever inspected at once here (see PROJECT_STATUS.md Phase 8 notes).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd

from src.ingestion.session import load_race_list, load_race_session

log = logging.getLogger("overtake.performance.corner")

CORNER_TELEMETRY_COLUMNS = [
    "corner_number", "corner_label", "min_speed", "throttle_pct",
    "brake_point", "gear", "segment_time_s", "window_start_m", "window_end_m",
]


def _season_round_for_race(race_id: str) -> tuple[int, int]:
    for r in load_race_list():
        if r["race_id"] == race_id:
            return int(r["season"]), int(r["round"])
    raise ValueError(f"unknown race_id {race_id!r}; not in configs/races.toml")


@lru_cache(maxsize=8)
def _cached_session(race_id: str):
    """Load and cache a full FastF1 session in-process.

    Loading one session — even fully cached locally by FastF1 — takes tens
    of seconds with telemetry enabled, far more than the PRD's "few seconds"
    target, so repeat requests for the same race must not re-load it.
    Bounded to 8 races so a long-running server doesn't grow this
    unboundedly; each loaded session (with telemetry for ~20 drivers) is not
    small.
    """
    season, round_number = _season_round_for_race(race_id)
    log.info("loading FastF1 session for %s (season=%d round=%d) — this can take a while", race_id, season, round_number)
    return load_race_session(season, round_number, telemetry=True)


def _corner_windows(
    distances: list[float], lap_length: float,
    margin_frac: float = 0.35, max_margin: float = 120.0,
) -> list[tuple[float, float]]:
    """One [start, end) window per corner, sized as a fraction of the gap to
    its neighbours (capped at ``max_margin`` metres) so a window covers the
    braking zone and exit without swallowing the whole straight between
    corners — that would blur cornering performance with straight-line pace,
    which are different things. Windows are clipped to [0, lap_length]; the
    very first/last corner may lose a little entry/exit margin as a result
    (there's no lap before/after this one to borrow it from).
    """
    n = len(distances)
    windows = []
    for i, d in enumerate(distances):
        prev_d = distances[i - 1] if i > 0 else distances[-1] - lap_length
        next_d = distances[i + 1] if i < n - 1 else distances[0] + lap_length
        entry = min(max_margin, margin_frac * (d - prev_d))
        exit_ = min(max_margin, margin_frac * (next_d - d))
        windows.append((max(0.0, d - entry), min(lap_length, d + exit_)))
    return windows


def get_corner_boundaries(session) -> pd.DataFrame:
    """Corner apex distances for this session's circuit, sorted by distance,
    with a sequential ``corner_number`` (1..N) used as the join key between a
    driver's and a benchmark's CornerTelemetry — their FastF1 labels (e.g.
    "3A") are otherwise awkward to join on directly.
    """
    corners = session.get_circuit_info().corners.sort_values("Distance").reset_index(drop=True)
    corners = corners.assign(corner_number=corners.index + 1)
    corners["corner_label"] = corners["Number"].astype(str) + corners["Letter"].fillna("")
    return corners[["corner_number", "corner_label", "Distance"]].rename(columns={"Distance": "distance"})


def build_corner_telemetry(session, driver: str, lap_number: int) -> pd.DataFrame:
    """One row per corner for this driver's lap — CornerTelemetry per
    Design.md Section 4, plus ``segment_time_s``/``window_*`` used by
    ``analyze_corner_performance`` below to compute time lost per corner.
    """
    laps = session.laps.pick_drivers([driver]).pick_laps([lap_number])
    if laps.empty:
        raise ValueError(f"no lap {lap_number} for driver {driver} in this session")
    lap = laps.iloc[0]
    car = lap.get_car_data().add_distance()
    car = car.assign(t=car["Time"].dt.total_seconds())
    lap_length = float(car["Distance"].max())
    if lap_length <= 0:
        raise ValueError(f"no telemetry distance for {driver} lap {lap_number}")

    corners = get_corner_boundaries(session)
    windows = _corner_windows(corners["distance"].tolist(), lap_length)

    rows = []
    for (_, corner), (start, end) in zip(corners.iterrows(), windows):
        seg = car[(car["Distance"] >= start) & (car["Distance"] < end)]
        if seg.empty:
            continue
        brake_on = seg[seg["Brake"]]
        brake_point = float(brake_on["Distance"].iloc[0] - start) if not brake_on.empty else None
        gear_mode = seg["nGear"].mode()
        rows.append({
            "corner_number": int(corner["corner_number"]),
            "corner_label": corner["corner_label"],
            "min_speed": float(seg["Speed"].min()),
            "throttle_pct": float(seg["Throttle"].mean()),
            "brake_point": brake_point,
            "gear": int(gear_mode.iloc[0]) if not gear_mode.empty else None,
            "segment_time_s": float(seg["t"].iloc[-1] - seg["t"].iloc[0]),
            "window_start_m": round(start, 1),
            "window_end_m": round(end, 1),
        })
    return pd.DataFrame(rows, columns=CORNER_TELEMETRY_COLUMNS)


def analyze_corner_performance(
    driver_telemetry: pd.DataFrame, benchmark_telemetry: pd.DataFrame,
) -> dict[str, Any]:
    """Per-corner time-loss breakdown vs. a benchmark lap (Design.md Section 5
    locked signature). Both arguments are CornerTelemetry-shaped frames from
    ``build_corner_telemetry``, aligned on ``corner_number``.

    ``time_lost_s`` > 0 means the driver's lap spent longer in that corner
    window than the benchmark; < 0 means they gained time there. Summed
    across corners this does not equal the full lap-time delta (it excludes
    the parts of the straights outside every corner's window by design —
    see ``_corner_windows``), but it isolates cornering performance from
    straight-line pace, which is the point of the feature.
    """
    empty = {"corners": [], "total_time_lost_s": 0.0, "worst_corner": None, "best_corner": None, "n_corners": 0}
    if driver_telemetry.empty or benchmark_telemetry.empty:
        return empty

    merged = driver_telemetry.merge(
        benchmark_telemetry, on="corner_number", suffixes=("_driver", "_benchmark"),
    )
    if merged.empty:
        return empty

    merged["time_lost_s"] = merged["segment_time_s_driver"] - merged["segment_time_s_benchmark"]
    merged["min_speed_delta_kmh"] = merged["min_speed_driver"] - merged["min_speed_benchmark"]
    merged["throttle_delta_pct"] = merged["throttle_pct_driver"] - merged["throttle_pct_benchmark"]
    merged["brake_point_delta_m"] = merged["brake_point_driver"] - merged["brake_point_benchmark"]

    corners = [
        {
            "corner_number": int(row["corner_number"]),
            "corner_label": row["corner_label_driver"],
            "time_lost_s": round(float(row["time_lost_s"]), 3),
            "min_speed_delta_kmh": round(float(row["min_speed_delta_kmh"]), 1),
            "throttle_delta_pct": round(float(row["throttle_delta_pct"]), 1),
            "brake_point_delta_m": (
                round(float(row["brake_point_delta_m"]), 1) if pd.notna(row["brake_point_delta_m"]) else None
            ),
            "gear_driver": (int(row["gear_driver"]) if pd.notna(row["gear_driver"]) else None),
            "gear_benchmark": (int(row["gear_benchmark"]) if pd.notna(row["gear_benchmark"]) else None),
        }
        for _, row in merged.iterrows()
    ]
    total = round(float(merged["time_lost_s"].sum()), 3)
    worst = max(corners, key=lambda c: c["time_lost_s"])
    best = min(corners, key=lambda c: c["time_lost_s"])
    return {
        "corners": corners, "total_time_lost_s": total,
        "worst_corner": worst, "best_corner": best, "n_corners": len(corners),
    }


def analyze_driver_lap_vs_benchmark(
    race_id: str, driver: str, lap_number: int,
    benchmark_driver: str | None = None, benchmark_lap: int | None = None,
) -> dict[str, Any]:
    """Handoff used by ``GET /api/corner-analysis/{race_id}/{driver}/{lap}``
    (Design.md Section 6.13). Defaults the benchmark to the session's overall
    fastest lap when not given.
    """
    session = _cached_session(race_id)
    driver_ct = build_corner_telemetry(session, driver, lap_number)

    if benchmark_driver is None or benchmark_lap is None:
        fastest = session.laps.pick_fastest()
        benchmark_driver = benchmark_driver or str(fastest["Driver"])
        benchmark_lap = benchmark_lap or int(fastest["LapNumber"])
    benchmark_ct = build_corner_telemetry(session, benchmark_driver, benchmark_lap)

    result = analyze_corner_performance(driver_ct, benchmark_ct)
    result.update({
        "race_id": race_id, "driver": driver, "lap": lap_number,
        "benchmark_driver": benchmark_driver, "benchmark_lap": benchmark_lap,
    })
    return result
