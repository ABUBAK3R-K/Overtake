"""Monte Carlo Forward Simulation Engine (Design.md Section 6.6).

Simulates multiple forward race trajectories from the current RaceState under uncertainty,
accounting for tyre degradation, pit stop delta, lap-time variance, and Safety Car risk.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from src.models.lap_time import make_lap_time_predictor
from src.models.tyre import predict_tyre_degradation
from src.simulation.safety_car import (
    apply_safety_car_bunching,
    safety_car_probability,
    sample_safety_car_duration,
)
from src.simulation.state import RaceState

log = logging.getLogger("overtake.simulation.monte_carlo")

# Baseline pit lane time loss (seconds) under racing conditions vs under Safety Car
PIT_LOSS_RACING = 22.0
PIT_LOSS_SAFETY_CAR = 12.5  # Pit delta under SC/VSC is significantly lower

# Compound max useful lifespan estimate before severe cliff
COMPOUND_MAX_LIFESPAN = {
    "SOFT": 22,
    "MEDIUM": 32,
    "HARD": 44,
    "INTERMEDIATE": 25,
    "WET": 20,
}


def _get_driver_base_pace(state: RaceState, driver: str) -> float:
    """Estimate a driver's baseline clean lap pace from recent state."""
    if driver in state.last_lap_times and state.last_lap_times[driver] > 60:
        return state.last_lap_times[driver]
    # Fallback to field median
    valid = [t for t in state.last_lap_times.values() if t > 60]
    return float(np.median(valid)) if valid else 90.0


def run_monte_carlo(
    state: RaceState,
    strategy: dict[str, Any] | None = None,
    target_driver: str = "VER",
    n_sims: int = 500,
    lap_time_predictor=None,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Run N forward stochastic simulations from `state.lap` to `total_laps`.

    Args:
        state: Starting RaceState as of decision lap.
        strategy: Dict containing planned pit stops for target_driver:
            e.g. {'pit_laps': [25], 'compounds': ['HARD']} or empty for stay out.
        target_driver: Driver code being evaluated (e.g. 'VER', 'HAM').
        n_sims: Number of Monte Carlo iterations (default: 500).
        lap_time_predictor: Pre-built LapTimePredictor or None to create one.
        seed: Random seed for reproducibility.

    Returns:
        Summary dict containing finish_prob_by_position, expected_position,
        expected_time, podium_prob, win_prob, percentiles, and sample paths.
    """
    if seed is not None:
        np.random.seed(seed)

    remaining_laps = state.total_laps - state.lap
    if remaining_laps <= 0:
        pos = state.positions.get(target_driver, 1)
        return {
            "target_driver": target_driver,
            "finish_prob_by_position": {pos: 1.0},
            "expected_position": float(pos),
            "expected_time": 0.0,
            "podium_prob": 1.0 if pos <= 3 else 0.0,
            "win_prob": 1.0 if pos == 1 else 0.0,
            "percentiles": {"p10": pos, "p50": pos, "p90": pos},
            "sample_trajectories": [[pos]],
        }

    # Parse strategy for target driver
    pit_plan = {}  # lap_number -> compound
    if strategy:
        pit_laps = strategy.get("pit_laps", [])
        compounds = strategy.get("compounds", [])
        for plap, pcomp in zip(pit_laps, compounds):
            if plap > state.lap:
                pit_plan[int(plap)] = str(pcomp).upper()

    # Pre-calculate base pace and tyre wear baseline for all active drivers
    drivers = list(state.positions.keys())
    if target_driver not in drivers:
        drivers.append(target_driver)

    driver_base_paces = {d: _get_driver_base_pace(state, d) for d in drivers}

    # Predictor initialization
    if lap_time_predictor is None:
        try:
            lap_time_predictor = make_lap_time_predictor(state.race_id, state.lap)
        except Exception:
            lap_time_predictor = None

    n_track = min(10, n_sims)  # sample trajectories kept for the response payload

    # Per-simulation trajectories, advanced lap-by-lap together (not sim-by-sim)
    # so predict_many() can be called once per lap across every simulation at
    # once, per its own docstring ("step all sims together"). Calling it once
    # per (sim, lap) instead was measured at several minutes per recommendation.
    sim_tyres = [{d: list(state.tyres.get(d, ("MEDIUM", 1))) for d in drivers} for _ in range(n_sims)]
    sim_gaps = [{d: float(state.gaps.get(d, 0.0)) for d in drivers} for _ in range(n_sims)]
    sim_positions = [dict(state.positions) for _ in range(n_sims)]
    for sp in sim_positions:
        sp.setdefault(target_driver, len(sp) + 1)
    sample_paths: list[list[int]] = [[] for _ in range(n_track)]
    # Track multi-lap Safety Car episodes across simulations
    sc_remaining = np.full(n_sims, 2 if state.safety_car else 0, dtype=int)

    for sim_lap in range(state.lap + 1, state.total_laps + 1):
        lap_frac = sim_lap / state.total_laps
        sc_prob = safety_car_probability(state.circuit, lap_frac)

        is_sc = np.zeros(n_sims, dtype=bool)
        for i in range(n_sims):
            if sc_remaining[i] > 0:
                sc_remaining[i] -= 1
                is_sc[i] = True
            elif np.random.random() < sc_prob:
                dur = sample_safety_car_duration(state.circuit, event_type="SAFETY_CAR")
                sc_remaining[i] = max(0, dur - 1)
                is_sc[i] = True

        # Decide this lap's pit stops and assemble one RaceState per
        # simulation representing state as of the end of sim_lap - 1.
        pit_this_lap: list[dict[str, str]] = []
        temp_states = []
        for i in range(n_sims):
            tyres_i = sim_tyres[i]
            stops: dict[str, str] = {}
            for d in drivers:
                comp, age = tyres_i[d]
                if d == target_driver:
                    if sim_lap in pit_plan:
                        stops[d] = pit_plan[sim_lap]
                    continue
                # Rival dynamic pit logic: pit past the compound's useful life,
                # or opportunistically under a safety car.
                max_life = COMPOUND_MAX_LIFESPAN.get(comp, 30)
                pit_chance = 0.0
                if age >= max_life:
                    pit_chance = 0.65
                elif is_sc[i] and age >= max_life * 0.6:
                    pit_chance = 0.40
                if pit_chance and np.random.random() < pit_chance:
                    stops[d] = "HARD" if comp == "MEDIUM" else "MEDIUM"
            pit_this_lap.append(stops)

            temp_tyres = {d: ((stops[d], 0) if d in stops else tuple(tyres_i[d])) for d in drivers}
            temp_states.append(RaceState(
                race_id=state.race_id, lap=sim_lap - 1, positions=sim_positions[i],
                gaps=sim_gaps[i], tyres=temp_tyres, safety_car=bool(is_sc[i]),
                total_laps=state.total_laps, circuit=state.circuit, weather=state.weather,
            ))

        pace_dicts = None
        if lap_time_predictor is not None:
            try:
                pace_dicts = lap_time_predictor.predict_many(temp_states)
            except Exception:
                log.warning("lap-time model prediction failed at lap %d; falling back", sim_lap, exc_info=True)
                pace_dicts = None

        for i in range(n_sims):
            stops = pit_this_lap[i]
            paces = pace_dicts[i] if pace_dicts is not None else {}
            for d in drivers:
                comp, age = sim_tyres[i][d]
                new_comp, new_age = (stops[d], 0) if d in stops else (comp, age)

                pace = paces.get(d)
                if pace is None or not np.isfinite(pace):
                    # Fallback when the model has no history for this race/lap
                    # (e.g. an unseen or minimally-ingested race in tests).
                    try:
                        wear = predict_tyre_degradation(
                            new_comp, new_age, state.circuit,
                            state.weather.get("track_temp", 30.0),
                        )
                    except Exception:
                        wear = float(new_age) * 0.05
                    pace = driver_base_paces[d] + wear
                    if is_sc[i]:
                        pace *= 1.537  # SC pace delta multiplier; predict_many applies its own

                lap_time = pace + np.random.normal(0.0, 0.35)
                if d in stops:
                    pit_loss = PIT_LOSS_SAFETY_CAR if is_sc[i] else PIT_LOSS_RACING
                    lap_time += pit_loss + np.random.normal(0.0, 0.5)

                sim_gaps[i][d] += lap_time
                sim_tyres[i][d] = [new_comp, new_age + 1]

            min_time = min(sim_gaps[i].values())
            for d in drivers:
                sim_gaps[i][d] -= min_time

            if is_sc[i]:
                sorted_d = sorted(drivers, key=lambda d: sim_gaps[i][d])
                curr_gap = 0.0
                for rank, d in enumerate(sorted_d):
                    if rank == 0:
                        sim_gaps[i][d] = 0.0
                    else:
                        curr_gap += 0.85 + float(np.random.uniform(-0.1, 0.15))
                        sim_gaps[i][d] = round(float(min(sim_gaps[i][d], curr_gap)), 2)

            ranked = sorted(drivers, key=lambda d: sim_gaps[i][d])
            sim_positions[i] = {d: r + 1 for r, d in enumerate(ranked)}
            if i < n_track:
                sample_paths[i].append(sim_positions[i][target_driver])

    finish_positions = [sim_positions[i][target_driver] for i in range(n_sims)]
    finish_times = [sim_gaps[i][target_driver] for i in range(n_sims)]

    # Summarize finishing distributions
    pos_counts: dict[int, int] = {}
    for p in finish_positions:
        pos_counts[p] = pos_counts.get(p, 0) + 1

    finish_prob_by_position = {
        p: round(count / n_sims, 4) for p, count in sorted(pos_counts.items())
    }
    expected_pos = float(np.mean(finish_positions))
    expected_time = float(np.mean(finish_times))
    podium_prob = sum(p for pos, p in finish_prob_by_position.items() if pos <= 3)
    win_prob = finish_prob_by_position.get(1, 0.0)

    p10 = int(np.percentile(finish_positions, 10))
    p50 = int(np.percentile(finish_positions, 50))
    p90 = int(np.percentile(finish_positions, 90))

    return {
        "target_driver": target_driver,
        "finish_prob_by_position": finish_prob_by_position,
        "expected_position": round(expected_pos, 2),
        "expected_time": round(expected_time, 2),
        "podium_prob": round(podium_prob, 4),
        "win_prob": round(win_prob, 4),
        "percentiles": {"p10": p10, "p50": p50, "p90": p90},
        "sample_trajectories": sample_paths,
    }
