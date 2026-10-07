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
        return float(state.last_lap_times[driver])
    # Fallback to field median
    valid = [t for t in state.last_lap_times.values() if t > 60]
    return float(np.median(valid)) if valid else 90.0


def run_monte_carlo(
    state: RaceState,
    strategy: dict[str, Any] | None = None,
    target_driver: str = "VER",
    n_sims: int = 1000,
    lap_time_predictor=None,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Run N forward stochastic Monte Carlo simulations (FR-6 >= 1000 samples)
    from `state.lap` to `total_laps` under full uncertainty.

    Args:
        state: Starting RaceState as of decision lap.
        strategy: Dict containing planned pit stops for target_driver:
            e.g. {'pit_laps': [25], 'compounds': ['HARD']} or empty for stay out.
        target_driver: Driver code being evaluated (e.g. 'VER', 'HAM').
        n_sims: Number of Monte Carlo iterations (default: 1000, PRD FR-6 requirement >= 1000).
        lap_time_predictor: Pre-built LapTimePredictor or None to create one.
        seed: Random seed for reproducibility.

    Returns:
        Comprehensive distribution dict containing:
        - finish_prob_by_position: Full PMF {1: p1, 2: p2, ...}
        - expected_position, position_std, expected_time, time_std
        - win_prob, podium_prob, top5_prob, points_prob
        - percentiles: {p10, p25, p50, p75, p90, iqr}
        - ci_95_expected_pos: [lower, upper]
        - trajectory_bands: Lap-by-lap percentiles (p10, p25, p50, p75, p90, mean_gap)
        - sample_trajectories: Sample individual paths for visualization
    """
    if seed is not None:
        np.random.seed(seed)

    remaining_laps = state.total_laps - state.lap
    if remaining_laps <= 0:
        pos = state.positions.get(target_driver, 1)
        return {
            "target_driver": target_driver,
            "n_sims": n_sims,
            "finish_prob_by_position": {pos: 1.0},
            "expected_position": float(pos),
            "position_std": 0.0,
            "expected_time": 0.0,
            "time_std": 0.0,
            "win_prob": 1.0 if pos == 1 else 0.0,
            "podium_prob": 1.0 if pos <= 3 else 0.0,
            "top5_prob": 1.0 if pos <= 5 else 0.0,
            "points_prob": 1.0 if pos <= 10 else 0.0,
            "percentiles": {"p10": pos, "p25": pos, "p50": pos, "p75": pos, "p90": pos, "iqr": 0},
            "ci_95_expected_pos": [float(pos), float(pos)],
            "trajectory_bands": [{"lap": state.lap, "p10": pos, "p25": pos, "p50": pos, "p75": pos, "p90": pos, "mean_gap": 0.0}],
            "sample_trajectories": [[pos]],
        }

    # Parse strategy for target driver
    pit_plan: dict[int, str] = {}
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
    num_drivers = len(drivers)

    driver_base_paces = {d: _get_driver_base_pace(state, d) for d in drivers}

    # Predictor initialization
    if lap_time_predictor is None:
        try:
            lap_time_predictor = make_lap_time_predictor(state.race_id, state.lap)
        except Exception:
            lap_time_predictor = None

    n_track = min(10, n_sims)  # sample individual paths kept for UI
    track_driver_idx = drivers.index(target_driver)

    # Fast state arrays:
    # gaps: (n_sims, num_drivers)
    # tyre_ages: (n_sims, num_drivers)
    # tyre_compounds: list of lists
    gaps_arr = np.zeros((n_sims, num_drivers), dtype=np.float64)
    for d_idx, d in enumerate(drivers):
        gaps_arr[:, d_idx] = float(state.gaps.get(d, 0.0))

    tyre_ages = np.zeros((n_sims, num_drivers), dtype=np.int32)
    tyre_comps = [[state.tyres.get(d, ("MEDIUM", 1))[0] for d in drivers] for _ in range(n_sims)]
    for d_idx, d in enumerate(drivers):
        tyre_ages[:, d_idx] = int(state.tyres.get(d, ("MEDIUM", 1))[1])

    # Tracking trajectories
    sample_paths: list[list[int]] = [[] for _ in range(n_track)]
    # Per-lap distribution tracking for target driver
    lap_target_positions: list[np.ndarray] = []
    lap_target_gaps: list[np.ndarray] = []
    simulated_laps: list[int] = []

    # Track multi-lap Safety Car episodes across simulations
    sc_remaining = np.full(n_sims, 2 if state.safety_car else 0, dtype=int)

    # Pre-fetch track temperature and precalculate degradation lookup table
    track_temp = float(state.weather.get("track_temp", 30.0)) if state.weather else 30.0
    wear_table: dict[tuple[str, int], float] = {}
    for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        for a in range(80):
            try:
                wear_table[(c, a)] = float(predict_tyre_degradation(c, a, state.circuit, track_temp))
            except Exception:
                wear_table[(c, a)] = float(a) * (0.07 if c == "SOFT" else (0.05 if c == "MEDIUM" else 0.035))

    for sim_lap in range(state.lap + 1, state.total_laps + 1):
        simulated_laps.append(sim_lap)
        lap_frac = sim_lap / state.total_laps
        sc_prob = safety_car_probability(state.circuit, lap_frac)

        # 1. Update Safety Car status across simulations
        is_sc = np.zeros(n_sims, dtype=bool)
        for i in range(n_sims):
            if sc_remaining[i] > 0:
                sc_remaining[i] -= 1
                is_sc[i] = True
            elif np.random.random() < sc_prob:
                dur = sample_safety_car_duration(state.circuit, event_type="SAFETY_CAR")
                sc_remaining[i] = max(0, dur - 1)
                is_sc[i] = True

        # 2. Plan pit stops this lap for each simulation
        stops_by_sim: list[dict[int, str]] = []  # sim_idx -> {driver_idx: new_compound}
        for i in range(n_sims):
            sim_stops: dict[int, str] = {}
            for d_idx, d in enumerate(drivers):
                comp = tyre_comps[i][d_idx]
                age = int(tyre_ages[i, d_idx])

                if d == target_driver:
                    if sim_lap in pit_plan:
                        sim_stops[d_idx] = pit_plan[sim_lap]
                    continue

                # Dynamic rival pit logic: cliff degradation or opportunistic SC pit
                max_life = COMPOUND_MAX_LIFESPAN.get(comp, 32)
                pit_chance = 0.0
                if age >= max_life:
                    pit_chance = 0.70
                elif is_sc[i] and age >= int(max_life * 0.55):
                    pit_chance = 0.45
                
                if pit_chance > 0.0 and np.random.random() < pit_chance:
                    sim_stops[d_idx] = "HARD" if comp in ("MEDIUM", "SOFT") else "MEDIUM"
            stops_by_sim.append(sim_stops)

        # 3. Predict lap paces using fast wear lookup table
        pace_matrix = np.zeros((n_sims, num_drivers), dtype=np.float64)
        for d_idx, d in enumerate(drivers):
            base_p = driver_base_paces[d]
            for i in range(n_sims):
                stops = stops_by_sim[i]
                if d_idx in stops:
                    comp_eval = stops[d_idx]
                    age_eval = 0
                else:
                    comp_eval = tyre_comps[i][d_idx]
                    age_eval = int(tyre_ages[i, d_idx])

                wear = wear_table.get((comp_eval, min(79, age_eval)), float(age_eval) * 0.05)
                pace_matrix[i, d_idx] = base_p + wear

        # Lap-to-lap stochastic variance & pit execution uncertainty
        noise = np.random.normal(0.0, 0.32, size=(n_sims, num_drivers))
        lap_times = pace_matrix + noise

        # Pit loss calculation with tail risk (pit stop fumbles / delays)
        for i in range(n_sims):
            stops = stops_by_sim[i]
            if is_sc[i]:
                lap_times[i, :] *= 1.537  # SC pace delta
                for d_idx, new_c in stops.items():
                    # SC pit delta ~12.5s with slight execution noise
                    sc_pit_loss = PIT_LOSS_SAFETY_CAR + np.random.normal(0.0, 0.4)
                    lap_times[i, d_idx] += sc_pit_loss
                    tyre_comps[i][d_idx] = new_c
                    tyre_ages[i, d_idx] = 0
            else:
                for d_idx, new_c in stops.items():
                    # Racing pit delta ~22.0s + 3% tail risk of slow pit stop (2.5s delay)
                    pit_loss = PIT_LOSS_RACING + np.random.normal(0.0, 0.45)
                    if np.random.random() < 0.035:
                        pit_loss += float(np.random.uniform(2.0, 4.5))
                    lap_times[i, d_idx] += pit_loss
                    tyre_comps[i][d_idx] = new_c
                    tyre_ages[i, d_idx] = 0

        # Increment tyre age for non-stopping cars
        for i in range(n_sims):
            stops = stops_by_sim[i]
            for d_idx in range(num_drivers):
                if d_idx not in stops:
                    tyre_ages[i, d_idx] += 1

        # 4. Update cumulative gaps
        gaps_arr += lap_times

        # Normalize gaps to race leader
        min_gaps = np.min(gaps_arr, axis=1, keepdims=True)
        gaps_arr -= min_gaps

        # 5. Bunching compression under Safety Car
        for i in range(n_sims):
            if is_sc[i]:
                sorted_idx = np.argsort(gaps_arr[i])
                curr_gap = 0.0
                for rank, d_idx in enumerate(sorted_idx):
                    if rank == 0:
                        gaps_arr[i, d_idx] = 0.0
                    else:
                        curr_gap += 0.85 + float(np.random.uniform(-0.1, 0.15))
                        gaps_arr[i, d_idx] = round(float(min(gaps_arr[i, d_idx], curr_gap)), 2)

        # 6. Rank positions across simulations
        # argsort of argsort gives 0-indexed rank; add 1 for 1-indexed finishing position
        positions_arr = np.argsort(np.argsort(gaps_arr, axis=1), axis=1) + 1
        target_pos_this_lap = positions_arr[:, track_driver_idx]
        target_gaps_this_lap = gaps_arr[:, track_driver_idx]

        lap_target_positions.append(target_pos_this_lap.copy())
        lap_target_gaps.append(target_gaps_this_lap.copy())

        for trk_i in range(n_track):
            sample_paths[trk_i].append(int(target_pos_this_lap[trk_i]))

    # Final finishing distributions
    final_positions = positions_arr[:, track_driver_idx]
    final_times = gaps_arr[:, track_driver_idx]

    pos_counts: dict[int, int] = {}
    for p in final_positions:
        pos_counts[int(p)] = pos_counts.get(int(p), 0) + 1

    finish_prob_by_position = {
        p: round(count / n_sims, 4) for p, count in sorted(pos_counts.items())
    }

    # Summary statistics
    expected_pos = float(np.mean(final_positions))
    position_std = float(np.std(final_positions))
    expected_time = float(np.mean(final_times))
    time_std = float(np.std(final_times))

    win_prob = float(finish_prob_by_position.get(1, 0.0))
    podium_prob = float(sum(p for pos, p in finish_prob_by_position.items() if pos <= 3))
    top5_prob = float(sum(p for pos, p in finish_prob_by_position.items() if pos <= 5))
    points_prob = float(sum(p for pos, p in finish_prob_by_position.items() if pos <= 10))

    p10 = int(np.percentile(final_positions, 10))
    p25 = int(np.percentile(final_positions, 25))
    p50 = int(np.percentile(final_positions, 50))
    p75 = int(np.percentile(final_positions, 75))
    p90 = int(np.percentile(final_positions, 90))
    iqr = p75 - p25

    # 95% confidence interval for expected position (Central Limit Theorem: mean +/- 1.96 * SE)
    se_pos = position_std / np.sqrt(n_sims)
    ci_95 = [
        round(max(1.0, expected_pos - 1.96 * se_pos), 3),
        round(min(float(num_drivers), expected_pos + 1.96 * se_pos), 3),
    ]

    # Trajectory ribbon bands per simulated lap
    trajectory_bands = []
    for lap_idx, l_num in enumerate(simulated_laps):
        l_positions = lap_target_positions[lap_idx]
        l_gaps = lap_target_gaps[lap_idx]
        trajectory_bands.append({
            "lap": l_num,
            "p10": int(np.percentile(l_positions, 10)),
            "p25": int(np.percentile(l_positions, 25)),
            "p50": int(np.percentile(l_positions, 50)),
            "p75": int(np.percentile(l_positions, 75)),
            "p90": int(np.percentile(l_positions, 90)),
            "mean_gap": round(float(np.mean(l_gaps)), 2),
        })

    return {
        "target_driver": target_driver,
        "n_sims": n_sims,
        "finish_prob_by_position": finish_prob_by_position,
        "expected_position": round(expected_pos, 2),
        "position_std": round(position_std, 2),
        "expected_time": round(expected_time, 2),
        "time_std": round(time_std, 2),
        "win_prob": round(win_prob, 4),
        "podium_prob": round(podium_prob, 4),
        "top5_prob": round(top5_prob, 4),
        "points_prob": round(points_prob, 4),
        "percentiles": {
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "iqr": iqr,
        },
        "ci_95_expected_pos": ci_95,
        "trajectory_bands": trajectory_bands,
        "sample_trajectories": sample_paths,
    }
