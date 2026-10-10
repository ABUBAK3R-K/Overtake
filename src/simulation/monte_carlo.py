"""Monte Carlo Forward Simulation Engine (Design.md Section 6.6).

Simulates multiple forward race trajectories from the current RaceState under uncertainty,
accounting for tyre degradation, pit stop delta, lap-time variance, and Safety Car risk.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from src.models.lap_time import COMPOUNDS, make_lap_time_predictor
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


SOFT, MEDIUM, HARD = (COMPOUNDS.index(c) for c in ("SOFT", "MEDIUM", "HARD"))
MAX_LIFE_BY_CODE = np.array([COMPOUND_MAX_LIFESPAN.get(c, 32) for c in COMPOUNDS])


def _compound_code(compound: str) -> int:
    compound = str(compound).upper()
    return COMPOUNDS.index(compound) if compound in COMPOUNDS else MEDIUM


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
    rival_strategies: dict[str, dict[str, Any]] | None = None,
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
        rival_strategies: Fixed pit plans for other drivers, same shape as
            `strategy` ({driver: {'pit_laps': [...], 'compounds': [...]}}).
            Those drivers follow their plan instead of the rival pit
            heuristic (used by the game-theory engine).

    Returns:
        Comprehensive distribution dict containing:
        - finish_prob_by_position: Full PMF {1: p1, 2: p2, ...}
        - expected_position, position_std, expected_time, time_std
        - win_prob, podium_prob, top5_prob, points_prob
        - percentiles: {p10, p25, p50, p75, p90, iqr}
        - ci_95_expected_pos: [lower, upper]
        - trajectory_bands: Lap-by-lap percentiles (p10, p25, p50, p75, p90, mean_gap)
        - sample_trajectories: Sample individual paths for visualization
        - expected_position_by_driver: mean finishing position of every car
    """
    scenario = {"strategy": strategy, "rival_strategies": rival_strategies}
    return run_monte_carlo_batch(state, [scenario], target_driver, n_sims,
                                 lap_time_predictor, seed)[0]


def run_monte_carlo_batch(
    state: RaceState,
    scenarios: list[dict[str, Any]],
    target_driver: str = "VER",
    n_sims: int = 1000,
    lap_time_predictor=None,
    seed: int | None = 42,
) -> list[dict[str, Any]]:
    """run_monte_carlo() for several scenarios at once, one result per scenario.

    Each scenario is {'strategy': plan for target_driver, 'rival_strategies':
    {driver: plan}} (both optional). All scenarios are stacked into one
    simulation, so the lap-time model is called once per lap for the whole
    batch rather than once per lap per scenario, and they share common random
    numbers: simulation i of every scenario sees the same safety cars, noise
    and rival pit draws. Differences between scenarios are therefore paired,
    i.e. due to the plans, not to luck.
    """
    if seed is not None:
        np.random.seed(seed)
    n_scen = len(scenarios)
    if n_scen == 0:
        return []

    if state.total_laps - state.lap <= 0:
        return [_finished_result(state, target_driver, n_sims) for _ in scenarios]

    def _pit_plan(plan: dict[str, Any] | None) -> dict[int, int]:
        """{lap: compound code} for the stops still ahead of state.lap."""
        plan = plan or {}
        return {int(plap): _compound_code(pcomp)
                for plap, pcomp in zip(plan.get("pit_laps", []), plan.get("compounds", []))
                if plap > state.lap}

    drivers = list(state.positions.keys())
    if target_driver not in drivers:
        drivers.append(target_driver)
    num_drivers = len(drivers)
    track_driver_idx = drivers.index(target_driver)
    n_rows = n_scen * n_sims  # row b * n_sims + i = simulation i of scenario b

    # Drivers on a fixed plan per scenario (target + any given rivals):
    # (row slice, driver index, {lap: compound code})
    fixed: list[tuple[slice, int, dict[int, int]]] = []
    for b, sc_spec in enumerate(scenarios):
        rows = slice(b * n_sims, (b + 1) * n_sims)
        fixed.append((rows, track_driver_idx, _pit_plan(sc_spec.get("strategy"))))
        for d, plan in (sc_spec.get("rival_strategies") or {}).items():
            if d in drivers and d != target_driver:
                fixed.append((rows, drivers.index(d), _pit_plan(plan)))

    if lap_time_predictor is None:
        try:
            lap_time_predictor = make_lap_time_predictor(state.race_id, state.lap)
        except Exception:
            lap_time_predictor = None

    def _draw(fn, *args, cols: int | None = None) -> np.ndarray:
        """One draw per simulation, repeated for every scenario (common random numbers)."""
        shape = (n_sims,) if cols is None else (n_sims, cols)
        return np.tile(fn(*args, size=shape), (n_scen,) + (1,) * (len(shape) - 1))

    gaps_arr = np.tile([float(state.gaps.get(d, 0.0)) for d in drivers], (n_rows, 1))
    # Compounds as int codes into COMPOUNDS (unknown labels read as MEDIUM)
    start_tyres = [state.tyres.get(d, ("MEDIUM", 1)) for d in drivers]
    tyre_comps = np.tile([_compound_code(c) for c, _ in start_tyres], (n_rows, 1)).astype(np.int64)
    tyre_ages = np.tile([int(a) for _, a in start_tyres], (n_rows, 1)).astype(np.int64)

    # Fallback pace (no model answer): recent pace + tyre wear lookup
    base_paces = np.array([_get_driver_base_pace(state, d) for d in drivers])
    track_temp = float(state.weather.get("track_temp", 30.0)) if state.weather else 30.0
    wear_lookup = np.zeros((len(COMPOUNDS), 80))
    for code, c in enumerate(COMPOUNDS):
        for a in range(80):
            try:
                wear_lookup[code, a] = float(predict_tyre_degradation(c, a, state.circuit, track_temp))
            except Exception:
                wear_lookup[code, a] = float(a) * (0.07 if c == "SOFT" else (0.05 if c == "MEDIUM" else 0.035))

    # Safety-car episodes are independent of any plan: tracked per simulation
    sc_remaining = np.full(n_sims, 2 if state.safety_car else 0, dtype=int)

    lap_target_positions: list[np.ndarray] = []
    lap_target_gaps: list[np.ndarray] = []
    simulated_laps = list(range(state.lap + 1, state.total_laps + 1))

    for sim_lap in simulated_laps:
        sc_prob = safety_car_probability(state.circuit, sim_lap / state.total_laps)

        # 1. Update Safety Car status across simulations
        sim_sc = sc_remaining > 0
        sc_remaining[sim_sc] -= 1
        new_sc = ~sim_sc & (np.random.random(n_sims) < sc_prob)
        for i in np.nonzero(new_sc)[0]:
            dur = sample_safety_car_duration(state.circuit, event_type="SAFETY_CAR")
            sc_remaining[i] = max(0, dur - 1)
        is_sc = np.tile(sim_sc | new_sc, n_scen)

        # 2. Plan pit stops this lap. Rivals: past the compound's useful life,
        # or opportunistically under a safety car. Target (and any rival
        # given a fixed plan): their plan.
        max_life = MAX_LIFE_BY_CODE[tyre_comps]
        pit_chance = np.where(tyre_ages >= max_life, 0.70,
                              np.where(is_sc[:, None] & (tyre_ages >= (max_life * 0.55).astype(int)), 0.45, 0.0))
        for rows, d_idx, _ in fixed:
            pit_chance[rows, d_idx] = 0.0
        stop_mask = _draw(np.random.random, cols=num_drivers) < pit_chance
        new_comps = np.where(np.isin(tyre_comps, (SOFT, MEDIUM)), HARD, MEDIUM)
        for rows, d_idx, plan in fixed:
            if sim_lap in plan:
                stop_mask[rows, d_idx] = True
                new_comps[rows, d_idx] = plan[sim_lap]

        # 3. Predict this lap's pace. Tyres as they leave the end of the
        # previous lap: a car stopping this lap runs it on the new set (age 0).
        lap_comps = np.where(stop_mask, new_comps, tyre_comps)
        lap_ages = np.where(stop_mask, 0, tyre_ages)

        # Primary: the lap-time model (fuel burn-off, compound pace, traffic,
        # SC pace), every sim and driver in one call. It returns SC-scaled
        # times already.
        pace_matrix = np.full((n_rows, num_drivers), np.nan)
        if lap_time_predictor is not None:
            try:
                pace_matrix = lap_time_predictor.predict_arrays(
                    sim_lap - 1, drivers, lap_comps, lap_ages, gaps_arr, is_sc)
            except Exception:
                log.warning("lap-time model prediction failed at lap %d; falling back",
                            sim_lap, exc_info=True)

        # Fallback where the model has no answer (unseen race, no clean
        # history yet): recent pace + tyre wear, SC-scaled here.
        missing = ~np.isfinite(pace_matrix)
        if missing.any():
            fallback = base_paces[None, :] + wear_lookup[lap_comps, np.minimum(lap_ages, 79)]
            fallback = np.where(is_sc[:, None], fallback * 1.537, fallback)
            pace_matrix = np.where(missing, fallback, pace_matrix)

        # Lap-to-lap stochastic variance
        lap_times = pace_matrix + _draw(np.random.normal, 0.0, 0.32, cols=num_drivers)

        # Pit loss with execution uncertainty and tail risk (slow stops)
        pit_loss = np.where(
            is_sc[:, None],
            PIT_LOSS_SAFETY_CAR + _draw(np.random.normal, 0.0, 0.4, cols=num_drivers),
            PIT_LOSS_RACING + _draw(np.random.normal, 0.0, 0.45, cols=num_drivers)
            + np.where(_draw(np.random.random, cols=num_drivers) < 0.035,
                       _draw(np.random.uniform, 2.0, 4.5, cols=num_drivers), 0.0),
        )
        lap_times += np.where(stop_mask, pit_loss, 0.0)

        # Tyres at the end of this lap: one lap older (a fresh set has done 1)
        tyre_comps = lap_comps
        tyre_ages = lap_ages + 1

        # 4. Update cumulative gaps, normalised to the race leader
        gaps_arr += lap_times
        gaps_arr -= np.min(gaps_arr, axis=1, keepdims=True)

        # 5. Bunching compression under Safety Car: each car closes to
        # ~0.85 s behind the car ahead (never pushed further back).
        if is_sc.any():
            spacing_all = 0.85 + _draw(np.random.uniform, -0.1, 0.15, cols=num_drivers)
            rows = np.nonzero(is_sc)[0]
            order = np.argsort(gaps_arr[rows], axis=1)
            spacing = spacing_all[rows]
            spacing[:, 0] = 0.0
            sorted_gaps = np.take_along_axis(gaps_arr[rows], order, axis=1)
            bunched = np.round(np.minimum(sorted_gaps, np.cumsum(spacing, axis=1)), 2)
            bunched[:, 0] = 0.0
            block = gaps_arr[rows]
            np.put_along_axis(block, order, bunched, axis=1)
            gaps_arr[rows] = block

        # 6. Rank positions (argsort of argsort = 0-indexed rank)
        positions_arr = np.argsort(np.argsort(gaps_arr, axis=1), axis=1) + 1
        lap_target_positions.append(positions_arr[:, track_driver_idx].copy())
        lap_target_gaps.append(gaps_arr[:, track_driver_idx].copy())

    all_positions = np.stack(lap_target_positions)  # (laps, n_rows)
    all_gaps = np.stack(lap_target_gaps)
    results = []
    for b in range(n_scen):
        rows = slice(b * n_sims, (b + 1) * n_sims)
        results.append(_summarize(
            target_driver, n_sims, num_drivers, simulated_laps,
            all_positions[:, rows], all_gaps[:, rows],
            {d: round(float(m), 3) for d, m in zip(drivers, positions_arr[rows].mean(axis=0))},
        ))
    return results


def _finished_result(state: RaceState, target_driver: str, n_sims: int) -> dict[str, Any]:
    """Result when no laps remain: the current order is the finish."""
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
        "expected_position_by_driver": {d: float(p) for d, p in state.positions.items()},
    }


def _summarize(target_driver: str, n_sims: int, num_drivers: int, simulated_laps: list[int],
               lap_positions: np.ndarray, lap_gaps: np.ndarray,
               expected_position_by_driver: dict[str, float]) -> dict[str, Any]:
    """Distribution summary for one scenario from (laps, n_sims) target-driver arrays."""
    final_positions = lap_positions[-1]
    final_times = lap_gaps[-1]

    values, counts = np.unique(final_positions, return_counts=True)
    finish_prob_by_position = {int(p): round(c / n_sims, 4) for p, c in zip(values, counts)}

    expected_pos = float(np.mean(final_positions))
    position_std = float(np.std(final_positions))

    win_prob = float(finish_prob_by_position.get(1, 0.0))
    podium_prob = float(sum(p for pos, p in finish_prob_by_position.items() if pos <= 3))
    top5_prob = float(sum(p for pos, p in finish_prob_by_position.items() if pos <= 5))
    points_prob = float(sum(p for pos, p in finish_prob_by_position.items() if pos <= 10))

    p10, p25, p50, p75, p90 = (int(np.percentile(final_positions, q)) for q in (10, 25, 50, 75, 90))

    # 95% confidence interval for expected position (Central Limit Theorem: mean +/- 1.96 * SE)
    se_pos = position_std / np.sqrt(n_sims)
    ci_95 = [
        round(max(1.0, expected_pos - 1.96 * se_pos), 3),
        round(min(float(num_drivers), expected_pos + 1.96 * se_pos), 3),
    ]

    # Trajectory ribbon bands per simulated lap
    bands = np.percentile(lap_positions, [10, 25, 50, 75, 90], axis=1).astype(int)
    mean_gaps = lap_gaps.mean(axis=1)
    trajectory_bands = [
        {"lap": l_num, "p10": int(bands[0, k]), "p25": int(bands[1, k]), "p50": int(bands[2, k]),
         "p75": int(bands[3, k]), "p90": int(bands[4, k]), "mean_gap": round(float(mean_gaps[k]), 2)}
        for k, l_num in enumerate(simulated_laps)
    ]

    n_track = min(10, n_sims)  # sample individual paths kept for UI
    sample_paths = lap_positions[:, :n_track].T.astype(int).tolist()

    return {
        "target_driver": target_driver,
        "n_sims": n_sims,
        "finish_prob_by_position": finish_prob_by_position,
        "expected_position": round(expected_pos, 2),
        "position_std": round(position_std, 2),
        "expected_time": round(float(np.mean(final_times)), 2),
        "time_std": round(float(np.std(final_times)), 2),
        "win_prob": round(win_prob, 4),
        "podium_prob": round(podium_prob, 4),
        "top5_prob": round(top5_prob, 4),
        "points_prob": round(points_prob, 4),
        "percentiles": {"p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "iqr": p75 - p25},
        "ci_95_expected_pos": ci_95,
        "trajectory_bands": trajectory_bands,
        "sample_trajectories": sample_paths,
        "expected_position_by_driver": expected_position_by_driver,
    }
