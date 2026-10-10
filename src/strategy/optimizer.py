"""Strategy Optimizer Engine (Design.md Section 6.7).

Evaluates candidate pit stop strategies (pit lap windows × compound choices)
via Monte Carlo simulation and selects the optimal call.
Locked interface contract per Design.md Section 5 and overtake-15-day-sprint-plan.md Day 8.
"""

from __future__ import annotations

import logging
from typing import Any
from src.simulation.monte_carlo import run_monte_carlo_batch
from src.simulation.state import RaceState

log = logging.getLogger("overtake.strategy.optimizer")

CANDIDATE_DRY_COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]


def generate_candidate_strategies(
    state: RaceState,
    target_driver: str,
) -> list[dict[str, Any]]:
    """Generate plausible candidate pit strategies from current race state."""
    current_lap = state.lap
    total_laps = state.total_laps
    remaining_laps = total_laps - current_lap

    if remaining_laps <= 1:
        return [{"name": "STAY_OUT", "pit_laps": [], "compounds": []}]

    current_compound, current_age = state.tyres.get(target_driver, ("MEDIUM", 1))

    candidates: list[dict[str, Any]] = []

    # 1. Baseline: Stay out (no more pit stops)
    candidates.append({
        "name": "STAY_OUT",
        "action": "STAY OUT",
        "pit_laps": [],
        "compounds": [],
        "description": "Stay out to the end on current set",
    })

    # Available new compounds (prefer switching compounds to satisfy 2-compound rule)
    available_compounds = [c for c in CANDIDATE_DRY_COMPOUNDS if c != current_compound] or CANDIDATE_DRY_COMPOUNDS

    # 2. Pit this lap / next lap (Box Now)
    for comp in available_compounds:
        # Check if compound is viable for remaining distance
        if comp == "SOFT" and remaining_laps > 18:
            continue  # Softs won't survive
        candidates.append({
            "name": f"BOX_NOW_{comp}",
            "action": "BOX THIS LAP",
            "pit_laps": [current_lap + 1],
            "compounds": [comp],
            "description": f"Box now on Lap {current_lap + 1} for {comp}",
        })

    # 3. Pit in short window (+2 to +8 laps) if sufficient race remaining
    for offset in [3, 6, 10]:
        target_pit_lap = current_lap + offset
        if target_pit_lap < total_laps - 3:
            for comp in available_compounds:
                if comp == "SOFT" and (total_laps - target_pit_lap) > 18:
                    continue
                candidates.append({
                    "name": f"BOX_LAP_{target_pit_lap}_{comp}",
                    "action": f"BOX IN {offset} LAPS",
                    "pit_laps": [target_pit_lap],
                    "compounds": [comp],
                    "description": f"Extend stint to Lap {target_pit_lap}, then switch to {comp}",
                })

    # 4. Two-stop option if current tyre is very old and long race remains
    if remaining_laps >= 25:
        first_stop = current_lap + 1
        second_stop = current_lap + 1 + (remaining_laps // 2)
        candidates.append({
            "name": "TWO_STOP_MED_HARD",
            "action": "2-STOP AGGRESSIVE",
            "pit_laps": [first_stop, second_stop],
            "compounds": ["MEDIUM", "HARD"],
            "description": f"2-Stop strategy: Box Lap {first_stop} (MED) and Lap {second_stop} (HARD)",
        })

    return candidates


def describe_plan(candidate: dict[str, Any], lap: int) -> str:
    """Plain-English plan, e.g. 'stay out', 'box now for Hard', 'box on lap 31 for Medium'."""
    pit_laps = candidate.get("pit_laps") or []
    compounds = [str(c).capitalize() for c in candidate.get("compounds") or []]
    if not pit_laps:
        return "stay out"
    first = "box now" if pit_laps[0] <= lap + 1 else f"box on lap {pit_laps[0]}"
    text = f"{first} for {compounds[0]}"
    if len(pit_laps) > 1:
        text += f", then lap {pit_laps[1]} for {compounds[1]}"
    return text


def explain_choice(best: dict[str, Any], evaluated: list[dict[str, Any]], lap: int) -> str:
    """Why `best` won, in the simulator's own numbers: it against staying out
    and against the runner-up."""
    ranked = sorted(evaluated, key=lambda c: c["expected_position"])
    stay = next((c for c in evaluated if not c.get("pit_laps")), None)
    text = (f"Of {len(evaluated)} plans simulated, {describe_plan(best, lap)} finishes best "
            f"on average (P{best['expected_position']:.1f}).")
    if stay is not None and stay is not best:
        text += f" Staying out: P{stay['expected_position']:.1f}."
    runner = next((c for c in ranked if c is not best), None)
    if runner is not None and runner is not stay:
        text += f" Next best, {describe_plan(runner, lap)}: P{runner['expected_position']:.1f}."
    return text[0].upper() + text[1:]


def get_strategy_recommendation(
    state: RaceState,
    target_driver: str = "VER",
    n_sims: int = 500,
) -> dict[str, Any]:
    """Calculate the optimal pit/tyre strategy recommendation for target_driver
    across candidate strategies using Monte Carlo forward simulations (FR-6).

    Interface contract:
        Returns {
            'target_driver': str,
            'action': str,
            'tyre': str,
            'pit_lap': int | None,
            'expected_gain': float,
            'confidence': float,
            'expected_position': float,
            'position_std': float,
            'win_prob': float,
            'podium_prob': float,
            'points_prob': float,
            'percentiles': dict,
            'ci_95': list[float],
            'candidates': list[dict],
            'reasoning': str,
        }
    """
    candidates = generate_candidate_strategies(state, target_driver)

    evaluated_candidates = []
    baseline_result = None

    # All candidates in one batched simulation (paired: same random draws)
    sim_results = run_monte_carlo_batch(
        state, [{"strategy": c} for c in candidates],
        target_driver=target_driver, n_sims=n_sims,
    )
    for candidate, sim_res in zip(candidates, sim_results):
        cand_data = {
            **candidate,
            "expected_position": sim_res["expected_position"],
            "position_std": sim_res["position_std"],
            "win_prob": sim_res["win_prob"],
            "podium_prob": sim_res["podium_prob"],
            "points_prob": sim_res["points_prob"],
            "percentiles": sim_res["percentiles"],
            "ci_95": sim_res["ci_95_expected_pos"],
            "finish_prob_by_position": sim_res["finish_prob_by_position"],
            "expected_time": sim_res["expected_time"],
            "time_std": sim_res["time_std"],
        }
        evaluated_candidates.append(cand_data)
        if candidate["name"] == "STAY_OUT":
            baseline_result = cand_data

    if baseline_result is None and evaluated_candidates:
        baseline_result = evaluated_candidates[0]

    baseline_exp_pos = baseline_result["expected_position"] if baseline_result else 1.0

    # Best candidate minimizes expected position (1 is best, 20 is worst)
    best_candidate = min(
        evaluated_candidates,
        key=lambda c: c["expected_position"],
    )

    # Position gain relative to baseline (positive = positions gained)
    expected_gain = round(baseline_exp_pos - best_candidate["expected_position"], 2)

    # Confidence calculation based on podium probability or position certainty
    confidence = round(float(best_candidate["podium_prob"] if best_candidate["podium_prob"] > 0.1 else best_candidate["win_prob"]), 2)
    if confidence == 0.0:
        confidence = 0.75  # Default reasonable baseline confidence

    action = best_candidate.get("action", "STAY OUT")
    compounds = best_candidate.get("compounds", [])
    tyre = compounds[0] if compounds else state.tyres.get(target_driver, ("MEDIUM", 1))[0]
    pit_laps = best_candidate.get("pit_laps", [])
    pit_lap = pit_laps[0] if pit_laps else None

    # Reasoning built only from what was simulated
    reasoning = explain_choice(best_candidate, evaluated_candidates, state.lap)

    return {
        "target_driver": target_driver,
        "action": action,
        "tyre": tyre,
        "pit_lap": pit_lap,
        "expected_gain": expected_gain,
        "confidence": confidence,
        "expected_position": best_candidate["expected_position"],
        "position_std": best_candidate["position_std"],
        "win_prob": best_candidate["win_prob"],
        "podium_prob": best_candidate["podium_prob"],
        "points_prob": best_candidate["points_prob"],
        "percentiles": best_candidate["percentiles"],
        "ci_95": best_candidate["ci_95"],
        "reasoning": reasoning,
        "candidates": evaluated_candidates,
        # The full chosen plan (a two-stop keeps both stops, unlike pit_lap/tyre).
        "strategy": {k: best_candidate.get(k) for k in ("name", "pit_laps", "compounds")},
    }
