"""Strategy Optimizer Engine (Design.md Section 6.7).

Evaluates candidate pit stop strategies (pit lap windows × compound choices)
via Monte Carlo simulation and selects the optimal call.
Locked interface contract per Design.md Section 5 and overtake-15-day-sprint-plan.md Day 8.
"""

from __future__ import annotations

import logging
from typing import Any
from src.simulation.monte_carlo import run_monte_carlo
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


def get_strategy_recommendation(
    state: RaceState,
    target_driver: str = "VER",
    n_sims: int = 300,
) -> dict[str, Any]:
    """Calculate the optimal pit/tyre strategy recommendation for target_driver.

    Interface contract:
        Returns {
            'action': str,
            'tyre': str,
            'pit_lap': int | None,
            'expected_gain': float,
            'confidence': float,
            'expected_position': float,
            'candidates': list[dict],
            'reasoning': str,
        }
    """
    candidates = generate_candidate_strategies(state, target_driver)

    evaluated_candidates = []
    baseline_result = None

    for candidate in candidates:
        sim_res = run_monte_carlo(
            state=state,
            strategy=candidate,
            target_driver=target_driver,
            n_sims=n_sims,
        )
        cand_data = {
            **candidate,
            "expected_position": sim_res["expected_position"],
            "win_prob": sim_res["win_prob"],
            "podium_prob": sim_res["podium_prob"],
            "finish_prob_by_position": sim_res["finish_prob_by_position"],
            "expected_time": sim_res["expected_time"],
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

    # Construct tactical reasoning
    current_comp, current_age = state.tyres.get(target_driver, ("MEDIUM", 1))
    if action == "STAY OUT":
        reasoning = (
            f"Current {current_comp} tyres (age {current_age}) have sufficient life. "
            f"Pitting incurs ~22s track loss which cannot be recovered over remaining {state.total_laps - state.lap} laps."
        )
    elif "BOX THIS LAP" in action:
        reasoning = (
            f"Box immediately for fresh {tyre}. Current {current_comp} (age {current_age}) is degrading rapidly. "
            f"Expected gain of {expected_gain:+.1f} positions with fresh tyre pace advantage."
        )
    else:
        reasoning = (
            f"Target pit window around Lap {pit_lap} for {tyre}. Maximizes tyre offset against competitors "
            f"while protecting track position (expected finish P{best_candidate['expected_position']:.1f})."
        )

    return {
        "target_driver": target_driver,
        "action": action,
        "tyre": tyre,
        "pit_lap": pit_lap,
        "expected_gain": expected_gain,
        "confidence": confidence,
        "expected_position": best_candidate["expected_position"],
        "win_prob": best_candidate["win_prob"],
        "podium_prob": best_candidate["podium_prob"],
        "reasoning": reasoning,
        "candidates": evaluated_candidates,
    }
