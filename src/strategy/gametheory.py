"""Strategy Engine 2 — Competitor-Aware Game Theory (Design.md Section 6.9 / FR-7).

A Stackelberg leader/follower game between this car (the leader: it commits
to a plan first) and its nearest rival (the follower: it sees the plan and
answers it). Both plans are fixed inside one joint Monte Carlo simulation,
which reports both cars' expected finishing positions at once:

  for each of our candidates C:
      for each rival option R:  joint sim(C, R) -> (our E[pos], rival E[pos])
      R*(C) = the R that is best for the rival, given C
      value(C) = our E[pos] under (C, R*(C))
  pick argmin value(C)

All joint simulations share one seed, so every comparison is paired.
Everyone else on track follows the simulator's usual rival pit heuristic.

This differs from Engine 1 (exhaustive search) exactly where a rival's
reply changes the answer: e.g. an undercut that only works if the car
ahead doesn't cover it.

Return shape matches Engine 1:
    {action, tyre, pit_lap, expected_gain, confidence, engine, ...}
so the API layer can render any engine identically.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.lap_time import make_lap_time_predictor
from src.simulation.monte_carlo import run_monte_carlo_batch
from src.simulation.state import RaceState
from src.strategy.optimizer import describe_plan, explain_choice, generate_candidate_strategies

log = logging.getLogger("overtake.strategy.gametheory")

# The rival's options: stay out, or box now / soon. Later stops barely
# interact with a decision made now and would multiply the joint grid.
RIVAL_OPTION_PREFIXES = ("STAY_OUT", "BOX_NOW_")
RIVAL_MAX_OFFSET = 3

# ─── Rival modelling ──────────────────────────────────────────────────────────


def _nearest_rival(state: RaceState, target_driver: str) -> str | None:
    """The car directly ahead (the one to undercut or defend against); the car
    behind if the target leads."""
    order = sorted(state.positions, key=state.positions.get)
    if target_driver not in order or len(order) < 2:
        return None
    i = order.index(target_driver)
    return order[i - 1] if i > 0 else order[1]


def _rival_candidates(state: RaceState, rival_driver: str) -> list[dict[str, Any]]:
    """The rival's near-term options, from the same generator as Engine 1."""
    out = []
    for cand in generate_candidate_strategies(state, rival_driver):
        pit_laps = cand.get("pit_laps", [])
        soon = len(pit_laps) == 1 and pit_laps[0] - state.lap <= RIVAL_MAX_OFFSET
        if cand["name"].startswith(RIVAL_OPTION_PREFIXES) or soon:
            out.append(cand)
    return out or [{"name": "STAY_OUT", "pit_laps": [], "compounds": []}]


# ─── Public handoff function ──────────────────────────────────────────────────


def get_strategy_recommendation_gametheory(
    state: RaceState,
    rival_state: RaceState,
    target_driver: str | None = None,
    n_sims: int = 200,
) -> dict[str, Any]:
    """Stackelberg game-theoretic strategy recommendation.

    Interface contract (Design.md §5): returns
    ``{action, tyre, pit_lap, expected_gain, confidence, engine, ...}``
    identical shape to Engine 1, plus ``rival_driver`` and, per candidate,
    the rival's best response to it.

    Args:
        state:         Current race state used for the joint simulation.
        rival_state:   Race state the rival's options are generated from
                       (usually the same object as ``state``).
        target_driver: The driver to optimise for. Defaults to the leader of
                       ``state.positions`` if omitted.
        n_sims:        Monte Carlo simulations per (our plan, rival plan) pair.
    """
    sorted_positions = sorted(state.positions, key=lambda d: state.positions[d])
    if target_driver is None:
        target_driver = sorted_positions[0] if sorted_positions else "VER"

    rival_driver = _nearest_rival(state, target_driver)
    if rival_driver is None:
        log.warning("No rival driver found; falling back to exhaustive search.")
        from src.strategy.optimizer import get_strategy_recommendation
        result = get_strategy_recommendation(state, target_driver=target_driver, n_sims=n_sims)
        result["engine"] = "gametheory"
        return result

    try:
        predictor = make_lap_time_predictor(state.race_id, state.lap)
    except Exception:
        predictor = None  # run_monte_carlo falls back to recent pace + wear

    rival_options = _rival_candidates(rival_state, rival_driver)
    leader_options = generate_candidate_strategies(state, target_driver)
    # The whole (our plan x rival plan) grid in one batched joint simulation
    grid = [(c, r) for c in leader_options for r in rival_options]
    try:
        results = run_monte_carlo_batch(
            state, [{"strategy": c, "rival_strategies": {rival_driver: r}} for c, r in grid],
            target_driver=target_driver, n_sims=n_sims, lap_time_predictor=predictor,
        )
    except Exception as exc:
        log.warning("joint Monte Carlo failed: %s", exc)
        results = []

    evaluated: list[dict[str, Any]] = []
    n_r = len(rival_options)
    for k, cand in enumerate(leader_options if results else []):
        responses = [
            (res["expected_position_by_driver"].get(rival_driver, 99.0), rival_plan, res)
            for rival_plan, res in zip(rival_options, results[k * n_r:(k + 1) * n_r])
        ]
        # Follower's best response to this plan (first listed wins ties)
        rival_pos, rival_plan, res = min(responses, key=lambda r: r[0])
        evaluated.append({
            **cand,
            "expected_position": res["expected_position"],
            "win_prob": res["win_prob"],
            "podium_prob": res["podium_prob"],
            "finish_prob_by_position": res["finish_prob_by_position"],
            "expected_time": res["expected_time"],
            "rival_response": rival_plan["name"],
            "rival_expected_position": rival_pos,
            # Our E[pos] against each rival option, for the "why" in the UI
            "vs_rival_options": {r[1]["name"]: r[2]["expected_position"] for r in responses},
        })

    if not evaluated:
        from src.strategy.optimizer import get_strategy_recommendation
        result = get_strategy_recommendation(state, target_driver=target_driver, n_sims=n_sims)
        result["engine"] = "gametheory"
        return result

    baseline_result = next((c for c in evaluated if c["name"] == "STAY_OUT"), evaluated[0])
    best = min(evaluated, key=lambda c: c["expected_position"])
    baseline_pos = baseline_result["expected_position"]
    expected_gain = round(baseline_pos - best["expected_position"], 2)

    confidence = round(float(
        best["podium_prob"] if best["podium_prob"] > 0.1 else best["win_prob"]
    ), 2)
    if confidence == 0.0:
        confidence = 0.70

    action = best.get("action", "STAY OUT")
    compounds = best.get("compounds", [])
    tyre = compounds[0] if compounds else state.tyres.get(target_driver, ("MEDIUM", 1))[0]
    pit_laps = best.get("pit_laps", [])
    pit_lap = pit_laps[0] if pit_laps else None

    rival_plan = next((r for r in rival_options if r["name"] == best["rival_response"]), {})
    reasoning = (
        f"{rival_driver} answers best by choosing to {describe_plan(rival_plan, state.lap)}. "
        + explain_choice(best, evaluated, state.lap)
        + " Every position here is after the rival's best answer to that plan."
    )

    return {
        "target_driver": target_driver,
        "rival_driver": rival_driver,
        "action": action,
        "tyre": tyre,
        "pit_lap": pit_lap,
        "expected_gain": expected_gain,
        "confidence": confidence,
        "expected_position": best["expected_position"],
        "win_prob": best["win_prob"],
        "podium_prob": best["podium_prob"],
        "reasoning": reasoning,
        "engine": "gametheory",
        "strategy": {k: best.get(k) for k in ("name", "pit_laps", "compounds")},
        "candidates": evaluated,
    }
