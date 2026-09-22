"""Strategy Engine 3 — Reinforcement Learning Policy (Design.md Section 6.10 / FR-7).

Provides:

  * ``RaceStrategyEnv``  — a gymnasium.Env wrapping the Monte Carlo / replay
    state machine. Actions are discrete: 0 = stay out, 1..N = pit for compound.
  * ``get_strategy_recommendation_rl(state, policy)``  — acts with the trained
    PPO policy (``rl_policy.py``; loaded automatically) and returns the same
    ``{action, tyre, pit_lap, expected_gain, confidence, engine, ...}`` shape as
    the other two engines so the API layer can render it identically.

The environment is self-contained and does NOT require live FastF1 data — it
rebuilds observation vectors from the frozen ``RaceState`` snapshot, exactly as
the Monte Carlo simulator does. This keeps the no-leakage guarantee intact.

Training: ``python -m src.strategy.rl_policy all`` (see that module — it
trains on pre-simulated states because live Monte Carlo per step is too slow).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("overtake.strategy.rl_env")

# ─── Lazy imports so the module is importable without gymnasium/stable-baselines3
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from src.simulation.monte_carlo import run_monte_carlo
from src.simulation.state import RaceState

# Dry compounds in order (action index maps to compound choice)
_COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
OBS_DIM = 14

# ─── Gymnasium Environment ────────────────────────────────────────────────────


def _make_observation(state: RaceState, driver: str) -> np.ndarray:
    """Build a fixed-length float32 observation vector from a RaceState.

    Every input is a field of ``state``, which replay builds through
    ``as_of_lap()``, so the observation can't see past ``state.lap``.

    Features (OBS_DIM = 14):
        0  current_lap / total_laps          (race progress, 0–1)
        1  remaining_laps / total_laps        (inverse of above)
        2  tyre_age / 50.0                   (tyre age normalised)
        3  compound_soft                      (one-hot)
        4  compound_medium                    (one-hot)
        5  compound_hard                      (one-hot)
        6  current_position / n_drivers       (track position, 0–1)
        7  safety_car_active                  (binary)
        8  gap_to_leader / 120.0             (gap, clamped to 2 min)
        9  estimated_pace_loss                (tyre wear in s, 0–5 norm)
       10  laps_since_last_pit / 50.0        (stint length proxy)
       11  n_drivers / 20.0                  (field size)
       12  gap_to_car_ahead / 30.0           (undercut/overcut target, clamped)
       13  gap_to_car_behind / 30.0          (pit-loss cover, clamped)
    """
    total = max(state.total_laps, 1)
    lap = state.lap
    remaining = max(total - lap, 0)

    compound, age = state.tyres.get(driver, ("MEDIUM", 1))
    compound_up = compound.upper()
    c_soft = 1.0 if compound_up == "SOFT" else 0.0
    c_med = 1.0 if compound_up == "MEDIUM" else 0.0
    c_hard = 1.0 if compound_up == "HARD" else 0.0

    n_drivers = max(len(state.positions), 1)
    position = state.positions.get(driver, n_drivers)

    sc_active = 1.0 if state.safety_car else 0.0
    gap = state.gaps.get(driver, 0.0)

    # Estimate pace loss from tyre wear
    pace_loss = 0.0
    try:
        from src.models.tyre import predict_tyre_degradation
        pace_loss = predict_tyre_degradation(compound_up, age, state.circuit, 30.0)
    except Exception:
        pace_loss = age * 0.05

    # Laps since last pit (approximated from tyre age)
    laps_since_pit = age

    # Gaps to the neighbours: whether a stop drops you behind the car you're
    # covering is the core of an undercut call, and gap_to_leader alone can't
    # show it. Missing neighbour (leader / last car) → clamped maximum.
    order = sorted(state.positions, key=lambda d: state.positions[d])
    idx = order.index(driver) if driver in order else len(order) - 1
    ahead = order[idx - 1] if idx > 0 else None
    behind = order[idx + 1] if idx + 1 < len(order) else None
    gap_ahead = gap - state.gaps.get(ahead, gap) if ahead else 30.0
    gap_behind = state.gaps.get(behind, gap) - gap if behind else 30.0

    obs = np.array([
        lap / total,
        remaining / total,
        min(age / 50.0, 1.0),
        c_soft,
        c_med,
        c_hard,
        position / n_drivers,
        sc_active,
        min(abs(gap) / 120.0, 1.0),
        min(pace_loss / 5.0, 1.0),
        min(laps_since_pit / 50.0, 1.0),
        min(n_drivers / 20.0, 1.0),
        min(max(gap_ahead, 0.0) / 30.0, 1.0),
        min(max(gap_behind, 0.0) / 30.0, 1.0),
    ], dtype=np.float32)
    return obs


if _GYM_AVAILABLE:
    class RaceStrategyEnv(gym.Env):
        """Single-step race strategy environment.

        Each episode begins at a fixed ``RaceState``. The agent chooses one
        action: stay out (0) or pit for a compound (1=HARD, 2=MEDIUM, 3=SOFT).
        The environment runs ``n_sims`` Monte Carlo forward simulations and
        returns the negative expected finishing position as the reward (so
        lower expected position → higher reward).

        The episode always terminates after one step (single decision-point
        formulation). For multi-step training, chain multiple instantiations
        or subclass to advance the state after each pit.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            state: RaceState,
            target_driver: str,
            n_sims: int = 100,
        ) -> None:
            super().__init__()
            self._initial_state = state
            self._driver = target_driver
            self._n_sims = n_sims
            self._state = state

            # Action: 0 = stay out, 1 = pit HARD, 2 = pit MEDIUM, 3 = pit SOFT
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
            )

        @classmethod
        def from_state(
            cls, state: RaceState, target_driver: str, n_sims: int = 100
        ) -> "RaceStrategyEnv":
            return cls(state=state, target_driver=target_driver, n_sims=n_sims)

        # gymnasium API -------------------------------------------------

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self._state = self._initial_state
            return _make_observation(self._state, self._driver), {}

        def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
            """Execute one strategy decision and return (obs, reward, done, truncated, info)."""
            state = self._state
            driver = self._driver

            # Translate action → strategy dict
            if action == 0:
                strategy = {"name": "STAY_OUT", "pit_laps": [], "compounds": []}
            else:
                compound = _COMPOUNDS[action - 1]
                next_lap = state.lap + 1
                strategy = {
                    "name": f"PIT_{compound}",
                    "pit_laps": [next_lap],
                    "compounds": [compound],
                }

            try:
                # seed=None (not the run_monte_carlo default of 42): each
                # call must sample fresh noise, or every roll-out of a
                # deterministic policy against a fixed state collapses to the
                # same number — which silently made n_eval_episodes/confidence
                # in get_strategy_recommendation_rl() below meaningless.
                result = run_monte_carlo(
                    state=state,
                    strategy=strategy,
                    target_driver=driver,
                    n_sims=self._n_sims,
                    seed=None,
                )
                expected_pos = result["expected_position"]
            except Exception as exc:
                log.warning("MC error in env step: %s", exc)
                expected_pos = 10.0  # Penalty for failed sim
                result = {
                    "expected_position": expected_pos, "win_prob": 0.0, "podium_prob": 0.0,
                    "finish_prob_by_position": {}, "expected_time": 0.0,
                }

            # Reward: negative expected position so lower pos → higher reward
            reward = -float(expected_pos)
            obs = _make_observation(state, driver)
            return obs, reward, True, False, {
                "expected_position": expected_pos, "strategy": strategy, "mc_result": result,
            }

        def render(self) -> None:
            pass


# ─── Public handoff function ──────────────────────────────────────────────────


def _action_meta(action: int, lap: int) -> dict[str, Any]:
    if action == 0:
        return {"name": "STAY_OUT", "action": "STAY OUT", "pit_laps": [], "compounds": [],
                "description": "Stay out to the end on current set"}
    comp = _COMPOUNDS[action - 1]
    return {"name": f"PIT_{comp}", "action": "BOX THIS LAP", "pit_laps": [lap + 1], "compounds": [comp],
            "description": f"Box now on Lap {lap + 1} for {comp}"}


def _recommend_with_policy(
    state: RaceState, policy: Any, target_driver: str, n_sims: int,
) -> dict[str, Any]:
    """Act with a trained policy. The policy picks the action from the
    observation alone; every action is still Monte Carlo'd (one shared seed,
    so they're compared on the same random draws) to report expected
    positions, the gain vs. staying out, and the candidates table the other
    engines return. That also means the policy's pick can lose to another
    candidate — which is exactly the regret FR-8 measures."""
    from src.strategy.rl_policy import action_probabilities

    obs = _make_observation(state, target_driver)
    try:
        probs = action_probabilities(policy, obs)
        chosen = int(np.argmax(probs))
    except Exception:
        # Not an SB3 actor-critic (e.g. a stub with only .predict()).
        action_arr, _ = policy.predict(obs, deterministic=True)
        chosen = int(np.asarray(action_arr).reshape(-1)[0])
        probs = np.eye(4)[chosen]

    try:
        from src.models.lap_time import make_lap_time_predictor
        predictor = make_lap_time_predictor(state.race_id, state.lap)
    except Exception:
        predictor = None

    seed = int(np.random.default_rng().integers(2**31 - 1))
    results = {}
    for a in range(4):
        meta = _action_meta(a, state.lap)
        results[a] = run_monte_carlo(
            state, {"name": meta["name"], "pit_laps": meta["pit_laps"], "compounds": meta["compounds"]},
            target_driver=target_driver, n_sims=n_sims, lap_time_predictor=predictor, seed=seed,
        )

    candidates = [
        {
            **_action_meta(a, state.lap),
            "expected_position": round(float(r["expected_position"]), 2),
            "win_prob": round(float(r["win_prob"]), 4),
            "podium_prob": round(float(r["podium_prob"]), 4),
            "finish_prob_by_position": r["finish_prob_by_position"],
            "expected_time": round(float(r["expected_time"]), 2),
            "policy_prob": round(float(probs[a]), 3),
        }
        for a, r in results.items()
    ]
    chosen_res = results[chosen]
    expected_position = float(chosen_res["expected_position"])
    expected_gain = round(float(results[0]["expected_position"]) - expected_position, 2)
    mc_best = min(results, key=lambda a: results[a]["expected_position"])

    current_comp, current_age = state.tyres.get(target_driver, ("MEDIUM", 1))
    if chosen == 0:
        action_str, tyre, pit_lap = "STAY OUT", current_comp, None
        reasoning = (f"RL policy ({probs[0]:.0%} on this action) stays out on {current_comp} "
                     f"(age {current_age}). Simulated finish: P{expected_position:.1f}.")
    else:
        action_str, tyre, pit_lap = "BOX THIS LAP", _COMPOUNDS[chosen - 1], state.lap + 1
        reasoning = (f"RL policy ({probs[chosen]:.0%} on this action) boxes for {tyre} on lap {pit_lap}. "
                     f"Simulated finish: P{expected_position:.1f} ({expected_gain:+.1f} vs stay-out).")
    if mc_best != chosen:
        gap = expected_position - float(results[mc_best]["expected_position"])
        reasoning += (f" Monte Carlo at this state rates {_action_meta(mc_best, state.lap)['name']} "
                      f"{gap:.1f} positions better — the policy generalises across races and can miss that.")

    return {
        "target_driver": target_driver,
        "action": action_str,
        "tyre": tyre,
        "pit_lap": pit_lap,
        "expected_gain": expected_gain,
        "confidence": round(float(probs[chosen]), 2),
        "expected_position": round(expected_position, 1),
        "win_prob": round(float(chosen_res["win_prob"]), 4),
        "podium_prob": round(float(chosen_res["podium_prob"]), 4),
        "finish_prob_by_position": chosen_res["finish_prob_by_position"],
        "expected_time": round(float(chosen_res["expected_time"]), 2),
        "reasoning": reasoning,
        "engine": "rl",
        "policy_provided": True,
        "candidates": candidates,
    }


def get_strategy_recommendation_rl(
    state: RaceState,
    policy: Any | None = "auto",
    target_driver: str | None = None,
    n_eval_episodes: int = 10,
    n_sims: int = 100,
) -> dict[str, Any]:
    """Return RL-policy strategy recommendation for the current race state.

    Interface contract (Design.md §5): returns
    ``{action, tyre, pit_lap, expected_gain, confidence, engine, ...}``
    identical shape to Engines 1 and 2.

    Args:
        state:            Current RaceState.
        policy:           A trained stable-baselines3 model (e.g. PPO or DQN).
                          ``"auto"`` (default) loads the saved policy from
                          ``data/models/rl/``; if none is trained yet, or
                          ``policy=None``, falls back to a greedy MC search over
                          all 4 actions (reported as ``policy_provided: False``).
        target_driver:    Driver to optimise for. Defaults to race leader.
        n_eval_episodes:  Number of roll-outs to average over for confidence.
        n_sims:           MC simulations per environment step.

    Returns the same shape dict as ``get_strategy_recommendation`` so the API
    ``?engine=rl`` parameter can render it without special-casing.
    """
    sorted_positions = sorted(state.positions, key=lambda d: state.positions[d])
    if target_driver is None:
        target_driver = sorted_positions[0] if sorted_positions else "VER"

    if not _GYM_AVAILABLE:
        log.warning("gymnasium not installed; RL engine falling back to exhaustive search")
        from src.strategy.optimizer import get_strategy_recommendation
        result = get_strategy_recommendation(state, target_driver=target_driver, n_sims=n_sims)
        result["engine"] = "rl"
        return result

    if isinstance(policy, str) and policy == "auto":
        from src.strategy.rl_policy import load_policy
        policy = load_policy()
    if policy is not None:
        return _recommend_with_policy(state, policy, target_driver, n_sims)

    # Greedy MC fallback (no trained policy).
    env = RaceStrategyEnv.from_state(state, target_driver, n_sims=n_sims)
    obs, _ = env.reset()

    # Collect full MC results per action across episodes (not just the reward
    # scalar) so we can report podium/win probability and a candidates table
    # in the same shape Engines 1 and 2 return — the API dispatches all three
    # engines through one rendering path (Design.md §6.13), so a field missing
    # here silently breaks the dashboard for anyone who selects ?engine=rl.
    action_results: dict[int, list[dict[str, Any]]] = {a: [] for a in range(4)}
    episode_choices: list[int] = []  # the action each episode would have picked, for confidence

    for _ in range(max(n_eval_episodes, 4)):
        # Try each action this episode, keep every result, and let the outer
        # aggregation below pick the best by mean reward. seed=None inside
        # step() means each sweep samples fresh noise, so repeating this
        # across episodes is genuine Monte Carlo averaging.
        episode_best_action, episode_best_reward = 0, float("-inf")
        for a in range(4):
            _, reward, _, _, info = env.step(a)
            action_results[a].append(info["mc_result"])
            if reward > episode_best_reward:
                episode_best_reward, episode_best_action = reward, a
        episode_choices.append(episode_best_action)
        obs, _ = env.reset()

    # Aggregate: choose the action with the best mean expected position
    mean_positions = {
        a: float(np.mean([r["expected_position"] for r in results]))
        for a, results in action_results.items() if results
    }
    if not mean_positions:
        mean_positions = {0: 10.0}
    best_action = min(mean_positions, key=lambda a: mean_positions[a])
    expected_position = mean_positions[best_action]

    def _mean_field(results: list[dict[str, Any]], field: str, default: float = 0.0) -> float:
        vals = [r.get(field, default) for r in results]
        return float(np.mean(vals)) if vals else default

    def _mean_finish_dist(results: list[dict[str, Any]]) -> dict[int, float]:
        dists = [r.get("finish_prob_by_position") or {} for r in results]
        dists = [d for d in dists if d]
        if not dists:
            return {}
        positions = sorted({p for d in dists for p in d})
        return {p: round(float(np.mean([d.get(p, 0.0) for d in dists])), 4) for p in positions}

    best_results = action_results.get(best_action, [])
    best_podium_prob = _mean_field(best_results, "podium_prob")
    best_win_prob = _mean_field(best_results, "win_prob")
    best_finish_dist = _mean_finish_dist(best_results)
    best_expected_time = _mean_field(best_results, "expected_time")

    # Baseline: stay-out expected position (action 0), reusing whatever
    # roll-outs we already collected for it rather than a fresh MC call.
    stay_out_results = action_results.get(0, [])
    baseline_pos = (
        _mean_field(stay_out_results, "expected_position", expected_position)
        if stay_out_results else expected_position
    )
    expected_gain = round(baseline_pos - expected_position, 2)

    candidates = [
        {
            **_action_meta(a, state.lap),
            "expected_position": round(float(np.mean([r["expected_position"] for r in results])), 2),
            "win_prob": round(_mean_field(results, "win_prob"), 4),
            "podium_prob": round(_mean_field(results, "podium_prob"), 4),
            "finish_prob_by_position": _mean_finish_dist(results),
            "expected_time": round(_mean_field(results, "expected_time"), 2),
        }
        for a, results in action_results.items() if results
    ]

    # Map best_action → strategy fields
    current_comp, current_age = state.tyres.get(target_driver, ("MEDIUM", 1))
    if best_action == 0:
        action_str = "STAY OUT"
        tyre = current_comp
        pit_lap = None
        reasoning = (
            f"Greedy Monte Carlo (no trained RL policy) recommends staying out on {current_comp} (age {current_age}). "
            f"Expected finish: P{expected_position:.1f}."
        )
    else:
        tyre = _COMPOUNDS[best_action - 1]
        pit_lap = state.lap + 1
        action_str = f"BOX THIS LAP"
        reasoning = (
            f"Greedy Monte Carlo (no trained RL policy) recommends pitting for {tyre} on lap {pit_lap}. "
            f"Expected finish: P{expected_position:.1f} "
            f"({expected_gain:+.1f} vs stay-out)."
        )

    # Confidence: fraction of episodes whose own choice agreed with best_action
    confidence = round(
        sum(1 for c in episode_choices if c == best_action) / max(len(episode_choices), 1), 2
    )
    if confidence == 0.0:
        confidence = 0.65

    return {
        "target_driver": target_driver,
        "action": action_str,
        "tyre": tyre,
        "pit_lap": pit_lap,
        "expected_gain": expected_gain,
        "confidence": confidence,
        "expected_position": round(expected_position, 1),
        "win_prob": round(best_win_prob, 4),
        "podium_prob": round(best_podium_prob, 4),
        "finish_prob_by_position": best_finish_dist,
        "expected_time": round(best_expected_time, 2),
        "reasoning": reasoning,
        "engine": "rl",
        "policy_provided": False,
        "candidates": candidates,
    }
