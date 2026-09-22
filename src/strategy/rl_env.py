"""Strategy Engine 3 — Reinforcement Learning Policy (Design.md Section 6.10 / FR-7).

Provides:

  * ``RaceStrategyEnv``  — a gymnasium.Env wrapping the Monte Carlo / replay
    state machine. Actions are discrete: 0 = stay out, 1..N = pit for compound.
  * ``get_strategy_recommendation_rl(state, policy)``  — evaluates a trained
    policy over N roll-outs and returns the same
    ``{action, tyre, pit_lap, expected_gain, confidence, engine, ...}`` shape as
    the other two engines so the API layer can render it identically.

The environment is self-contained and does NOT require live FastF1 data — it
rebuilds observation vectors from the frozen ``RaceState`` snapshot, exactly as
the Monte Carlo simulator does. This keeps the no-leakage guarantee intact.

Training:
    env = RaceStrategyEnv.from_state(state, target_driver=driver)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50_000)
    rec = get_strategy_recommendation_rl(state, policy=model)
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

# ─── Gymnasium Environment ────────────────────────────────────────────────────


def _make_observation(state: RaceState, driver: str) -> np.ndarray:
    """Build a fixed-length float32 observation vector from a RaceState.

    Features (12 dims):
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
       11  n_drivers / 20.0                  (field size, usually 1)
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
        n_drivers / 20.0,
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
            # Observation: 12 normalised floats
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(12,), dtype=np.float32
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
                result = run_monte_carlo(
                    state=state,
                    strategy=strategy,
                    target_driver=driver,
                    n_sims=self._n_sims,
                )
                expected_pos = result["expected_position"]
            except Exception as exc:
                log.warning("MC error in env step: %s", exc)
                expected_pos = 10.0  # Penalty for failed sim

            # Reward: negative expected position so lower pos → higher reward
            reward = -float(expected_pos)
            obs = _make_observation(state, driver)
            return obs, reward, True, False, {"expected_position": expected_pos, "strategy": strategy}

        def render(self) -> None:
            pass


# ─── Public handoff function ──────────────────────────────────────────────────


def get_strategy_recommendation_rl(
    state: RaceState,
    policy: Any | None = None,
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
        policy:           A trained stable-baselines3 model (e.g. PPO or DQN)
                          that implements ``.predict(obs)``. If None, falls back
                          to a greedy MC policy across all 4 actions.
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

    # Build env for evaluation
    env = RaceStrategyEnv.from_state(state, target_driver, n_sims=n_sims)
    obs, _ = env.reset()

    # Collect action scores across episodes
    action_rewards: dict[int, list[float]] = {a: [] for a in range(4)}

    for _ in range(max(n_eval_episodes, 4)):
        if policy is not None:
            try:
                action_arr, _ = policy.predict(obs, deterministic=True)
                action = int(action_arr)
            except Exception:
                action = 0
        else:
            # Greedy: try each action, pick best (no trained policy provided)
            best_action, best_reward = 0, float("-inf")
            for a in range(4):
                _, reward, _, _, _ = env.step(a)
                if reward > best_reward:
                    best_reward, best_action = reward, a
                obs, _ = env.reset()
            action = best_action

        _, reward, _, _, _ = env.step(action)
        action_rewards[action].append(reward)
        obs, _ = env.reset()

    # Aggregate: choose action with best mean reward
    mean_rewards = {
        a: float(np.mean(v)) if v else float("-inf")
        for a, v in action_rewards.items()
    }
    best_action = max(mean_rewards, key=lambda a: mean_rewards[a])
    best_mean_reward = mean_rewards[best_action]

    expected_position = -best_mean_reward  # reward = -expected_pos

    # Baseline: stay-out expected position
    try:
        baseline_res = run_monte_carlo(
            state=state,
            strategy={"name": "STAY_OUT", "pit_laps": [], "compounds": []},
            target_driver=target_driver,
            n_sims=n_sims,
        )
        baseline_pos = baseline_res["expected_position"]
    except Exception:
        baseline_pos = expected_position

    expected_gain = round(baseline_pos - expected_position, 2)

    # Map best_action → strategy fields
    current_comp, current_age = state.tyres.get(target_driver, ("MEDIUM", 1))
    if best_action == 0:
        action_str = "STAY OUT"
        tyre = current_comp
        pit_lap = None
        reasoning = (
            f"RL policy recommends staying out on {current_comp} (age {current_age}). "
            f"Expected finish: P{expected_position:.1f}."
        )
    else:
        tyre = _COMPOUNDS[best_action - 1]
        pit_lap = state.lap + 1
        action_str = f"BOX THIS LAP"
        reasoning = (
            f"RL policy recommends pitting for {tyre} on lap {pit_lap}. "
            f"Expected finish: P{expected_position:.1f} "
            f"({expected_gain:+.1f} vs stay-out)."
        )

    # Confidence: fraction of episodes that agreed on best action
    total_eps = sum(len(v) for v in action_rewards.values())
    best_count = len(action_rewards[best_action])
    confidence = round(best_count / max(total_eps, 1), 2)
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
        "reasoning": reasoning,
        "engine": "rl",
        "policy_provided": policy is not None,
    }
