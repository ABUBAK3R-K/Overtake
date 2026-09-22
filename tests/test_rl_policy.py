"""Tests for the trained RL strategy engine (src/strategy/rl_policy.py) and
its use in get_strategy_recommendation_rl (FR-7 Engine 3)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from src.simulation.state import RaceState  # noqa: E402
from src.strategy import rl_policy  # noqa: E402
from src.strategy.rl_env import OBS_DIM, get_strategy_recommendation_rl  # noqa: E402
from src.strategy.rl_policy import CachedStrategyEnv, evaluate  # noqa: E402


def _record(obs0: float, expected: list[float], race_id: str = "2099_test") -> dict:
    """A synthetic dataset row: finishing distribution is a point mass at the
    expected position, so rewards are deterministic."""
    obs = [0.5] * OBS_DIM
    obs[0] = obs0
    return {
        "race_id": race_id, "driver": "VER", "lap": 10,
        "obs": obs, "expected_position": expected,
        "finish_dist": [[[int(e), 1.0]] for e in expected],
    }


def _state() -> RaceState:
    return RaceState(
        race_id="2023_bahrain", lap=25,
        positions={"VER": 1, "HAM": 2}, gaps={"VER": 0.0, "HAM": 5.0},
        tyres={"VER": ("HARD", 20), "HAM": ("MEDIUM", 15)}, safety_car=False,
        total_laps=57, circuit="Sakhir",
        last_lap_times={"VER": 95.0, "HAM": 95.5}, weather={"track_temp": 30.0},
    )


def test_cached_env_reward_is_positions_gained_vs_stay_out():
    env = CachedStrategyEnv([_record(0.2, [8, 5, 6, 7])], seed=0)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    _, r_stay, done, _, _ = env.step(0)
    _, r_hard, _, _, info = env.step(1)
    assert done is True
    assert r_stay == 0.0
    assert r_hard == 3.0 and info["finish_position"] == 5.0


def test_cached_env_requires_records():
    with pytest.raises(ValueError):
        CachedStrategyEnv([])


def test_evaluate_regret_zero_for_oracle_and_positive_otherwise():
    recs = [_record(0.1, [3, 5, 5, 5]), _record(0.9, [9, 5, 4, 6])]
    oracle = np.array([0, 2])
    rep = evaluate(oracle, recs, recs)
    assert rep["rl_policy"]["mean_regret"] == 0.0
    assert rep["rl_policy"]["picked_best_pct"] == 100.0
    assert rep["always_stay_out"]["mean_regret"] == pytest.approx(2.5)


def test_policy_learns_state_dependent_action():
    """Early in the race pitting is bad; late, pitting for SOFT wins. PPO must
    learn to condition on the observation, not pick one fixed action."""
    recs = [_record(0.1 + 0.01 * i, [4, 9, 9, 9]) for i in range(10)]
    recs += [_record(0.8 + 0.01 * i, [9, 7, 6, 3]) for i in range(10)]
    model = rl_policy.train_policy(recs, timesteps=12_000, seed=0)
    actions = rl_policy.policy_actions(model, recs)
    assert (actions[:10] == 0).mean() >= 0.9
    assert (actions[10:] == 3).mean() >= 0.9
    probs = rl_policy.action_probabilities(model, np.asarray(recs[0]["obs"], dtype=np.float32))
    assert probs.shape == (4,) and probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_split_holds_out_frozen_test_races():
    held = sorted(rl_policy.held_out_races())
    recs = [_record(0.5, [5, 5, 5, 5], race_id=held[0]), _record(0.5, [5, 5, 5, 5], race_id="2023_bahrain")]
    train, test = rl_policy.split_records(recs)
    assert [r["race_id"] for r in train] == ["2023_bahrain"]
    assert [r["race_id"] for r in test] == [held[0]]


def test_load_policy_none_when_untrained(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_policy, "POLICY_PATH", tmp_path / "missing.zip")
    rl_policy.load_policy.cache_clear()
    try:
        assert rl_policy.load_policy() is None
    finally:
        rl_policy.load_policy.cache_clear()


class _StubPolicy:
    """Only .predict(), like a non-actor-critic SB3 model."""

    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, obs, deterministic=True):
        return np.array(self.action), None


def test_recommendation_uses_policy_action_and_reports_all_candidates():
    rec = get_strategy_recommendation_rl(_state(), policy=_StubPolicy(2), target_driver="VER", n_sims=20)
    assert rec["engine"] == "rl" and rec["policy_provided"] is True
    assert rec["action"] == "BOX THIS LAP" and rec["tyre"] == "MEDIUM" and rec["pit_lap"] == 26
    assert len(rec["candidates"]) == 4
    chosen = next(c for c in rec["candidates"] if c["name"] == "PIT_MEDIUM")
    assert rec["expected_position"] == pytest.approx(chosen["expected_position"], abs=0.051)
    assert 0.0 <= rec["confidence"] <= 1.0
    for key in ("win_prob", "podium_prob", "finish_prob_by_position"):
        assert key in rec


def test_explicit_none_policy_is_greedy_fallback():
    rec = get_strategy_recommendation_rl(_state(), policy=None, target_driver="VER",
                                         n_eval_episodes=4, n_sims=20)
    assert rec["policy_provided"] is False
    assert "no trained RL policy" in rec["reasoning"]
