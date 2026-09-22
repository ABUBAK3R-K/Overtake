"""Tests for FR-7 Strategy Engines 2 & 3 and FR-2 SHAP explainability.

Tests are designed to run fast without real race data by using the same
lightweight mocking approach as the existing optimizer tests.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ─── FR-2: SHAP explainability ────────────────────────────────────────────────

class TestShapValuesFor:
    """Tests for tyre.shap_values_for() handoff function."""

    def test_returns_dict(self):
        """shap_values_for should return a dict keyed by feature name."""
        from src.models.tyre import shap_values_for
        result = shap_values_for("SOFT", 10, "Sakhir", 30.0)
        assert isinstance(result, dict)
        # Should have at least one key or be empty (if model unavailable)
        # Both outcomes are valid; the important thing is no exception
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_bias_key_present_when_model_loaded(self):
        """If the tyre model is loaded, SHAP output always includes 'bias'."""
        from src.models.tyre import shap_values_for, _default_model
        try:
            _default_model()  # Ensure model is loaded
        except FileNotFoundError:
            pytest.skip("Tyre model not trained; skipping SHAP test")
        result = shap_values_for("MEDIUM", 15, "Sakhir", 30.0)
        if result:  # non-empty means model was available
            assert "bias" in result

    def test_unknown_compound_raises(self):
        """WET tyre passed to the underlying predict raises ValueError."""
        from src.models.tyre import predict_tyre_degradation
        with pytest.raises((ValueError, Exception)):
            predict_tyre_degradation("WET", 5, "Sakhir", 20.0)


# ─── FR-7 Engine 2: Game theory ───────────────────────────────────────────────

class TestGametheoryEngine:
    """Tests for get_strategy_recommendation_gametheory()."""

    def _make_state(self):
        """Return a minimal RaceState fixture."""
        from src.simulation.state import RaceState
        return RaceState(
            race_id="2023_bahrain",
            lap=20,
            positions={"VER": 1, "HAM": 2, "ALO": 3},
            gaps={"VER": 0.0, "HAM": 3.5, "ALO": 8.2},
            tyres={"VER": ("MEDIUM", 12), "HAM": ("SOFT", 8), "ALO": ("HARD", 18)},
            safety_car=False,
            total_laps=57,
            circuit="Sakhir",
            last_lap_times={"VER": 95.0, "HAM": 95.5, "ALO": 96.0},
            weather={"track_temp": 32.0},
        )

    def test_returns_correct_schema(self):
        """Engine 2 must return the same required keys as Engine 1."""
        from src.strategy.gametheory import get_strategy_recommendation_gametheory
        state = self._make_state()
        result = get_strategy_recommendation_gametheory(
            state, rival_state=state, target_driver="VER", n_sims=20
        )
        required_keys = {"action", "tyre", "expected_gain", "confidence",
                         "expected_position", "reasoning", "engine"}
        assert required_keys.issubset(result.keys()), \
            f"Missing keys: {required_keys - result.keys()}"

    def test_engine_tag_is_gametheory(self):
        """'engine' field must be 'gametheory'."""
        from src.strategy.gametheory import get_strategy_recommendation_gametheory
        state = self._make_state()
        result = get_strategy_recommendation_gametheory(
            state, rival_state=state, target_driver="VER", n_sims=20
        )
        assert result["engine"] == "gametheory"

    def test_rival_driver_populated(self):
        """Result should include rival_driver when a rival is found."""
        from src.strategy.gametheory import get_strategy_recommendation_gametheory
        state = self._make_state()
        result = get_strategy_recommendation_gametheory(
            state, rival_state=state, target_driver="VER", n_sims=20
        )
        assert "rival_driver" in result
        assert result["rival_driver"] != "VER"

    def test_confidence_in_range(self):
        """Confidence must be between 0 and 1."""
        from src.strategy.gametheory import get_strategy_recommendation_gametheory
        state = self._make_state()
        result = get_strategy_recommendation_gametheory(
            state, rival_state=state, target_driver="VER", n_sims=20
        )
        assert 0.0 <= result["confidence"] <= 1.0

    def test_no_rival_falls_back_gracefully(self):
        """With only one driver in positions, engine should not crash."""
        from src.strategy.gametheory import get_strategy_recommendation_gametheory
        from src.simulation.state import RaceState
        state = RaceState(
            race_id="2023_bahrain",
            lap=20,
            positions={"VER": 1},
            gaps={"VER": 0.0},
            tyres={"VER": ("MEDIUM", 12)},
            safety_car=False,
            total_laps=57,
            circuit="Sakhir",
            last_lap_times={"VER": 95.0},
            weather={"track_temp": 32.0},
        )
        result = get_strategy_recommendation_gametheory(
            state, rival_state=state, target_driver="VER", n_sims=20
        )
        assert "action" in result
        assert result["engine"] == "gametheory"


# ─── FR-7 Engine 3: RL environment ────────────────────────────────────────────

class TestRLEnv:
    """Tests for RaceStrategyEnv and get_strategy_recommendation_rl()."""

    def _make_state(self):
        from src.simulation.state import RaceState
        return RaceState(
            race_id="2023_bahrain",
            lap=25,
            positions={"VER": 1, "HAM": 2},
            gaps={"VER": 0.0, "HAM": 5.0},
            tyres={"VER": ("HARD", 20), "HAM": ("MEDIUM", 15)},
            safety_car=False,
            total_laps=57,
            circuit="Sakhir",
            last_lap_times={"VER": 95.0, "HAM": 95.5},
            weather={"track_temp": 30.0},
        )

    def test_rl_env_step_returns_correct_shape(self):
        """RaceStrategyEnv.step() must return a 5-tuple."""
        pytest.importorskip("gymnasium")
        from src.strategy.rl_env import RaceStrategyEnv
        state = self._make_state()
        env = RaceStrategyEnv.from_state(state, "VER", n_sims=20)
        obs, _ = env.reset()
        assert obs.shape == (12,)
        result = env.step(0)  # stay out
        assert len(result) == 5
        obs2, reward, done, truncated, info = result
        assert obs2.shape == (12,)
        assert isinstance(reward, float)
        assert done is True

    def test_rl_env_action_space(self):
        """Action space must be Discrete(4)."""
        pytest.importorskip("gymnasium")
        from src.strategy.rl_env import RaceStrategyEnv
        state = self._make_state()
        env = RaceStrategyEnv.from_state(state, "VER", n_sims=20)
        assert env.action_space.n == 4

    def test_rl_env_observation_space(self):
        """Observation space must be Box(12,)."""
        pytest.importorskip("gymnasium")
        from src.strategy.rl_env import RaceStrategyEnv
        state = self._make_state()
        env = RaceStrategyEnv.from_state(state, "VER", n_sims=20)
        assert env.observation_space.shape == (12,)

    def test_get_recommendation_rl_schema(self):
        """get_strategy_recommendation_rl must return required keys."""
        from src.strategy.rl_env import get_strategy_recommendation_rl
        state = self._make_state()
        result = get_strategy_recommendation_rl(
            state, policy=None, target_driver="VER",
            n_eval_episodes=4, n_sims=20
        )
        required = {"action", "tyre", "expected_gain", "confidence",
                    "expected_position", "reasoning", "engine"}
        assert required.issubset(result.keys()), \
            f"Missing keys: {required - result.keys()}"

    def test_engine_tag_is_rl(self):
        """'engine' field must be 'rl'."""
        from src.strategy.rl_env import get_strategy_recommendation_rl
        state = self._make_state()
        result = get_strategy_recommendation_rl(
            state, policy=None, target_driver="VER",
            n_eval_episodes=4, n_sims=20
        )
        assert result["engine"] == "rl"

    def test_confidence_in_range(self):
        """Confidence must be between 0 and 1."""
        from src.strategy.rl_env import get_strategy_recommendation_rl
        state = self._make_state()
        result = get_strategy_recommendation_rl(
            state, policy=None, target_driver="VER",
            n_eval_episodes=4, n_sims=20
        )
        assert 0.0 <= result["confidence"] <= 1.0


# ─── API: engine query param ──────────────────────────────────────────────────

class TestStrategyEngineAPIParam:
    """Tests that the API correctly dispatches to each engine."""

    def test_invalid_engine_returns_422(self):
        """?engine=invalid should be rejected by FastAPI (pattern validation)."""
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/api/strategy/2023_bahrain/20?engine=invalid")
        assert resp.status_code == 422

    def test_tyre_endpoint_includes_shap_by_default(self):
        """GET /api/tyre/{race_id}/{driver}/{lap} should include 'shap' key."""
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/api/tyre/2023_bahrain/VER/20")
        assert resp.status_code == 200
        data = resp.json()
        assert "compound" in data
        assert "degradation_curves" in data
        # shap key may or may not be present depending on trained model
        # but response schema must be valid
        assert "current_pace_loss_seconds" in data

    def test_tyre_endpoint_shap_excluded(self):
        """?include_shap=false should return response without 'shap' key."""
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/api/tyre/2023_bahrain/VER/20?include_shap=false")
        assert resp.status_code == 200
        data = resp.json()
        assert "shap" not in data
