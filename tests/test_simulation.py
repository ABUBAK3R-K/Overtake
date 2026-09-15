"""Offline unit tests for the simulation engine (RaceState, replay, SC, and Monte Carlo)."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.simulation.monte_carlo import run_monte_carlo
from src.simulation.replay import build_state_at_lap, init_state
from src.simulation.safety_car import apply_safety_car_bunching, safety_car_probability
from src.simulation.state import RaceState


@pytest.fixture
def mock_state() -> RaceState:
    return RaceState(
        race_id="2023_bahrain",
        lap=20,
        positions={"VER": 1, "PER": 2, "ALO": 3, "HAM": 4, "SAI": 5},
        gaps={"VER": 0.0, "PER": 4.5, "ALO": 12.0, "HAM": 18.5, "SAI": 24.0},
        tyres={
            "VER": ("SOFT", 20),
            "PER": ("SOFT", 20),
            "ALO": ("MEDIUM", 15),
            "HAM": ("MEDIUM", 15),
            "SAI": ("HARD", 10),
        },
        safety_car=False,
        total_laps=57,
        circuit="Sakhir",
        status="RACING",
        weather={"track_temp": 32.0, "air_temp": 26.0, "is_wet": False},
        last_lap_times={"VER": 92.5, "PER": 92.8, "ALO": 93.0, "HAM": 93.2, "SAI": 93.5},
    )


# --- RaceState serialization tests ---

def test_race_state_to_and_from_dict(mock_state):
    d = mock_state.to_dict()
    restored = RaceState.from_dict(d)
    assert restored.race_id == mock_state.race_id
    assert restored.lap == 20
    assert restored.positions["VER"] == 1
    assert restored.tyres["VER"] == ("SOFT", 20)
    assert restored.weather["track_temp"] == 32.0


# --- Safety Car model tests ---

def test_safety_car_probability_bounds():
    prob_early = safety_car_probability("Monaco", 0.05)
    prob_mid = safety_car_probability("Sakhir", 0.50)
    assert 0.005 <= prob_early <= 0.25
    assert 0.005 <= prob_mid <= 0.25
    # Monaco SC rate should be noticeably higher than Sakhir
    assert safety_car_probability("Monaco", 0.50) > safety_car_probability("Sakhir", 0.50)


def test_apply_safety_car_bunching(mock_state):
    bunched = apply_safety_car_bunching(mock_state, interval_spacing=0.8)
    assert bunched.safety_car is True
    assert bunched.status == "SAFETY_CAR"
    assert bunched.gaps["VER"] == 0.0
    # Gaps behind leader should be compressed
    assert bunched.gaps["PER"] <= mock_state.gaps["PER"]
    assert bunched.gaps["SAI"] < mock_state.gaps["SAI"]


# --- Monte Carlo forward rollout tests ---

def test_monte_carlo_produces_distribution(mock_state):
    strategy = {"pit_laps": [25], "compounds": ["HARD"]}
    res = run_monte_carlo(mock_state, strategy=strategy, target_driver="VER", n_sims=50, seed=42)

    assert "finish_prob_by_position" in res
    assert "expected_position" in res
    assert "podium_prob" in res
    assert "win_prob" in res
    assert 1.0 <= res["expected_position"] <= 5.0
    # Probabilities sum to ~1.0
    total_prob = sum(res["finish_prob_by_position"].values())
    assert abs(total_prob - 1.0) < 0.01


def test_monte_carlo_stay_out_vs_fresh_tyres(mock_state):
    # Staying out on 20-lap old softs for 37 more laps vs pitting for fresh hard tyres
    res_stay_out = run_monte_carlo(mock_state, strategy={"pit_laps": [], "compounds": []}, target_driver="VER", n_sims=60, seed=42)
    res_box = run_monte_carlo(mock_state, strategy={"pit_laps": [21], "compounds": ["HARD"]}, target_driver="VER", n_sims=60, seed=42)

    # Box for fresh hards should yield a significantly better expected finish than staying out on softs
    assert res_box["expected_position"] <= res_stay_out["expected_position"]


# --- Day 6 Prediction Integration tests ---

def test_advance_lap_with_predictions(mock_state):
    from src.simulation.replay import advance_lap_with_predictions

    # Mock prediction functions
    def dummy_pace(state, driver):
        return 91.0 if driver == "VER" else 92.5

    def dummy_deg(compound, age, circuit, temp):
        return age * 0.05

    advanced = advance_lap_with_predictions(
        mock_state,
        predict_lap_time_fn=dummy_pace,
        predict_degradation_fn=dummy_deg,
    )

    assert advanced.lap == mock_state.lap + 1
    # Tyre age increments
    assert advanced.tyres["VER"][1] == mock_state.tyres["VER"][1] + 1
    # VER had faster pace so should remain leader
    assert advanced.positions["VER"] == 1
    assert advanced.gaps["VER"] == 0.0

