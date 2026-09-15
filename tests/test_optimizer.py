"""Unit tests for the Strategy Optimizer engine."""

import pytest
from src.simulation.state import RaceState
from src.strategy.optimizer import generate_candidate_strategies, get_strategy_recommendation


@pytest.fixture
def sample_state() -> RaceState:
    return RaceState(
        race_id="2023_bahrain",
        lap=22,
        positions={"VER": 1, "PER": 2, "ALO": 3, "HAM": 4, "SAI": 5},
        gaps={"VER": 0.0, "PER": 3.2, "ALO": 8.5, "HAM": 14.0, "SAI": 19.5},
        tyres={
            "VER": ("SOFT", 22),
            "PER": ("SOFT", 22),
            "ALO": ("MEDIUM", 12),
            "HAM": ("MEDIUM", 12),
            "SAI": ("HARD", 10),
        },
        safety_car=False,
        total_laps=57,
        circuit="Sakhir",
        status="RACING",
        weather={"track_temp": 32.0, "air_temp": 26.0, "is_wet": False},
        last_lap_times={"VER": 92.8, "PER": 93.0, "ALO": 92.5, "HAM": 92.7, "SAI": 93.1},
    )


def test_generate_candidate_strategies(sample_state):
    candidates = generate_candidate_strategies(sample_state, target_driver="VER")
    assert len(candidates) >= 3
    names = [c["name"] for c in candidates]
    assert "STAY_OUT" in names
    assert any("BOX_NOW" in n for n in names)


def test_get_strategy_recommendation_contract(sample_state):
    rec = get_strategy_recommendation(sample_state, target_driver="VER", n_sims=40)
    assert "action" in rec
    assert "tyre" in rec
    assert "expected_gain" in rec
    assert "confidence" in rec
    assert "reasoning" in rec
    assert "candidates" in rec
    assert isinstance(rec["candidates"], list)
    assert len(rec["candidates"]) > 0
    # Action should be actionable
    assert rec["action"] in ["STAY OUT", "BOX THIS LAP", "2-STOP AGGRESSIVE"] or "BOX IN" in rec["action"]
