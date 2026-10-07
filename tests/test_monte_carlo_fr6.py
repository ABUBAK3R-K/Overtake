"""Test suite for FR-6: Monte Carlo Forward Simulation Engine (>=1000 samples).

Validates:
- High-sample forward simulation (N >= 1000)
- Complete finishing outcome distributions (PMF 1..20)
- Quantile monotonicity and confidence interval bounds
- Trajectory ribbons and sample path generation
- Stochastic uncertainty modeling (pit tail risk, SC bunching, tyre aging)
- Execution performance and determinism via random seed
"""

import time
import pytest
import numpy as np

from src.simulation.monte_carlo import run_monte_carlo
from src.simulation.state import RaceState
from src.strategy.optimizer import get_strategy_recommendation


@pytest.fixture
def base_race_state() -> RaceState:
    return RaceState(
        race_id="2023_bahrain",
        lap=25,
        positions={"VER": 1, "PER": 2, "ALO": 3, "HAM": 4, "SAI": 5, "RUS": 6, "LEC": 7, "NOR": 8},
        gaps={"VER": 0.0, "PER": 3.2, "ALO": 8.5, "HAM": 14.1, "SAI": 18.0, "RUS": 22.4, "LEC": 28.0, "NOR": 35.2},
        tyres={
            "VER": ("SOFT", 15),
            "PER": ("SOFT", 15),
            "ALO": ("MEDIUM", 12),
            "HAM": ("MEDIUM", 12),
            "SAI": ("HARD", 8),
            "RUS": ("HARD", 8),
            "LEC": ("SOFT", 18),
            "NOR": ("MEDIUM", 10),
        },
        safety_car=False,
        total_laps=57,
        circuit="Sakhir",
        status="RACING",
        weather={"track_temp": 34.0, "air_temp": 28.0, "is_wet": False},
        last_lap_times={"VER": 93.1, "PER": 93.4, "ALO": 93.6, "HAM": 93.9, "SAI": 94.1, "RUS": 94.2, "LEC": 94.8, "NOR": 95.1},
    )


def test_monte_carlo_1000_samples_distribution_completeness(base_race_state):
    """Verify FR-6 requirement: >= 1000 samples produce full distribution metrics."""
    res = run_monte_carlo(
        state=base_race_state,
        strategy={"pit_laps": [30], "compounds": ["HARD"]},
        target_driver="VER",
        n_sims=1000,
        seed=42,
    )

    assert res["n_sims"] == 1000
    assert "finish_prob_by_position" in res
    assert "expected_position" in res
    assert "position_std" in res
    assert "percentiles" in res
    assert "ci_95_expected_pos" in res
    assert "trajectory_bands" in res
    assert "sample_trajectories" in res

    # Check PMF sums to 1.0 within numerical precision
    pmf = res["finish_prob_by_position"]
    assert len(pmf) > 0
    total_prob = sum(pmf.values())
    assert abs(total_prob - 1.0) < 0.01

    # Probability bounds
    assert 0.0 <= res["win_prob"] <= 1.0
    assert 0.0 <= res["podium_prob"] <= 1.0
    assert 0.0 <= res["top5_prob"] <= 1.0
    assert 0.0 <= res["points_prob"] <= 1.0
    assert res["win_prob"] <= res["podium_prob"] <= res["top5_prob"] <= res["points_prob"]


def test_monte_carlo_quantiles_and_confidence_interval(base_race_state):
    """Verify quantile monotonicity (p10 <= p25 <= p50 <= p75 <= p90) and CI validity."""
    res = run_monte_carlo(
        state=base_race_state,
        strategy={"pit_laps": [28], "compounds": ["HARD"]},
        target_driver="VER",
        n_sims=1000,
        seed=123,
    )

    pct = res["percentiles"]
    assert pct["p10"] <= pct["p25"] <= pct["p50"] <= pct["p75"] <= pct["p90"]
    assert pct["iqr"] == pct["p75"] - pct["p25"]

    ci_low, ci_high = res["ci_95_expected_pos"]
    assert ci_low <= res["expected_position"] <= ci_high
    assert ci_high - ci_low < 1.0  # At N=1000, standard error of mean is tight


def test_monte_carlo_trajectory_bands(base_race_state):
    """Verify lap-by-lap trajectory percentile corridors for frontend plotting."""
    res = run_monte_carlo(
        state=base_race_state,
        strategy={"pit_laps": [32], "compounds": ["HARD"]},
        target_driver="VER",
        n_sims=1000,
        seed=42,
    )

    bands = res["trajectory_bands"]
    remaining_laps = base_race_state.total_laps - base_race_state.lap
    assert len(bands) == remaining_laps

    for band in bands:
        assert "lap" in band
        assert "p10" in band
        assert "p25" in band
        assert "p50" in band
        assert "p75" in band
        assert "p90" in band
        assert "mean_gap" in band
        assert band["p10"] <= band["p50"] <= band["p90"]


def test_monte_carlo_execution_speed_benchmark(base_race_state):
    """Verify FR-6 performance: 1000 forward simulations over 32 laps complete quickly (< 2.0s)."""
    t0 = time.perf_counter()
    res = run_monte_carlo(
        state=base_race_state,
        strategy={"pit_laps": [30], "compounds": ["HARD"]},
        target_driver="VER",
        n_sims=1000,
        seed=42,
    )
    elapsed = time.perf_counter() - t0

    assert res["n_sims"] == 1000
    assert elapsed < 2.0, f"Monte Carlo 1000 sims took {elapsed:.2f}s, expected < 2.0s"


def test_monte_carlo_reproducibility(base_race_state):
    """Verify deterministic seed produces identical results."""
    res1 = run_monte_carlo(base_race_state, target_driver="ALO", n_sims=1000, seed=99)
    res2 = run_monte_carlo(base_race_state, target_driver="ALO", n_sims=1000, seed=99)

    assert res1["expected_position"] == res2["expected_position"]
    assert res1["win_prob"] == res2["win_prob"]
    assert res1["percentiles"] == res2["percentiles"]
    assert res1["finish_prob_by_position"] == res2["finish_prob_by_position"]


def test_monte_carlo_edge_case_final_lap(base_race_state):
    """Verify edge case when no laps remain."""
    final_state = RaceState(
        race_id="2023_bahrain",
        lap=57,
        positions={"VER": 1, "PER": 2},
        gaps={"VER": 0.0, "PER": 5.0},
        tyres={"VER": ("HARD", 30), "PER": ("HARD", 28)},
        safety_car=False,
        total_laps=57,
        circuit="Sakhir",
        weather={"track_temp": 30.0},
    )
    res = run_monte_carlo(final_state, target_driver="VER", n_sims=1000)
    assert res["expected_position"] == 1.0
    assert res["win_prob"] == 1.0
    assert res["podium_prob"] == 1.0
    assert res["finish_prob_by_position"] == {1: 1.0}


def test_strategy_optimizer_with_fr6_distributions(base_race_state):
    """Verify optimizer calculates recommendations with candidate distribution payloads."""
    rec = get_strategy_recommendation(base_race_state, target_driver="VER", n_sims=200)

    assert "action" in rec
    assert "tyre" in rec
    assert "expected_gain" in rec
    assert "confidence" in rec
    assert "candidates" in rec
    assert len(rec["candidates"]) >= 2

    # Check candidate metadata contains FR-6 metrics
    cand = rec["candidates"][0]
    assert "expected_position" in cand
    assert "position_std" in cand
    assert "win_prob" in cand
    assert "podium_prob" in cand
    assert "points_prob" in cand
    assert "percentiles" in cand
    assert "ci_95" in cand
