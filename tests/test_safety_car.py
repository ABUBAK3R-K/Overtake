"""Unit and integration tests for Probabilistic Safety Car Model & Field Bunching (PRD FR-5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.monte_carlo import run_monte_carlo
from src.simulation.safety_car import (
    CALIBRATED_CIRCUITS,
    CIRCUIT_ALIASES,
    GLOBAL_AVERAGES,
    apply_safety_car_bunching,
    get_circuit_safety_car_profile,
    resolve_circuit_name,
    safety_car_probability,
    sample_safety_car_duration,
)
from src.simulation.state import RaceState


def test_calibrated_rates_data_integrity():
    """Verify calibration JSON loaded cleanly with >30 circuits and realistic globals."""
    assert len(CALIBRATED_CIRCUITS) >= 30
    assert "global_averages" in repr(GLOBAL_AVERAGES) or "sc_prob_per_lap" in GLOBAL_AVERAGES
    assert 0.010 <= GLOBAL_AVERAGES["sc_prob_per_lap"] <= 0.025
    assert 0.004 <= GLOBAL_AVERAGES["vsc_prob_per_lap"] <= 0.015
    assert GLOBAL_AVERAGES["any_sc_prob_per_lap"] > GLOBAL_AVERAGES["sc_prob_per_lap"]


def test_circuit_alias_resolution():
    """Verify alias mapping correctly resolves informal/regional track names."""
    assert resolve_circuit_name("bahrain") == "Sakhir"
    assert resolve_circuit_name("SAKHIR") == "Sakhir"
    assert resolve_circuit_name("singapore") == "Marina Bay"
    assert resolve_circuit_name("albert park") == "Melbourne"
    assert resolve_circuit_name("interlagos") == "São Paulo"
    assert resolve_circuit_name("monte carlo") == "Monaco"
    assert resolve_circuit_name("cota") == "Austin"
    assert resolve_circuit_name("red bull ring") == "Spielberg"
    assert resolve_circuit_name("spa") == "Spa-Francorchamps"
    # Unseen circuit returns stripped original name
    assert resolve_circuit_name("BrandNewCircuit") == "BrandNewCircuit"


def test_safety_car_probability_bounds_and_phases():
    """Verify probabilities adhere to [0.005, 0.25] and reflect race phase dynamics."""
    # Start of race (lap fraction <= 0.08) should have higher risk than early stint rhythm (0.20)
    p_start = safety_car_probability("Melbourne", 0.04)
    p_early = safety_car_probability("Melbourne", 0.25)
    p_late = safety_car_probability("Melbourne", 0.85)

    assert 0.005 <= p_start <= 0.25
    assert 0.005 <= p_early <= 0.25
    assert 0.005 <= p_late <= 0.25

    # Start incident multiplier (1.9x) > early stint (0.85x)
    assert p_start > p_early
    # Late race fatigue/battles (1.35x) > early stint (0.85x)
    assert p_late > p_early


def test_safety_car_probability_event_types():
    """Verify event-type queries (SAFETY_CAR vs VIRTUAL_SAFETY_CAR vs ANY)."""
    p_sc = safety_car_probability("Marina Bay", 0.50, event_type="SAFETY_CAR")
    p_vsc = safety_car_probability("Marina Bay", 0.50, event_type="VIRTUAL_SAFETY_CAR")
    p_any = safety_car_probability("Marina Bay", 0.50, event_type="ANY")

    assert p_sc > 0
    assert p_vsc > 0
    # Any SC probability should exceed individual event type probabilities
    assert p_any >= p_sc
    assert p_any >= p_vsc


def test_circuit_relative_risks():
    """Verify street and high-incident tracks carry higher empirical risk than low-incident tracks."""
    p_melbourne = safety_car_probability("Melbourne", 0.50)
    p_marina_bay = safety_car_probability("Marina Bay", 0.50)
    p_monaco = safety_car_probability("Monaco", 0.50)
    p_barcelona = safety_car_probability("Barcelona", 0.50)
    p_sakhir = safety_car_probability("Sakhir", 0.50)

    # Street circuits are noticeably higher risk than permanent open runoff tracks
    assert p_melbourne > p_barcelona
    assert p_marina_bay > p_barcelona
    assert p_monaco > p_barcelona
    assert p_monaco > p_sakhir


def test_sample_safety_car_duration():
    """Verify sampled durations conform to empirical multi-lap boundaries."""
    np.random.seed(42)
    sc_durations = [sample_safety_car_duration("Monaco", event_type="SAFETY_CAR") for _ in range(200)]
    vsc_durations = [sample_safety_car_duration("Monaco", event_type="VIRTUAL_SAFETY_CAR") for _ in range(200)]

    # SC duration: min 3, max 7, mean around 4.5
    assert all(3 <= d <= 7 for d in sc_durations)
    assert 4.0 <= np.mean(sc_durations) <= 5.0

    # VSC duration: min 1, max 4, mean around 2.2
    assert all(1 <= d <= 4 for d in vsc_durations)
    assert 1.8 <= np.mean(vsc_durations) <= 2.8


def test_get_circuit_safety_car_profile():
    """Verify circuit safety car profile output contains complete schema."""
    profile = get_circuit_safety_car_profile("Baku")
    assert profile["circuit"] == "Baku"
    assert profile["n_races"] > 0
    assert profile["total_laps"] > 0
    assert "mean_sc_per_race" in profile
    assert "mean_vsc_per_race" in profile
    assert "sc_prob_per_lap" in profile
    assert "vsc_prob_per_lap" in profile
    assert "any_sc_prob_per_lap" in profile
    assert len(profile["phase_multipliers"]) == 4
    assert "SAFETY_CAR" in profile["durations"]


def test_apply_safety_car_bunching_with_intervals_and_retirements():
    """Verify bunching updates both gaps and intervals and isolates retired drivers."""
    initial_state = RaceState(
        race_id="2023_test",
        lap=25,
        positions={"VER": 1, "PER": 2, "ALO": 3, "HAM": 4, "SAR": 5},
        gaps={"VER": 0.0, "PER": 14.5, "ALO": 28.0, "HAM": 42.0, "SAR": 99.0},
        intervals={"VER": 0.0, "PER": 14.5, "ALO": 13.5, "HAM": 14.0, "SAR": 57.0},
        tyres={"VER": ("MEDIUM", 15), "PER": ("MEDIUM", 15), "ALO": ("HARD", 25), "HAM": ("HARD", 25), "SAR": ("HARD", 5)},
        safety_car=False,
        total_laps=57,
        circuit="Melbourne",
        status="RACING",
        retired={"SAR": "Accident Turn 3"},
    )

    bunched = apply_safety_car_bunching(initial_state, interval_spacing=0.8)

    assert bunched.safety_car is True
    assert bunched.status == "SAFETY_CAR"
    # Leader
    assert bunched.gaps["VER"] == 0.0
    assert bunched.intervals["VER"] == 0.0

    # Active cars: gaps compressed substantially
    assert bunched.gaps["PER"] < 3.0
    assert bunched.gaps["ALO"] < 5.0
    assert bunched.gaps["HAM"] < 7.0

    # Intervals between adjacent cars in queue compressed to ~0.8s (e.g. 0.4s - 1.5s)
    assert 0.3 <= bunched.intervals["PER"] <= 1.6
    assert 0.3 <= bunched.intervals["ALO"] <= 1.6
    assert 0.3 <= bunched.intervals["HAM"] <= 1.6

    # Retired car untouched
    assert bunched.gaps["SAR"] == 99.0
    assert bunched.intervals["SAR"] == 57.0


def test_monte_carlo_forward_rollout_with_sc_episodes():
    """Verify Monte Carlo simulation advances cleanly with multi-lap SC episodes."""
    state = RaceState(
        race_id="2023_test",
        lap=40,
        positions={"VER": 1, "PER": 2, "ALO": 3},
        gaps={"VER": 0.0, "PER": 5.0, "ALO": 10.0},
        tyres={"VER": ("HARD", 20), "PER": ("HARD", 20), "ALO": ("HARD", 20)},
        safety_car=True,
        total_laps=50,
        circuit="Monaco",
        status="SAFETY_CAR",
    )

    result = run_monte_carlo(
        state=state,
        strategy={"pit_laps": [42], "compounds": ["SOFT"]},
        target_driver="VER",
        n_sims=50,
        seed=123,
    )

    assert "finish_prob_by_position" in result
    assert "expected_position" in result
    assert "sample_trajectories" in result
    assert len(result["sample_trajectories"]) > 0
    assert result["win_prob"] >= 0.0
