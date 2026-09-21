"""Unit and integration tests for FastAPI backend endpoints."""

from fastapi.testclient import TestClient
import pytest

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_races_endpoint():
    response = client.get("/api/races")
    assert response.status_code == 200
    races = response.json()
    assert isinstance(races, list)
    assert len(races) > 0
    assert "race_id" in races[0]


def test_backtest_endpoints():
    response = client.get("/api/backtest/2023_bahrain")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "driver" in data[0]
        assert "verdict" in data[0]

    response_full = client.get("/api/backtest")
    assert response_full.status_code == 200
    summary = response_full.json()
    assert "total_evaluations" in summary
    assert "success_rate_pct" in summary


def test_frontend_index_serving():
    response = client.get("/")
    assert response.status_code == 200
    assert "OVERTAKE" in response.text


def test_lap_prediction_model_param_validation():
    # Invalid model name should fail with 422 Unprocessable Entity
    response = client.get("/api/lap-prediction/2023_bahrain/VER/15?model=invalid_model")
    assert response.status_code == 422


def test_replay_summary_and_events_endpoints():
    response = client.get("/api/replay/2023_bahrain/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["race_id"] == "2023_bahrain"
    assert "total_laps" in data
    assert "circuit" in data

    response_events = client.get("/api/replay/2023_bahrain/events/15")
    assert response_events.status_code == 200
    events = response_events.json()
    assert events["race_id"] == "2023_bahrain"
    assert events["lap"] == 15
    assert "pit_stops" in events
    assert "race_control" in events


def test_safety_car_risk_endpoint():
    # Test valid circuit with query parameters
    response = client.get("/api/safety-car/Monaco?lap_fraction=0.05&event_type=SAFETY_CAR")
    assert response.status_code == 200
    data = response.json()
    assert data["circuit"] == "Monaco"
    assert "sc_prob_per_lap" in data
    assert "any_sc_prob_per_lap" in data
    assert "phase_multipliers" in data
    assert "durations" in data
    assert data["queried_lap_fraction"] == 0.05
    assert data["queried_event_type"] == "SAFETY_CAR"
    assert data["current_lap_probability"] > 0.0

    # Test alias resolution (e.g. 'singapore' -> 'Marina Bay')
    response_alias = client.get("/api/safety-car/singapore")
    assert response_alias.status_code == 200
    data_alias = response_alias.json()
    assert data_alias["circuit"] == "Marina Bay"
    assert data_alias["n_races"] > 0

    # Test unknown circuit graceful fallback
    response_unknown = client.get("/api/safety-car/unseen_test_circuit?lap_fraction=0.5")
    assert response_unknown.status_code == 200
    data_unknown = response_unknown.json()
    assert "any_sc_prob_per_lap" in data_unknown
    assert 0.005 <= data_unknown["current_lap_probability"] <= 0.25




