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

