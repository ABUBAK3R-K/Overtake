"""Unit tests for the backtesting and historical evaluation suite."""

import pytest
from src.evaluation.backtest import (
    HISTORICAL_BENCHMARKS,
    backtest_decision_point,
    run_full_backtest,
    summarize_backtest,
)


def test_historical_benchmarks_exist():
    assert len(HISTORICAL_BENCHMARKS) >= 5
    for bm in HISTORICAL_BENCHMARKS:
        assert "race_id" in bm
        assert "driver" in bm
        assert "decision_lap" in bm
        assert "actual_action" in bm


def test_backtest_decision_point_returns_schema():
    res = backtest_decision_point(
        race_id="2023_bahrain",
        driver="ALO",
        decision_lap=14,
        n_sims=30,
    )
    assert res["race_id"] == "2023_bahrain"
    assert res["driver"] == "ALO"
    assert "ai_action" in res
    assert "verdict" in res
    assert "position_delta" in res
    assert "confidence" in res


def test_summarize_backtest():
    mock_results = [
        {"position_delta": 1.0, "verdict": "AI BEAT REAL STRATEGY (+1 POS)"},
        {"position_delta": 0.0, "verdict": "MATCHED REAL STRATEGY"},
        {"position_delta": 0.5, "verdict": "AI ADVANTAGE (+0.5 POS)"},
    ]
    summary = summarize_backtest(mock_results)
    assert summary["total_evaluations"] == 3
    assert summary["strategies_improved"] == 2
    assert summary["strategies_matched"] == 1
    assert summary["success_rate_pct"] == 100.0
