"""Unit tests for the backtesting and historical evaluation suite."""

import pytest
from src.evaluation.backtest import (
    ENGINES,
    HISTORICAL_BENCHMARKS,
    _actual_outcome,
    backtest_decision_point,
    decision_points_for_race,
    run_dataset_backtest,
    run_full_backtest,
    run_multi_engine_backtest,
    summarize_backtest,
    summarize_multi_engine_backtest,
)
from src.ingestion.storage import connect, table_exists


def test_historical_benchmarks_exist():
    assert len(HISTORICAL_BENCHMARKS) >= 5
    for bm in HISTORICAL_BENCHMARKS:
        assert "race_id" in bm
        assert "driver" in bm
        assert "decision_lap" in bm
        assert "actual_action" in bm


@pytest.mark.skipif(not table_exists("races", "2023_bahrain"), reason="2023_bahrain not ingested")
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
        {"position_delta": 1.0, "verdict": "AI ADVANTAGE (+1.0 POS)", "regret_vs_hindsight": 0.0},
        {"position_delta": 0.0, "verdict": "MATCHED REAL STRATEGY", "regret_vs_hindsight": 0.2},
        {"position_delta": 0.5, "verdict": "AI ADVANTAGE (+0.5 POS)", "regret_vs_hindsight": 0.0},
        {"position_delta": -1.0, "verdict": "REAL STRATEGY BETTER (-1.0 POS)", "regret_vs_hindsight": 1.5},
        {"verdict": "EVALUATION_FAILED", "error": "boom"},
    ]
    summary = summarize_backtest(mock_results)
    assert summary["total_evaluations"] == 5
    assert summary["failed_evaluations"] == 1
    assert summary["strategies_improved"] == 2
    assert summary["strategies_matched"] == 1
    assert summary["strategies_worse"] == 1
    assert summary["success_rate_pct"] == 75.0
    assert summary["average_regret_vs_hindsight"] == pytest.approx(0.42)


def test_summarize_backtest_missing_regret_field_is_none():
    """Legacy result dicts without regret_vs_hindsight shouldn't crash the summary."""
    summary = summarize_backtest([{"position_delta": 0.0, "verdict": "MATCHED REAL STRATEGY"}])
    assert summary["average_regret_vs_hindsight"] is None


# ─── FR-8: multi-engine comparison ─────────────────────────────────────────────

def test_engines_tuple_has_all_three():
    assert set(ENGINES) == {"search", "gametheory", "rl"}


@pytest.mark.skipif(not table_exists("races", "2023_bahrain"), reason="2023_bahrain not ingested")
def test_backtest_decision_point_engine_dispatch():
    """Each engine name should route to its own strategy function and tag the result."""
    for engine in ENGINES:
        res = backtest_decision_point(
            race_id="2023_bahrain", driver="ALO", decision_lap=14, n_sims=20, engine=engine,
        )
        assert res["engine"] == engine
        assert "regret_vs_hindsight" in res
        assert res["regret_vs_hindsight"] >= -1e-6  # never negative (best candidate is a lower bound)


def test_summarize_multi_engine_backtest():
    mock_multi = {
        "search": [
            {"position_delta": 1.0, "verdict": "AI ADVANTAGE (+1.0 POS)", "regret_vs_hindsight": 0.0},
        ],
        "gametheory": [
            {"position_delta": -1.0, "verdict": "REAL STRATEGY BETTER (-1.0 POS)", "regret_vs_hindsight": 0.5},
        ],
        "rl": [
            {"verdict": "EVALUATION_FAILED", "error": "boom"},
        ],
    }
    result = summarize_multi_engine_backtest(mock_multi)
    assert set(result["engines"].keys()) == {"search", "gametheory", "rl"}
    # search has a 100% success rate (its only point was an AI advantage);
    # gametheory's only point lost to the real strategy; rl has no scored
    # points at all (its "success_rate_pct" defaults to 0.0, not None, so
    # it still participates in ranking but never wins over a real success).
    assert result["best_engine_by_success_rate"] == "search"
    assert result["ranking"][0] == "search"


# ─── FR-8: full-dataset backtesting (real pit stops, not curated benchmarks) ───

@pytest.mark.skipif(not table_exists("races", "2023_bahrain"), reason="2023_bahrain not ingested")
class TestFullDatasetBacktest:
    def test_actual_outcome_matches_real_pit_stop(self):
        """_actual_outcome should find ALO's real pit stop after lap 14 and
        their real classified finishing position, from ingested data."""
        con = connect()
        result = _actual_outcome("2023_bahrain", "ALO", 14, con)
        assert result["actual_finish"] is not None
        assert 1 <= result["actual_finish"] <= 20
        # ALO pitted for HARD on real lap 15 in this race (see HISTORICAL_BENCHMARKS)
        assert result["actual_action"].startswith("BOX LAP")
        assert result["actual_compound"] == "HARD"

    def test_actual_outcome_no_further_stop(self):
        """A decision lap after a driver's last real stop should report no further stop."""
        con = connect()
        # Lap 56 of a 57-lap race: essentially nothing pits again this late.
        result = _actual_outcome("2023_bahrain", "ALO", 56, con)
        assert result["actual_action"] == "STAY OUT (no further stop)"
        assert result["actual_compound"] is None

    def test_decision_points_for_race_returns_real_pit_laps(self):
        con = connect()
        points = decision_points_for_race("2023_bahrain", con=con, max_points=2)
        assert len(points) <= 2
        for driver, lap in points:
            assert isinstance(driver, str)
            assert lap >= 3  # decision_points_for_race requires lap > 3 stops

    def test_run_dataset_backtest_single_race(self):
        """End-to-end: derive decision points from real data, run the search
        engine, and get back real-outcome-scored results with no curated
        benchmark involved for most/all of them."""
        results = run_dataset_backtest(
            race_ids=["2023_bahrain"], n_sims=20, engine="search", decision_points_per_race=2,
        )
        assert len(results) >= 1
        for r in results:
            assert r["race_id"] == "2023_bahrain"
            assert r["engine"] == "search"
            assert "verdict" in r

    def test_run_dataset_backtest_calls_on_result_per_point(self):
        """on_result is the checkpointing hook the CLI script relies on."""
        seen = []
        results = run_dataset_backtest(
            race_ids=["2023_bahrain"], n_sims=20, engine="search",
            decision_points_per_race=1, on_result=seen.append,
        )
        assert len(seen) == len(results)


def test_run_dataset_backtest_skips_unignested_race():
    """A race with no data (or no pit stops) should be skipped, not crash the run."""
    results = run_dataset_backtest(
        race_ids=["not_a_real_race_id"], n_sims=20, engine="search",
    )
    assert results == []
