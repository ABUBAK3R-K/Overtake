"""Tests for the common-yardstick three-engine backtest (FR-8):
summarize_fair_backtest, real_strategy, decision-point selection."""

from __future__ import annotations

import pytest

from src.evaluation.backtest import (
    _strategy_key,
    decision_points_for_race,
    real_strategy,
    summarize_fair_backtest,
)
from src.ingestion.storage import connect, table_exists


def _row(race_id: str, search: float, rl: float, real: float | None, best: float, actual: int = 5) -> dict:
    def eng(pos):
        return {"strategy": {"name": "X", "pit_laps": [], "compounds": []},
                "self_expected_position": pos - 0.5, "expected_position": pos,
                "regret": round(pos - best, 3), "picked_best": pos == best, "seconds": 1.0}
    real_entry = {"strategy": {"name": "REAL", "pit_laps": [9], "compounds": ["HARD"]}}
    if real is not None:
        real_entry.update(expected_position=real, regret=round(real - best, 3), picked_best=real == best)
    return {"race_id": race_id, "driver": "VER", "decision_lap": 8,
            "engines": {"search": eng(search), "rl": eng(rl)}, "real": real_entry, "actual_finish": actual}


def test_summary_regret_ranking_and_vs_real():
    rows = [_row("r1", 3.0, 4.0, 5.0, 3.0), _row("r2", 6.0, 6.0, 6.1, 6.0),
            {"race_id": "r3", "skipped": "driver not running"}]
    s = summarize_fair_backtest(rows)
    assert s["n_skipped"] == 1 and s["all"]["n_points"] == 2
    assert s["all"]["search"]["mean_regret"] == 0.0
    assert s["all"]["rl"]["mean_regret"] == 0.5
    assert s["all"]["real"]["mean_regret"] == pytest.approx(1.05)
    assert s["ranking_by_mean_regret"] == ["search", "rl", "real"]
    vr = s["all"]["search"]["vs_real"]
    assert vr["mean_diff"] == pytest.approx(-1.05)
    assert vr["better_pct"] == 50.0 and vr["tie_pct"] == 50.0 and vr["worse_pct"] == 0.0
    assert s["all"]["search"]["self_report_optimism"] == -0.5


def test_summary_excludes_unscorable_real_from_pairing_and_calibration():
    rows = [_row("r1", 3.0, 4.0, 5.0, 3.0, actual=4), _row("r2", 6.0, 6.0, None, 6.0)]
    s = summarize_fair_backtest(rows)
    assert s["all"]["real"]["n"] == 1 and s["all"]["real"]["failed"] == 1
    assert s["all"]["search"]["vs_real"]["n"] == 1
    assert s["simulator_calibration"]["n"] == 1
    assert s["simulator_calibration"]["mean_bias"] == 1.0


def test_summary_held_out_and_tag_blocks():
    rows = [_row("held", 3.0, 3.0, 4.0, 3.0), _row("seen", 3.0, 5.0, 4.0, 3.0)]
    s = summarize_fair_backtest(rows, held_out={"held"}, race_tags={"seen": {"wet": True}})
    assert s["held_out_races"]["rl"]["mean_regret"] == 0.0
    assert s["training_races"]["rl"]["mean_regret"] == 2.0
    assert s["tag_wet"]["n_points"] == 1 and s["tag_sc_heavy"]["n_points"] == 0


def test_strategy_key_normalises():
    assert _strategy_key({"pit_laps": [27.0], "compounds": ["hard"]}) == _strategy_key(
        {"pit_laps": [27], "compounds": ["HARD"], "name": "other"})
    assert _strategy_key({"pit_laps": [], "compounds": []}) == ((), ())


needs_data = pytest.mark.skipif(not table_exists("pit_stops", "2018_australia"), reason="pit_stops not ingested")


@needs_data
def test_decision_points_skip_retirements():
    con = connect()
    for driver, lap in decision_points_for_race("2018_australia", con=con, max_points=5):
        later = con.sql("SELECT MAX(lap_number) FROM laps WHERE race_id = '2018_australia' AND driver = ?",
                        params=[driver]).fetchone()[0]
        assert later > lap + 3


@needs_data
def test_real_strategy_flags_retirement_as_unscorable():
    con = connect()
    rs = real_strategy("2018_australia", "GAS", 13, con)
    assert "unscorable" in rs
    ok = real_strategy("2023_spain", "VER", 26, con)
    assert "unscorable" not in ok and ok["compounds"] and ok["pit_laps"][0] > 26
