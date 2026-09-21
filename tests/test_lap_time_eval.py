"""Unit tests for the lap-time evaluation harness (src/evaluation/lap_time_eval.py)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.lap_time_eval import breakdown, load_split, split_laps, summarise


def _synthetic_test_predictions(seed=42, n_races=4, laps_per_race=30):
    rng = np.random.default_rng(seed)
    rows = []
    circuits = ["CircA", "CircB", "Monaco", "Monza"]

    for r in range(n_races):
        race_id = f"2023_race_{r}"
        circ = circuits[r % len(circuits)]
        for lap in range(1, laps_per_race + 1):
            for d in range(4):
                driver = f"D{d}"
                true_time = 90.0 + rng.normal(0, 0.5)
                ref_p = 90.0
                gap_ahead = float(d * 0.8)  # D0: lead, D1: 0.8s (DRS), D2: 1.6s (wake), D3: 2.4s, D4: 3.5s
                if d == 3:
                    gap_ahead = 3.5  # clean air
                pred = true_time + rng.normal(0, 0.2)

                rows.append({
                    "race_id": race_id,
                    "circuit": circ,
                    "driver": driver,
                    "lap_number": lap,
                    "lap_time": true_time,
                    "ref_pace": ref_p,
                    "pred": pred,
                    "horizon": 1,
                    "prev_gap_ahead": gap_ahead,
                })
    return pd.DataFrame(rows)


def test_lap_time_split_laps():
    split = load_split()
    df = pd.DataFrame({
        "race_id": ["2022_austria", "2023_bahrain", "2024_monaco"],
        "circuit": ["Spielberg", "Sakhir", "Monaco"],
        "lap_time": [70.0, 92.0, 75.0],
    })
    train, test = split_laps(df, split)
    # 2022_austria is in test_races, 2024_monaco is in held_out_circuits and test_races
    assert "2022_austria" in test["race_id"].values
    assert "2023_bahrain" in train["race_id"].values
    assert "2024_monaco" not in train["race_id"].values


def test_summarise_metrics_and_ci():
    preds = _synthetic_test_predictions()
    res = summarise(preds, n_boot=100)

    assert "lap_mae" in res
    assert "lap_rmse" in res
    assert "lap_mae_ci" in res
    assert res["lap_mae"] > 0
    assert res["lap_rmse"] >= res["lap_mae"]

    lo, hi = res["lap_mae_ci"]
    assert lo <= res["lap_mae"] <= hi


def test_breakdown_traffic_slices():
    split = load_split()
    preds = _synthetic_test_predictions()
    report = breakdown(preds, split=split, n_boot=50)

    assert "all" in report
    assert "clean_air" in report
    assert "in_traffic_wake" in report
    assert "drs_range" in report

    assert report["all"]["n_laps"] > 0
    assert report["clean_air"]["n_laps"] > 0
    assert report["drs_range"]["n_laps"] > 0
