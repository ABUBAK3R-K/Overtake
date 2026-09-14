"""Tests for the lap-time model on synthetic races (tests/test_tyre_model.py).

Tiny models — these check leakage, plumbing and the handoff contract, not
accuracy; accuracy is reported by `python -m src.models.lap_time`.
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.models import lap_time
from src.models.lap_time import (
    LapTimeModel, LapTimePredictor, _gap_ahead, build_training_rows, history_features,
)
from tests.test_tyre_model import synthetic_race

SMALL_PARAMS = {**lap_time.XGB_PARAMS, "n_estimators": 30, "learning_rate": 0.3}
HISTORY_COLUMNS = ["ref_pace", "ref_tyre_age", "ref_compound", "driver_vs_field",
                   "field_pace_vs_ref", "ref_is_field", "laps_since_ref"]


@dataclass
class State:
    """Stand-in for the Decision-Making lane's RaceState (Design.md Section 5)."""
    lap: int
    positions: dict
    gaps: dict
    tyres: dict
    safety_car: bool = False
    race_id: str | None = None


def with_race_state(race: pd.DataFrame) -> pd.DataFrame:
    """Add position / gap_to_leader, derived from cumulative race time."""
    race = race.sort_values(["driver", "lap_number"]).copy()
    race["race_time"] = race.groupby("driver")["lap_time"].cumsum()
    race["position"] = race.groupby("lap_number")["race_time"].rank(method="first").astype(int)
    race["gap_to_leader"] = race["race_time"] - race.groupby("lap_number")["race_time"].transform("min")
    return race.assign(total_laps=int(race["lap_number"].max())).drop(columns="race_time")


@pytest.fixture(scope="module")
def laps() -> pd.DataFrame:
    return pd.concat([
        with_race_state(synthetic_race("r1", "One", wear_scale=1.0, seed=3)),
        with_race_state(synthetic_race("r2", "Two", wear_scale=1.5, seed=4)),
    ], ignore_index=True)


@pytest.fixture(scope="module")
def rows(laps) -> pd.DataFrame:
    return build_training_rows(laps, horizons=(1, 3, 10))


@pytest.fixture(scope="module")
def model(rows) -> LapTimeModel:
    return LapTimeModel(params=SMALL_PARAMS, sc_ratio=1.5).fit(rows)


def state_at(race: pd.DataFrame, lap: int, **kwargs) -> State:
    cur = race[race["lap_number"] == lap]
    return State(
        lap=lap,
        positions=dict(zip(cur["driver"], cur["position"])),
        gaps=dict(zip(cur["driver"], cur["gap_to_leader"])),
        tyres={d: (c, int(a)) for d, c, a in zip(cur["driver"], cur["compound"], cur["tyre_age_at_lap"])},
        **kwargs,
    )


# --- Leakage ---------------------------------------------------------------

def test_history_features_ignore_laps_after_anchor(laps):
    race = laps[laps["race_id"] == "r1"]
    corrupted = race.copy()
    corrupted.loc[corrupted["lap_number"] > 20, "lap_time"] = 999.0

    clean_hist, corrupt_hist = history_features(race, 20), history_features(corrupted, 20)
    pd.testing.assert_frame_equal(clean_hist["drivers"], corrupt_hist["drivers"])
    assert clean_hist["field_pace"] == corrupt_hist["field_pace"]
    assert clean_hist["field_lap"] <= 20


def test_training_rows_use_only_history_up_to_anchor(laps):
    corrupted = laps.copy()
    corrupted.loc[corrupted["lap_number"] > 25, "lap_time"] += 50.0
    a = build_training_rows(laps, horizons=(1, 5))
    b = build_training_rows(corrupted, horizons=(1, 5))
    key = ["race_id", "anchor_lap", "driver", "lap_number"]
    a, b = a[a["anchor_lap"] <= 25].set_index(key), b[b["anchor_lap"] <= 25].set_index(key)
    shared = a.index.intersection(b.index)
    assert len(shared) > 100
    pd.testing.assert_frame_equal(a.loc[shared, HISTORY_COLUMNS], b.loc[shared, HISTORY_COLUMNS])


def test_training_targets_are_after_anchor(rows):
    assert (rows["lap_number"] > rows["anchor_lap"]).all()
    assert set(rows["horizon"]) <= {1, 3, 10}


# --- Features --------------------------------------------------------------

def test_gap_ahead():
    positions = pd.Series({"A": 2, "B": 1, "C": 3})
    gaps = pd.Series({"A": 1.5, "B": 0.0, "C": 4.0})
    out = _gap_ahead(positions, gaps)
    assert np.isnan(out["B"]) and out["A"] == 1.5 and out["C"] == 2.5


def test_reference_pace_is_median_of_last_clean_laps(laps):
    race = laps[laps["race_id"] == "r1"]
    driver = "D00"
    hist = history_features(race, 10)
    expected = race[(race["driver"] == driver) & race["lap_number"].between(6, 10)]["lap_time"].median()
    assert hist["drivers"].loc[driver, "ref_pace"] == pytest.approx(expected)


# --- Predictor / handoff ---------------------------------------------------

def test_prediction_is_close_to_real_next_lap(laps, model):
    race = laps[laps["race_id"] == "r1"]
    predictor = LapTimePredictor("r1", 20, model=model, laps=laps)
    preds = predictor.predict_field(state_at(race, 20))
    actual = race[race["lap_number"] == 21].set_index("driver")["lap_time"]
    clean = actual.index[~race[race["lap_number"] == 21].set_index("driver")["pit_flag"]]
    errors = [abs(preds[d] - actual[d]) for d in clean]
    assert np.median(errors) < 1.0


def test_single_field_and_batch_predictions_agree(laps, model):
    race = laps[laps["race_id"] == "r1"]
    predictor = LapTimePredictor("r1", 15, model=model, laps=laps)
    state = state_at(race, 15)
    field = predictor.predict_field(state)
    assert predictor(state, "D03") == pytest.approx(field["D03"])

    later = state_at(race, 22)
    batch = predictor.predict_many([state, later, copy.deepcopy(state)])
    assert batch[0] == pytest.approx(field) and batch[2] == pytest.approx(field)
    assert set(batch[1]) == set(later.positions)


def test_safety_car_scales_prediction(laps, model):
    race = laps[laps["race_id"] == "r1"]
    predictor = LapTimePredictor("r1", 15, model=model, laps=laps)
    green = predictor(state_at(race, 15), "D01")
    sc = predictor(state_at(race, 15, safety_car=True), "D01")
    assert sc == pytest.approx(green * 1.5)


def test_fresh_tyres_after_simulated_stop_are_faster_than_old(laps, model):
    race = laps[laps["race_id"] == "r2"]
    predictor = LapTimePredictor("r2", 25, model=model, laps=laps)
    state = state_at(race, 25)
    old = copy.deepcopy(state); old.tyres["D02"] = ("MEDIUM", 30)
    new = copy.deepcopy(state); new.tyres["D02"] = ("MEDIUM", 0)
    assert predictor(new, "D02") < predictor(old, "D02")


def test_state_before_anchor_is_rejected(laps, model):
    race = laps[laps["race_id"] == "r1"]
    predictor = LapTimePredictor("r1", 20, model=model, laps=laps)
    with pytest.raises(ValueError, match="before this predictor's history"):
        predictor(state_at(race, 19), "D00")


def test_module_function_requires_race_id(laps):
    race = laps[laps["race_id"] == "r1"]
    with pytest.raises(AttributeError, match="make_lap_time_predictor"):
        lap_time.predict_lap_time(state_at(race, 10), "D00")


def test_module_function_uses_history_up_to_state_lap(laps, model, monkeypatch):
    race = laps[laps["race_id"] == "r1"]
    monkeypatch.setattr(lap_time, "_default_model", lambda: model)
    monkeypatch.setattr(lap_time, "_all_laps", lambda: laps)
    lap_time._predictor_at.cache_clear()
    try:
        state = state_at(race, 18, race_id="r1")
        expected = LapTimePredictor("r1", 18, model=model, laps=laps)(state, "D05")
        assert lap_time.predict_lap_time(state, "D05") == pytest.approx(expected)
    finally:
        lap_time._predictor_at.cache_clear()


def test_save_load_round_trip(model, rows, tmp_path):
    model.save(tmp_path)
    loaded = LapTimeModel.load(tmp_path)
    sample = rows.head(50)
    np.testing.assert_allclose(loaded.predict_rows(sample), model.predict_rows(sample), rtol=1e-6)
    assert loaded.sc_ratio == 1.5


def test_missing_model_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.models.lap_time"):
        LapTimeModel.load(tmp_path)
