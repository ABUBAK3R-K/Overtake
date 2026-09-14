"""Tests for the tyre degradation model on synthetic races with known wear.

Small, fast models (few trees) — these check the plumbing and the target
construction, not model quality; quality is reported by `python -m src.models.tyre`.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import tyre
from src.models.tyre import TyreModel, build_training_frame

WEAR_PER_LAP = {"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.03}
CONFIG = {
    "reference_max_age": 3, "max_age": 40, "min_laps": 30, "min_reference_laps": 8,
    "xgboost": {"n_estimators": 40, "learning_rate": 0.3, "max_depth": 3,
                "min_child_weight": 5, "random_state": 0},
}


def synthetic_race(race_id: str, circuit: str, wear_scale: float, seed: int,
                   n_drivers: int = 12, n_laps: int = 45) -> pd.DataFrame:
    """Lap frame shaped like tyre.load_lap_frame(), with a fuel trend,
    driver pace offsets, a compound offset and linear wear past age 3."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        pit_lap = int(rng.integers(12, 30))
        first, second = [("SOFT", "HARD"), ("MEDIUM", "HARD"), ("HARD", "MEDIUM")][d % 3]
        driver_offset = 0.1 * d
        for lap in range(1, n_laps + 1):
            stint = 1 if lap <= pit_lap else 2
            compound = first if stint == 1 else second
            age = lap if stint == 1 else lap - pit_lap
            wear = wear_scale * WEAR_PER_LAP[compound] * max(age - 3, 0)
            compound_offset = {"SOFT": -0.6, "MEDIUM": -0.3, "HARD": 0.0}[compound]
            lap_time = (95 - 0.06 * lap + driver_offset + compound_offset + wear
                        + rng.normal(0, 0.05))
            rows.append({
                "race_id": race_id, "driver": f"D{d:02d}", "lap_number": lap,
                "lap_time": lap_time, "pit_flag": lap == pit_lap,
                "pit_out_flag": lap == pit_lap + 1, "track_status": "1",
                "is_accurate": True, "stint": stint, "compound": compound,
                "tyre_age_at_lap": age, "circuit": circuit, "track_temp": np.nan,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def laps() -> pd.DataFrame:
    return pd.concat([
        synthetic_race("r_low", "LowWear", wear_scale=0.5, seed=1),
        synthetic_race("r_high", "HighWear", wear_scale=2.0, seed=2),
    ], ignore_index=True)


@pytest.fixture(scope="module")
def train(laps) -> pd.DataFrame:
    return build_training_frame(laps, CONFIG)


@pytest.fixture(scope="module")
def model(train) -> TyreModel:
    return TyreModel(variant="pooled", params=CONFIG["xgboost"]).fit(train)


def _state(compound, age, circuit):
    return pd.DataFrame({"compound": [compound], "tyre_age_at_lap": [age],
                         "circuit": [circuit], "track_temp": [np.nan]})


def test_target_recovers_true_wear_despite_fuel_and_driver_effects(train):
    scale = train["race_id"].map({"r_low": 0.5, "r_high": 2.0})
    true_wear = scale * train["compound"].map(WEAR_PER_LAP) * (train["tyre_age_at_lap"] - 3).clip(lower=0)
    assert (train["pace_loss"] - true_wear).abs().mean() < 0.1


def test_training_frame_keeps_only_clean_laps(train):
    assert (train["lap_number"] > 1).all()
    assert not train["pit_flag"].any() and not train["pit_out_flag"].any()


def test_prediction_rises_with_age_and_is_never_negative(model):
    ages = [1, 5, 10, 20, 30]
    frame = pd.concat([_state("SOFT", a, "HighWear") for a in ages], ignore_index=True)
    preds = model.predict_frame(frame)
    assert (np.diff(preds) >= 0).all()
    assert (preds >= 0).all()
    assert preds[-1] > 2.0          # 2.0 * 0.12 * 27 ~= 6.5 s of true wear


def test_circuit_matters(model):
    high = model.predict_frame(_state("MEDIUM", 25, "HighWear"))[0]
    low = model.predict_frame(_state("MEDIUM", 25, "LowWear"))[0]
    assert high > low + 0.5


def test_unseen_circuit_gets_average_of_known_circuits(model):
    known = [model.predict_frame(_state("HARD", 20, c))[0] for c in model.circuits]
    unseen = model.predict_frame(_state("HARD", 20, "Nowhere"))[0]
    assert unseen == pytest.approx(np.mean(known), abs=1e-5)


def test_wet_compound_raises(model):
    with pytest.raises(ValueError, match="dry compounds"):
        model.predict_frame(_state("INTERMEDIATE", 5, "HighWear"))


@pytest.mark.parametrize("variant", ["pooled", "per_compound"])
def test_save_load_round_trip(train, tmp_path, variant):
    fitted = TyreModel(variant=variant, params=CONFIG["xgboost"]).fit(train)
    fitted.save(tmp_path)
    loaded = TyreModel.load(tmp_path)
    frame = pd.concat([_state(c, 15, "HighWear") for c in ("SOFT", "MEDIUM", "HARD")],
                      ignore_index=True)
    np.testing.assert_allclose(loaded.predict_frame(frame), fitted.predict_frame(frame), rtol=1e-6)


def test_predict_tyre_degradation_handoff(model, tmp_path, monkeypatch):
    model.save(tmp_path)
    monkeypatch.setattr(tyre, "MODEL_DIR", tmp_path)
    tyre._default_model.cache_clear()
    tyre._predict_cached.cache_clear()
    try:
        value = tyre.predict_tyre_degradation("soft", 20, "HighWear", track_temp=None)
        assert isinstance(value, float)
        assert value == pytest.approx(model.predict_frame(_state("SOFT", 20, "HighWear"))[0], abs=1e-5)
    finally:
        tyre._default_model.cache_clear()
        tyre._predict_cached.cache_clear()


def test_missing_model_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.models.tyre"):
        TyreModel.load(tmp_path)
