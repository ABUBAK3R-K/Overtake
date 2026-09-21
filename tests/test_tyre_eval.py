"""Tyre evaluation harness, Bayesian model and SHAP (Phase 2)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.tyre_eval import load_split, split_frame, summarise
from src.models.tyre import TyreModel
from src.models.tyre_baselines import MeanCurveModel
from src.models.tyre_bayes import BayesTyreModel


def _frame(circuits=("A", "B", "C"), races_per=2, slope=0.1, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ci, circ in enumerate(circuits):
        for r in range(races_per):
            for comp in ("SOFT", "MEDIUM", "HARD"):
                age = np.tile(np.arange(1, 31), 6)
                s = slope * (1 + 0.1 * ci)
                rows.append(pd.DataFrame({
                    "race_id": f"{2022 + r}_{circ}", "circuit": circ, "compound": comp,
                    "tyre_age_at_lap": age,
                    "pace_loss": s * np.clip(age - 3, 0, None) + rng.normal(0, noise, len(age)),
                    "train_weight": 1.0}))
    return pd.concat(rows, ignore_index=True)


def test_frozen_split_shape():
    split = load_split()
    assert len(split["test_races"]) == 14
    assert set(split["wet_test_races"]) <= set(split["test_races"])
    assert len(split["wet_test_races"]) >= 2
    assert set(split["held_out_circuits"]) == {"Monaco", "Monza", "Baku"}


def test_split_drops_held_out_circuits_from_train_in_every_era():
    split = {"test_races": ["2023_Monza"], "held_out_circuits": ["Monza"], "wet_test_races": []}
    frame = pd.DataFrame({"race_id": ["2023_Monza", "2019_Monza", "2023_Spa"],
                          "circuit": ["Monza", "Monza", "Spa"]})
    train, test = split_frame(frame, split)
    assert list(train["race_id"]) == ["2023_Spa"] and list(test["race_id"]) == ["2023_Monza"]


def test_summarise_perfect_and_offset():
    f = _frame()
    perfect = summarise(f.assign(pred=f["pace_loss"]), n_boot=50)
    assert perfect["curve_mae"] == pytest.approx(0) and perfect["lap_rmse"] == pytest.approx(0)
    off = summarise(f.assign(pred=f["pace_loss"] + 0.5), n_boot=200)
    assert off["curve_mae"] == pytest.approx(0.5) and off["lap_rmse"] == pytest.approx(0.5)
    lo, hi = off["curve_mae_ci"]
    assert lo <= 0.5 <= hi


def test_bayes_recovers_slope_and_widens_for_unseen_circuit():
    f = _frame(slope=0.1, noise=0.05)
    m = BayesTyreModel().fit(f)
    probe = pd.DataFrame({"compound": ["MEDIUM"] * 2, "tyre_age_at_lap": [23, 23],
                          "circuit": ["A", "Z"]})            # seen vs unseen
    pred = m.predict_frame(probe)
    assert pred[0] == pytest.approx(0.1 * 20, rel=0.1)       # 20 laps past reference age
    assert pred[1] == pytest.approx(pred[0], rel=0.2)         # falls back to compound slope
    lo, hi = m.predict_interval(probe)
    assert (hi[1] - lo[1]) > (hi[0] - lo[0])                  # unseen circuit: more uncertainty


def test_mean_curve_is_monotone():
    f = _frame()
    m = MeanCurveModel().fit(f)
    probe = pd.DataFrame({"compound": "SOFT", "tyre_age_at_lap": np.arange(1, 31)})
    assert np.all(np.diff(m.predict_frame(probe)) >= -1e-9)


def test_shap_contributions_sum_to_prediction():
    f = _frame().assign(track_temp=30.0)
    m = TyreModel("pooled", {"n_estimators": 30, "max_depth": 3, "random_state": 0}).fit(f)
    sv = m.shap_values(f.head(50))
    raw = m.boosters["all"].predict(m._features(f.head(50).reset_index(drop=True)))
    assert sv.sum(axis=1).to_numpy() == pytest.approx(raw, abs=1e-3)
    assert "bias" in sv.columns and "circuit" in sv.columns


def test_xgb_without_circuit_ignores_it():
    f = _frame().assign(track_temp=30.0)
    m = TyreModel("pooled", {"n_estimators": 30, "max_depth": 3, "random_state": 0},
                  use_circuit=False, use_temp=False).fit(f)
    a = f.head(20).assign(circuit="A")
    b = f.head(20).assign(circuit="never-seen")
    assert m.predict_frame(a) == pytest.approx(m.predict_frame(b))


def test_routed_model_routes_by_circuit_and_ignores_temp():
    from src.models.tyre import RoutedTyreModel
    f = _frame().assign(track_temp=30.0)
    params = {"n_estimators": 30, "max_depth": 3, "random_state": 0}
    m = RoutedTyreModel.new(params, 40).fit(f)
    probe = f.head(30)
    seen = m.predict_frame(probe.assign(circuit="A"))
    assert seen == pytest.approx(m.known.predict_frame(probe.assign(circuit="A")))
    unseen = m.predict_frame(probe.assign(circuit="never-seen"))
    assert unseen == pytest.approx(m.agnostic.predict_frame(probe))
    # track temperature is not a model input
    assert m.predict_frame(probe.assign(track_temp=5.0)) == pytest.approx(m.predict_frame(probe))


def test_routed_model_save_load_roundtrip(tmp_path):
    from src.models.tyre import RoutedTyreModel
    f = _frame().assign(track_temp=30.0)
    m = RoutedTyreModel.new({"n_estimators": 20, "max_depth": 3, "random_state": 0}, 40).fit(f)
    m.save(tmp_path)
    loaded = RoutedTyreModel.load(tmp_path)
    probe = f.head(25).assign(circuit=["A"] * 10 + ["zzz"] * 15)
    assert loaded.predict_frame(probe) == pytest.approx(m.predict_frame(probe))
