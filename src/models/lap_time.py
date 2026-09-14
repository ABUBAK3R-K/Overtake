"""Lap-time model (Design.md Section 6.4).

Hands off predict_lap_time(state, driver) to the Decision-Making lane.

What it predicts
----------------
The driver's next lap time (lap state.lap + 1) in seconds, as a *racing* lap:
no pit-lane time. Pit loss belongs to the strategy side. If state.safety_car
is set, the racing prediction is scaled by the median safety-car lap ratio
measured from the data (~1.5x).

Two ways to call it
-------------------
1. Replay / API (real history up to the current lap is known):
       predict_lap_time(state, driver)
   `state` must carry `race_id`. History is read through as_of_lap(state.lap).

2. Monte Carlo (simulated laps after a real decision lap):
       predictor = make_lap_time_predictor(race_id, decision_lap)
       predictor(sim_state, driver)       # same (state, driver) signature
   Real history stops at decision_lap; later laps only come from sim_state.
   Build the predictor once per decision lap, not per call.

How the model works
-------------------
For an anchor lap A (the last lap of real history) and a target lap T > A,
features come from two places:
  - history as of A, strictly via as_of_lap(): the driver's median of their
    last 5 clean laps ("reference pace"), their pace relative to the field,
    the field's current pace;
  - race state at T-1, which the simulator supplies: compound and tyre age,
    position, gap to the car ahead and to the leader.
XGBoost predicts lap_time(T) - reference pace. Training uses several
horizons T - A (1 to 30 laps), so the same model serves one-step replay and
multi-lap simulation.

Usage:
    python -m src.models.lap_time      # evaluate, train on all races, save
"""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.tyre import REPO_ROOT, load_lap_frame
from src.preprocessing.cleaning import clean_gap_to_leader, is_clean_lap
from src.preprocessing.leakage import as_of_lap

MODEL_DIR = REPO_ROOT / "data" / "models" / "lap_time"

# Laps of real history used for a driver's reference pace.
REFERENCE_LAPS = 5
# A lap needs this many clean laps across the field to set the field pace.
MIN_FIELD_LAPS = 5
# Gaps between the anchor lap and the predicted lap used in training.
HORIZONS = (1, 2, 3, 5, 8, 12, 20, 30)

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
# No absolute lap times or lap numbers: with them the model memorises circuits
# (by lap length / race length) and did worse on held-out races than simply
# repeating the reference pace. The target is already relative to ref_pace.
NUMERIC_FEATURES = [
    "horizon", "lap_fraction", "tyre_age", "same_compound_as_ref",
    "tyre_age_vs_ref", "laps_since_ref", "ref_is_field", "driver_vs_field",
    "field_pace_vs_ref", "prev_position", "prev_gap_ahead", "prev_gap_to_leader",
]
CATEGORICAL_FEATURES = ["compound", "ref_compound"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

XGB_PARAMS = {
    # Absolute error -> predicts the median lap. Rain transitions put +10-20 s
    # outliers in the targets; squared error chased them and hurt dry laps.
    "objective": "reg:absoluteerror",
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
    "min_child_weight": 20, "subsample": 0.8, "random_state": 42,
}

log = logging.getLogger("overtake.models.lap_time")


# --- Features from history (as of the anchor lap) --------------------------

def history_features(race_laps: pd.DataFrame, anchor_lap: int) -> dict:
    """Everything the model knows from real laps 1..anchor_lap of one race.

    Returns {"drivers": per-driver frame, "field_pace": float, "field_lap": int}.
    """
    visible = as_of_lap(race_laps, "lap_number", anchor_lap)
    clean = visible[is_clean_lap(visible)].sort_values("lap_number")
    if clean.empty:
        return {"drivers": pd.DataFrame(), "field_pace": np.nan, "field_lap": np.nan}

    lap_median = clean.groupby("lap_number")["lap_time"].median()
    clean = clean.assign(vs_field=clean["lap_time"] - clean["lap_number"].map(lap_median))

    lap_counts = clean.groupby("lap_number").size()
    field_laps = lap_counts[lap_counts >= MIN_FIELD_LAPS].index
    field_lap = field_laps.max() if len(field_laps) else lap_counts.index.max()

    recent = clean.groupby("driver").tail(REFERENCE_LAPS).groupby("driver")
    drivers = pd.DataFrame({
        "ref_pace": recent["lap_time"].median(),
        "ref_lap": recent["lap_number"].mean(),
        "ref_tyre_age": recent["tyre_age_at_lap"].mean(),
        "ref_compound": recent["compound"].last(),
        "driver_vs_field": recent["vs_field"].median(),
    })
    return {"drivers": drivers, "field_pace": float(lap_median[field_lap]),
            "field_lap": int(field_lap)}


def assemble_features(history: dict, target: pd.DataFrame, anchor_lap: int) -> pd.DataFrame:
    """Model features for target rows given history as of anchor_lap.

    `target` has one row per prediction with columns: driver, lap_number
    (the lap being predicted), total_laps, compound, tyre_age_at_lap (on the
    predicted lap), prev_position, prev_gap_ahead, prev_gap_to_leader (end
    of the lap before).
    """
    X = target.reset_index(drop=True).copy()
    drivers = history["drivers"]
    X = X.join(drivers, on="driver") if not drivers.empty else X.assign(
        ref_pace=np.nan, ref_lap=np.nan, ref_tyre_age=np.nan,
        ref_compound=None, driver_vs_field=np.nan)

    # Drivers with no clean lap yet (early laps, after a long SC) fall back to
    # the field's pace.
    X["ref_is_field"] = X["ref_pace"].isna().astype(float)
    X["ref_pace"] = X["ref_pace"].fillna(history["field_pace"])
    X["ref_lap"] = X["ref_lap"].fillna(history["field_lap"])
    X["driver_vs_field"] = X["driver_vs_field"].fillna(0.0)

    X["horizon"] = X["lap_number"] - anchor_lap
    X["lap_fraction"] = X["lap_number"] / X["total_laps"]
    X["tyre_age"] = X["tyre_age_at_lap"].astype(float)
    X["same_compound_as_ref"] = (X["compound"] == X["ref_compound"]).astype(float)
    X["tyre_age_vs_ref"] = X["tyre_age"] - X["ref_tyre_age"].astype(float)
    X["laps_since_ref"] = X["lap_number"] - X["ref_lap"]
    X["field_pace_vs_ref"] = history["field_pace"] - X["ref_pace"]
    for col in ("prev_position", "prev_gap_ahead", "prev_gap_to_leader"):
        X[col] = X[col].astype(float)
    X["compound"] = pd.Categorical(X["compound"], categories=COMPOUNDS)
    X["ref_compound"] = pd.Categorical(X["ref_compound"], categories=COMPOUNDS)
    return X


def _gap_ahead(positions: pd.Series, gaps: pd.Series) -> pd.Series:
    """Gap (s) to the car one position ahead; NaN for the leader."""
    order = positions.sort_values()
    gap_sorted = gaps.reindex(order.index)
    ahead = gap_sorted - gap_sorted.shift(1)
    return ahead.reindex(positions.index)


# --- Training rows ---------------------------------------------------------

def build_training_rows(laps: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """(anchor, target lap) pairs for every race, driver and horizon.

    Targets are clean laps only. Race state at T-1 is taken from the real
    data, standing in for what the simulator would supply.
    """
    laps = laps.copy()
    laps["gap_clean"] = clean_gap_to_leader(laps)
    laps["clean"] = is_clean_lap(laps)

    parts = []
    for race_id, race in laps.groupby("race_id"):
        prev = race[["driver", "lap_number", "position", "gap_clean"]].copy()
        prev["prev_gap_ahead"] = np.nan
        for lap, idx in prev.groupby("lap_number").groups.items():
            prev.loc[idx, "prev_gap_ahead"] = _gap_ahead(
                prev.loc[idx, "position"].astype(float), prev.loc[idx, "gap_clean"])
        prev = prev.assign(lap_number=prev["lap_number"] + 1).rename(columns={
            "position": "prev_position", "gap_clean": "prev_gap_to_leader"})
        targets = race[race["clean"]].merge(prev, on=["driver", "lap_number"], how="left")

        for anchor in range(2, int(race["lap_number"].max())):
            wanted = targets[targets["lap_number"].isin([anchor + h for h in horizons])]
            if wanted.empty:
                continue
            history = history_features(race, anchor)
            if np.isnan(history["field_pace"]):
                continue
            X = assemble_features(history, wanted, anchor)
            X["race_id"], X["anchor_lap"] = race_id, anchor
            X["target"] = X["lap_time"] - X["ref_pace"]
            parts.append(X)
    return pd.concat(parts, ignore_index=True)


def safety_car_ratio(laps: pd.DataFrame) -> float:
    """Median (fully neutralised lap time / race median racing lap time)."""
    status = laps["track_status"].astype("string").fillna("")
    fully_neutral = status.str.fullmatch("[4567]+") & laps["lap_time"].notna() \
        & ~laps["pit_flag"].astype(bool) & ~laps["pit_out_flag"].astype(bool) \
        & (laps["lap_number"] > 1)
    race_median = laps[is_clean_lap(laps)].groupby("race_id")["lap_time"].median()
    ratio = laps.loc[fully_neutral, "lap_time"] / laps.loc[fully_neutral, "race_id"].map(race_median)
    return float(ratio.median())


# --- Model -----------------------------------------------------------------

class LapTimeModel:
    def __init__(self, params: dict | None = None, sc_ratio: float = 1.5):
        self.params = params or XGB_PARAMS
        self.sc_ratio = sc_ratio
        self.booster: xgb.XGBRegressor | None = None

    def fit(self, rows: pd.DataFrame) -> "LapTimeModel":
        self.booster = xgb.XGBRegressor(**self.params, tree_method="hist",
                                        enable_categorical=True)
        self.booster.fit(rows[FEATURES], rows["target"])
        return self

    def predict_rows(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted racing lap times for assembled feature rows."""
        return X["ref_pace"].to_numpy() + self.booster.predict(X[FEATURES])

    def feature_importance(self) -> dict[str, float]:
        gain = self.booster.get_booster().get_score(importance_type="total_gain")
        total = sum(gain.values()) or 1.0
        return dict(sorted(((k, v / total) for k, v in gain.items()), key=lambda kv: -kv[1]))

    def save(self, directory: Path = MODEL_DIR, metrics: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "booster.json")
        meta = {"params": self.params, "sc_ratio": self.sc_ratio, "features": FEATURES,
                "feature_importance": self.feature_importance(), "metrics": metrics or {}}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> "LapTimeModel":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no lap-time model at {directory}; train one with `python -m src.models.lap_time`"
            )
        meta = json.loads(meta_path.read_text())
        model = cls(params=meta["params"], sc_ratio=meta["sc_ratio"])
        model.booster = xgb.XGBRegressor(enable_categorical=True)
        model.booster.load_model(directory / "booster.json")
        return model


# --- Handoff (Design.md Section 5) -----------------------------------------

def _state_rows(states: list, total_laps: int) -> pd.DataFrame:
    """Target rows for lap state.lap + 1, every driver of every state.

    state.tyres[driver] = (compound, age) at the end of state.lap; the next
    lap is driven on that set at age + 1. After a simulated stop, set the new
    compound with age 0. Built with plain Python per state, since Monte Carlo
    passes hundreds of states at once and per-state pandas objects are slow.
    """
    cols = {k: [] for k in ("state", "driver", "lap_number", "compound", "tyre_age_at_lap",
                            "prev_position", "prev_gap_ahead", "prev_gap_to_leader")}
    for i, state in enumerate(states):
        order = sorted(state.positions, key=state.positions.get)
        car_ahead_gap = np.nan
        for driver in order:
            gap = state.gaps.get(driver, np.nan)
            gap = np.nan if gap is None else float(gap)
            compound, age = state.tyres[driver]
            cols["state"].append(i)
            cols["driver"].append(driver)
            cols["lap_number"].append(state.lap + 1)
            cols["compound"].append(str(compound).upper())
            cols["tyre_age_at_lap"].append(int(age) + 1)
            cols["prev_position"].append(float(state.positions[driver]))
            cols["prev_gap_ahead"].append(gap - car_ahead_gap)
            cols["prev_gap_to_leader"].append(gap)
            car_ahead_gap = gap
    return pd.DataFrame(cols).assign(total_laps=total_laps)


class LapTimePredictor:
    """predict_lap_time(state, driver) bound to one race and its real history.

    History is frozen at anchor_lap (read via as_of_lap). Any state with
    state.lap >= anchor_lap can be predicted; earlier states would need
    history the predictor has already discarded, and raise ValueError.
    """

    def __init__(self, race_id: str, anchor_lap: int, model: LapTimeModel | None = None,
                 laps: pd.DataFrame | None = None):
        self.race_id, self.anchor_lap = race_id, anchor_lap
        self.model = model or _default_model()
        race = laps if laps is not None else _all_laps()
        race = race[race["race_id"] == race_id]
        if race.empty:
            raise KeyError(f"no ingested laps for race {race_id!r}")
        self.total_laps = int(race["total_laps"].iloc[0])
        self.history = history_features(race, anchor_lap)

    def predict_lap_time(self, state, driver: str) -> float:
        if driver not in state.positions:
            raise KeyError(f"{driver!r} not in state.positions")
        return self.predict_many([state])[0][driver]

    __call__ = predict_lap_time

    def predict_field(self, state) -> dict[str, float]:
        """Next-lap time for every driver in state, in one model call."""
        return self.predict_many([state])[0]

    def predict_many(self, states: list) -> list[dict[str, float]]:
        """Next-lap times for every driver of every state, in one model call.

        For Monte Carlo: advance all simulations one lap together and call
        this once per lap, rather than once per driver per simulation.
        """
        early = [s.lap for s in states if s.lap < self.anchor_lap]
        if early:
            raise ValueError(f"state at lap {min(early)} is before this predictor's "
                             f"history (lap {self.anchor_lap})")
        rows = _state_rows(states, self.total_laps)
        pace = self.model.predict_rows(assemble_features(self.history, rows, self.anchor_lap))
        sc = np.array([states[i].safety_car for i in rows["state"]], dtype=bool)
        pace = np.where(sc, pace * self.model.sc_ratio, pace)

        out = [{} for _ in states]
        for i, driver, value in zip(rows["state"], rows["driver"], pace.tolist()):
            out[i][driver] = value
        return out


def make_lap_time_predictor(race_id: str, anchor_lap: int) -> Callable:
    """A (state, driver) -> seconds function with real history frozen at anchor_lap."""
    return LapTimePredictor(race_id, anchor_lap)


@lru_cache(maxsize=1)
def _default_model() -> LapTimeModel:
    return LapTimeModel.load(MODEL_DIR)


@lru_cache(maxsize=1)
def _all_laps() -> pd.DataFrame:
    return load_lap_frame()


@lru_cache(maxsize=256)
def _predictor_at(race_id: str, lap: int) -> LapTimePredictor:
    return LapTimePredictor(race_id, lap)


def predict_lap_time(state, driver: str) -> float:
    """Returns predicted next-lap time (seconds) for this driver
    given the current race state.

    Uses real history up to state.lap, so it is for replaying real laps.
    `state` needs a `race_id` attribute. For simulated futures use
    make_lap_time_predictor(race_id, decision_lap) instead.
    """
    race_id = getattr(state, "race_id", None)
    if race_id is None:
        raise AttributeError("predict_lap_time needs state.race_id; for Monte Carlo use "
                             "make_lap_time_predictor(race_id, decision_lap)")
    return _predictor_at(race_id, state.lap)(state, driver)


# --- Evaluation ------------------------------------------------------------

def summarize(rows: pd.DataFrame, pred: np.ndarray) -> dict:
    err = pred - rows["lap_time"].to_numpy()
    persistence = rows["ref_pace"].to_numpy() - rows["lap_time"].to_numpy()
    one = (rows["horizon"] == 1).to_numpy()
    return {
        "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
        "mae_h1": float(np.abs(err[one]).mean()), "rmse_h1": float(np.sqrt((err[one] ** 2).mean())),
        "baseline_mae": float(np.abs(persistence).mean()),
        "baseline_mae_h1": float(np.abs(persistence[one]).mean()),
        "n": int(len(rows)),
    }


def evaluate(rows: pd.DataFrame, sc_ratio: float, late_fraction: float = 0.7) -> dict:
    """Two held-out splits, each vs a 'reference pace' baseline (predict the
    median of the driver's last 5 clean laps):

    - held-out races: train on 7 races, test on the 8th (new circuit);
    - held-out laps: train on laps in the first 70% of every race, test on
      the rest. Training never sees a later lap, so the model can't learn
      e.g. that it rained late in Monaco. (A random driver split was tried
      and rejected for exactly that reason: other drivers' laps from the same
      race leak the conditions.)
    """
    loro_pred = np.full(len(rows), np.nan)
    per_race = {}
    for race_id in sorted(rows["race_id"].unique()):
        test = (rows["race_id"] == race_id).to_numpy()
        model = LapTimeModel(sc_ratio=sc_ratio).fit(rows[~test])
        loro_pred[test] = model.predict_rows(rows[test])
        per_race[race_id] = round(summarize(rows[test], loro_pred[test])["mae_h1"], 3)

    cutoff = late_fraction * rows["total_laps"]
    test = (rows["lap_number"] > cutoff).to_numpy()
    train = (rows["lap_number"] <= cutoff).to_numpy()
    model = LapTimeModel(sc_ratio=sc_ratio).fit(rows[train])

    return {
        "held_out_races": summarize(rows, loro_pred),
        "held_out_races_mae_h1_by_race": per_race,
        "held_out_late_laps": summarize(rows[test], model.predict_rows(rows[test])),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    laps = load_lap_frame()
    rows = build_training_rows(laps)
    sc_ratio = safety_car_ratio(laps)
    log.info("training rows: %d (%d races); safety-car lap ratio %.3f",
             len(rows), rows["race_id"].nunique(), sc_ratio)

    metrics = evaluate(rows, sc_ratio)
    for split, values in metrics.items():
        log.info("%s: %s", split, {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in values.items()})

    model = LapTimeModel(sc_ratio=sc_ratio).fit(rows)
    model.save(MODEL_DIR, metrics=metrics)
    top = list(model.feature_importance().items())[:8]
    log.info("top features (share of gain): %s", [(k, round(v, 3)) for k, v in top])
    log.info("saved to %s", MODEL_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
