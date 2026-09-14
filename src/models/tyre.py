"""Tyre degradation model (Design.md Section 6.3).

Hands off predict_tyre_degradation(compound, age, circuit, track_temp) to the
Decision-Making lane.

What "pace loss" means
----------------------
Seconds per lap lost to tyre wear, compared with the same compound when it
was fresh (age <= reference_max_age in configs/models.toml). It is ~0 for a
new tyre and rises with age. It does NOT include the pace gap between
compounds (soft vs hard) or fuel burn-off; those belong to the lap-time model.

How the training target is built
--------------------------------
Within a stint, tyre age and lap number rise together, so wear can't be told
apart from fuel burn-off by looking at one stint. Per race we fit

    lap_time = lap effect + driver effect + compound offset + wear(compound, age)

where the lap effect (fuel, track evolution, conditions) is shared by every
car on that lap, and wear is a piecewise-linear function of age per compound,
pinned to 0 at the reference ages. The target for each lap is its lap time
minus the lap, driver and compound terms, i.e. wear plus that lap's noise.
Identification comes from cars on the same lap running different tyre ages
and from a driver's pace resetting after a pit stop.

Building targets uses whole-race data, which is fine for *labels* on
historical races. The model's inputs (compound, age, circuit, track temp)
are all current-state values, so prediction at lap N needs nothing after N.

Usage:
    python -m src.models.tyre          # evaluate, train on all races, save
"""

from __future__ import annotations

import json
import logging
import sys
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.preprocessing.cleaning import DRY_COMPOUNDS, is_clean_lap

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = REPO_ROOT / "configs" / "models.toml"
MODEL_DIR = REPO_ROOT / "data" / "models" / "tyre"

# Laps past the reference age where the wear slope may change when building
# training targets (the XGBoost model itself is not limited to this shape).
WEAR_KNOTS_AFTER_REFERENCE = (10, 20)

log = logging.getLogger("overtake.models.tyre")


def load_config() -> dict:
    with open(MODELS_CONFIG, "rb") as f:
        return tomllib.load(f)["tyre"]


# --- Data ------------------------------------------------------------------

def load_lap_frame(con=None) -> pd.DataFrame:
    """Laps joined with tyre state, circuit and (if ingested) per-lap track temp.

    track_temp comes from the Decision-Making lane's `weather` table
    (race_id, lap, track_temp). Until that table exists the column is NaN and
    the model trains without it.
    """
    if con is None:
        from src.ingestion.storage import connect
        con = connect()

    tables = {row[0] for row in con.sql("SHOW TABLES").fetchall()}
    weather_cols = (
        set(con.sql("SELECT * FROM weather LIMIT 0").columns) if "weather" in tables else set()
    )
    has_weather = {"race_id", "lap", "track_temp"} <= weather_cols

    weather_join = (
        "LEFT JOIN (SELECT race_id, lap, AVG(track_temp) AS track_temp "
        "           FROM weather GROUP BY race_id, lap) w "
        "  ON w.race_id = l.race_id AND w.lap = l.lap_number"
        if has_weather else ""
    )
    track_temp = "w.track_temp" if has_weather else "CAST(NULL AS DOUBLE)"
    return con.sql(f"""
        SELECT l.race_id, l.driver, l.lap_number, l.lap_time,
               l.pit_flag, l.pit_out_flag, l.track_status, l.is_accurate,
               t.stint, t.compound, t.tyre_age_at_lap, r.circuit,
               {track_temp} AS track_temp
        FROM laps l
        JOIN tyres t USING (race_id, driver, lap_number)
        JOIN races r ON r.race_id = l.race_id
        {weather_join}
        ORDER BY l.race_id, l.lap_number, l.driver
    """).df()


def _race_pace_loss(race: pd.DataFrame, reference_max_age: int, max_age: int) -> pd.Series:
    """Wear target for one race's clean dry laps (see module docstring)."""
    age = race["tyre_age_at_lap"].astype(int).clip(upper=max_age)
    nuisance = pd.concat([
        pd.get_dummies(race["lap_number"], prefix="lap"),
        pd.get_dummies(race["driver"], prefix="drv"),
        pd.get_dummies(race["compound"], prefix="cmp", drop_first=True),
    ], axis=1).astype(float)

    # Piecewise-linear wear per compound: zero up to the reference age, with
    # the slope free to change at each knot. A free value per age was tried
    # first; at low-stop circuits (Monaco, Singapore) it trades off against the
    # lap effects and produced multi-second swings.
    laps_past_ref = (age - reference_max_age).clip(lower=0).astype(float)
    edges = [0, *WEAR_KNOTS_AFTER_REFERENCE, None]
    wear = {}
    for compound in sorted(race["compound"].unique()):
        on_compound = (race["compound"] == compound).astype(float)
        for lo, hi in zip(edges[:-1], edges[1:]):
            segment = (laps_past_ref - lo).clip(lower=0)
            if hi is not None:
                segment = segment.clip(upper=hi - lo)
            wear[f"wear_{compound}_{lo}"] = segment * on_compound
    wear = pd.DataFrame(wear, index=race.index)

    X = pd.concat([nuisance, wear], axis=1).to_numpy()
    beta, *_ = np.linalg.lstsq(X, race["lap_time"].to_numpy(), rcond=None)
    fitted_nuisance = nuisance.to_numpy() @ beta[: nuisance.shape[1]]
    return race["lap_time"] - fitted_nuisance


def build_training_frame(laps: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Clean dry laps with a `pace_loss` target, one row per driver-lap."""
    config = config or load_config()
    ref_age, max_age = config["reference_max_age"], config["max_age"]

    rows = laps[is_clean_lap(laps) & laps["compound"].isin(DRY_COMPOUNDS)].copy()

    # Keep only race-compounds with enough laps, and enough fresh-tyre laps
    # to pin the zero point; otherwise wear for that compound is unidentified.
    by = rows.groupby(["race_id", "compound"])
    enough = (
        (by["lap_time"].transform("size") >= config["min_laps"])
        & (by["tyre_age_at_lap"].transform(lambda a: (a <= ref_age).sum())
           >= config["min_reference_laps"])
    )
    rows = rows[enough]

    parts = [
        race.assign(pace_loss=_race_pace_loss(race, ref_age, max_age))
        for _, race in rows.groupby("race_id")
    ]
    return pd.concat(parts).sort_index() if parts else rows.assign(pace_loss=[])


# --- Model -----------------------------------------------------------------

@dataclass
class TyreModel:
    variant: str = "pooled"                      # "pooled" | "per_compound"
    params: dict = field(default_factory=dict)
    max_age: int = 40
    use_track_temp: bool = False
    circuits: list[str] = field(default_factory=list)
    boosters: dict[str, xgb.XGBRegressor] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        names = ["tyre_age", "circuit"]
        if self.variant == "pooled":
            names.insert(0, "compound")
        if self.use_track_temp:
            names.append("track_temp")
        return names

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Model inputs from a frame with compound, tyre_age_at_lap, circuit, track_temp."""
        X = pd.DataFrame({
            "compound": pd.Categorical(frame["compound"], categories=list(DRY_COMPOUNDS)),
            "tyre_age": frame["tyre_age_at_lap"].astype(float).clip(1, self.max_age),
            "circuit": pd.Categorical(frame["circuit"], categories=self.circuits),
            "track_temp": frame["track_temp"].astype(float),
        }, index=frame.index)
        return X[self.feature_names]

    def _new_booster(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            **self.params,
            tree_method="hist",
            enable_categorical=True,
            # Wear can only grow with age; stops noisy targets producing dips.
            monotone_constraints={"tyre_age": 1},
        )

    def fit(self, train: pd.DataFrame) -> "TyreModel":
        self.use_track_temp = bool(train["track_temp"].notna().any())
        self.circuits = sorted(train["circuit"].unique())
        groups = ({"all": train} if self.variant == "pooled"
                  else dict(iter(train.groupby("compound"))))
        self.boosters = {
            key: self._new_booster().fit(self._features(g), g["pace_loss"])
            for key, g in groups.items()
        }
        return self

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Vectorised prediction; use this in loops over many drivers/laps.

        A circuit not seen in training gets the average prediction over all
        training circuits. (Left as a missing category, XGBoost routes it down
        an arbitrary branch: held-out Australia was scored like Singapore.)

        Predictions are floored at 0: pace loss is measured against a fresh
        tyre, so a negative value would make a new set faster than new.
        """
        unknown = set(frame["compound"]) - set(DRY_COMPOUNDS)
        if unknown:
            raise ValueError(f"tyre model covers dry compounds {DRY_COMPOUNDS}, got {sorted(unknown)}")

        frame = frame.reset_index(drop=True)
        unseen = ~frame["circuit"].isin(self.circuits)
        out = np.full(len(frame), np.nan)
        if (~unseen).any():
            out[~unseen] = self._predict_known(frame[~unseen])
        if unseen.any():
            rows = frame[unseen]
            expanded = rows.loc[rows.index.repeat(len(self.circuits))].assign(
                circuit=np.tile(self.circuits, len(rows)))
            out[unseen] = self._predict_known(expanded).reshape(len(rows), -1).mean(axis=1)
        return np.maximum(out, 0.0)

    def _predict_known(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.reset_index(drop=True)
        if self.variant == "pooled":
            return self.boosters["all"].predict(self._features(frame))

        out = np.full(len(frame), np.nan)
        for compound, idx in frame.groupby("compound").indices.items():
            if compound not in self.boosters:
                raise ValueError(f"no {compound} tyre model was trained")
            out[idx] = self.boosters[compound].predict(self._features(frame.iloc[idx]))
        return out

    def save(self, directory: Path = MODEL_DIR, metrics: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for key, booster in self.boosters.items():
            booster.save_model(directory / f"booster_{key}.json")
        meta = {
            "variant": self.variant, "params": self.params, "max_age": self.max_age,
            "use_track_temp": self.use_track_temp, "circuits": self.circuits,
            "boosters": sorted(self.boosters), "metrics": metrics or {},
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> "TyreModel":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no tyre model at {directory}; train one with `python -m src.models.tyre`"
            )
        meta = json.loads(meta_path.read_text())
        model = cls(variant=meta["variant"], params=meta["params"], max_age=meta["max_age"],
                    use_track_temp=meta["use_track_temp"], circuits=meta["circuits"])
        for key in meta["boosters"]:
            booster = model._new_booster()
            booster.load_model(directory / f"booster_{key}.json")
            model.boosters[key] = booster
        return model


# --- Handoff function (Design.md Section 5) --------------------------------

@lru_cache(maxsize=1)
def _default_model() -> TyreModel:
    return TyreModel.load(MODEL_DIR)


@lru_cache(maxsize=65536)
def _predict_cached(compound: str, age: int, circuit: str, track_temp: float) -> float:
    frame = pd.DataFrame({"compound": [compound], "tyre_age_at_lap": [age],
                          "circuit": [circuit], "track_temp": [track_temp]})
    return float(_default_model().predict_frame(frame)[0])


def predict_tyre_degradation(compound: str, age: int, circuit: str,
                              track_temp: float) -> float:
    """Returns predicted pace loss (seconds) for this tyre state.

    compound:   SOFT / MEDIUM / HARD (case-insensitive). Wet compounds raise ValueError.
    age:        tyre age in laps (FastF1 TyreLife, as in the tyres table).
    circuit:    `races.circuit`, e.g. "Sakhir". Unseen circuits still get a
                prediction, from the model's circuit-agnostic branch.
    track_temp: degrees C at this lap. Ignored if the model was trained without it.

    Results are memoised (temp rounded to 0.5 C), so calling this for every
    driver on every simulated lap is cheap after the first few thousand calls.
    """
    temp = float("nan") if track_temp is None else round(float(track_temp) * 2) / 2
    return _predict_cached(str(compound).upper(), int(age), str(circuit), temp)


# --- Evaluation ------------------------------------------------------------

def leave_one_race_out(train: pd.DataFrame, variant: str, config: dict) -> pd.DataFrame:
    """Predictions for every race from a model trained on the other races."""
    parts = []
    for race_id in sorted(train["race_id"].unique()):
        held_out = train["race_id"] == race_id
        model = TyreModel(variant=variant, params=config["xgboost"],
                          max_age=config["max_age"]).fit(train[~held_out])
        test = train[held_out]
        parts.append(test.assign(pred=model.predict_frame(test)))
    return pd.concat(parts)


def held_out_stints(train: pd.DataFrame, variant: str, config: dict,
                    fraction: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Predictions for a random 20% of driver-stints, from a model trained on
    the rest. Circuits are all seen in training, as in the replay demo.
    Whole stints are held out because laps within a stint are near-duplicates.
    """
    stint_key = train["race_id"] + "|" + train["driver"] + "|" + train["stint"].astype(str)
    stints = stint_key.unique()
    rng = np.random.default_rng(seed)
    test_stints = rng.choice(stints, size=int(len(stints) * fraction), replace=False)
    held_out = stint_key.isin(test_stints)
    model = TyreModel(variant=variant, params=config["xgboost"],
                      max_age=config["max_age"]).fit(train[~held_out])
    test = train[held_out]
    return test.assign(pred=model.predict_frame(test))


def summarize_predictions(preds: pd.DataFrame, min_laps_per_point: int = 5) -> dict:
    """Per-lap error, plus error on the average wear curve.

    Per-lap error is dominated by lap-to-lap noise (0.3-0.8 s) that no tyre
    model can explain, so curve error (mean target vs mean prediction per
    race, compound and age) is the fairer measure of the wear curve itself.
    """
    err = preds["pred"] - preds["pace_loss"]
    curve = (preds.groupby(["race_id", "compound", "tyre_age_at_lap"])
             .agg(target=("pace_loss", "mean"), pred=("pred", "mean"), n=("pred", "size")))
    curve = curve[curve["n"] >= min_laps_per_point]
    curve_err = (curve["pred"] - curve["target"]).abs()
    return {
        "lap_mae": float(err.abs().mean()),
        "lap_rmse": float(np.sqrt((err ** 2).mean())),
        "curve_mae": float(np.average(curve_err, weights=curve["n"])),
        "zero_model_lap_mae": float(preds["pace_loss"].abs().mean()),
        "zero_model_curve_mae": float(np.average(curve["target"].abs(), weights=curve["n"])),
        "n_laps": int(len(preds)),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()

    train = build_training_frame(load_lap_frame(), config)
    log.info("training rows: %d across %d races; track_temp %s",
             len(train), train["race_id"].nunique(),
             "available" if train["track_temp"].notna().any() else "not ingested yet")

    results = {"leave_one_race_out": {}, "held_out_stints": {}}
    for variant in ("pooled", "per_compound"):
        preds = leave_one_race_out(train, variant, config)
        results["leave_one_race_out"][variant] = summarize_predictions(preds)
        per_race = {race_id: round(summarize_predictions(p)["curve_mae"], 3)
                    for race_id, p in preds.groupby("race_id")}
        results["held_out_stints"][variant] = summarize_predictions(
            held_out_stints(train, variant, config))
        for split in results:
            log.info("%s %s: %s", variant, split,
                     {k: round(v, 3) for k, v in results[split][variant].items()})
        log.info("%s curve MAE by held-out race: %s", variant, per_race)

    variant = config["variant"]
    if variant == "auto":
        loro = results["leave_one_race_out"]
        variant = min(loro, key=lambda v: loro[v]["curve_mae"])
    log.info("training final %s model on all races", variant)

    model = TyreModel(variant=variant, params=config["xgboost"],
                      max_age=config["max_age"]).fit(train)
    model.save(MODEL_DIR, metrics={**results, "chosen": variant})
    log.info("saved to %s", MODEL_DIR)

    curve = pd.DataFrame(
        {c: [predict_with(model, c, a, train) for a in (1, 5, 10, 20, 30)] for c in DRY_COMPOUNDS},
        index=pd.Index([1, 5, 10, 20, 30], name="age"),
    )
    log.info("mean predicted pace loss (s) across training circuits:\n%s", curve.round(3))
    return 0


def predict_with(model: TyreModel, compound: str, age: int, train: pd.DataFrame) -> float:
    """Average prediction for one tyre state across all training circuits."""
    circuits = train[["circuit"]].drop_duplicates()
    frame = circuits.assign(compound=compound, tyre_age_at_lap=age, track_temp=np.nan)
    return float(model.predict_frame(frame).mean())


if __name__ == "__main__":
    sys.exit(main())
