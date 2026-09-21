"""Held-out evaluation for tyre-degradation models (PRD §8 phase 2).

Whole races and whole circuits are held out, never random laps, using the split
frozen in configs/tyre_split.toml. Two headline errors, both in seconds of
pace loss:

  curve MAE  - mean target vs mean prediction per (race, compound, tyre age),
               cells with >= MIN_LAPS_PER_CELL laps, lap-weighted. Lap-to-lap
               noise (0.3-0.8 s) is averaged out, so this measures the wear
               curve itself.
  lap  MAE / RMSE - per driver-lap, includes that noise.

Confidence intervals resample RACES (not laps): laps in one race share track,
weather and strategy, so lap-level bootstrapping would be far too optimistic.

A model is anything with `fit(train_frame) -> self` and
`predict_frame(frame) -> np.ndarray` (see src.models.tyre.TyreModel).

Model selection must use the inner CV helpers on the TRAIN side only;
`evaluate_on_test` is the one look at the frozen test set.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CONFIG = REPO_ROOT / "configs" / "tyre_split.toml"
MIN_LAPS_PER_CELL = 5
AGE_BUCKETS = [(1, 10, "age 1-10"), (11, 20, "age 11-20"), (21, 99, "age 21+")]


class Model(Protocol):
    def fit(self, train: pd.DataFrame): ...
    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray: ...


def load_split(path: Path = SPLIT_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def split_frame(frame: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = frame["race_id"].isin(split["test_races"])
    # A held-out circuit must be unseen in every era: 2018-2021 races there
    # are dropped from training too, not just the 2022-2024 test races.
    unseen_circuit = frame["circuit"].isin(split["held_out_circuits"])
    train, test = frame[~test_mask & ~unseen_circuit], frame[test_mask]
    leaked = set(train["circuit"]) & set(split["held_out_circuits"])
    if leaked:
        raise AssertionError(f"held-out circuits present in train: {sorted(leaked)}")
    return train, test


# --- Metrics ----------------------------------------------------------------

def _race_sums(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-race sufficient statistics, so a bootstrap is just re-summing."""
    err = preds["pred"] - preds["pace_loss"]
    lap = (pd.DataFrame({"race_id": preds["race_id"], "abs": err.abs(), "sq": err ** 2,
                         "zero_abs": preds["pace_loss"].abs()})
           .groupby("race_id").agg(lap_abs=("abs", "sum"), lap_sq=("sq", "sum"),
                                   zero_lap_abs=("zero_abs", "sum"), n_laps=("abs", "size")))
    cells = (preds.groupby(["race_id", "compound", "tyre_age_at_lap"])
             .agg(t=("pace_loss", "mean"), p=("pred", "mean"), n=("pred", "size")).reset_index())
    cells = cells[cells["n"] >= MIN_LAPS_PER_CELL]
    cells = cells.assign(w_abs=(cells["p"] - cells["t"]).abs() * cells["n"],
                         w_zero=cells["t"].abs() * cells["n"])
    curve = cells.groupby("race_id").agg(curve_abs=("w_abs", "sum"), zero_curve_abs=("w_zero", "sum"),
                                         curve_n=("n", "sum"))
    return lap.join(curve, how="left").fillna(0.0)


def _reduce(sums: pd.DataFrame) -> dict:
    n, cn = sums["n_laps"].sum(), sums["curve_n"].sum()
    return {
        "curve_mae": sums["curve_abs"].sum() / cn if cn else np.nan,
        "zero_curve_mae": sums["zero_curve_abs"].sum() / cn if cn else np.nan,
        "lap_mae": sums["lap_abs"].sum() / n if n else np.nan,
        "lap_rmse": float(np.sqrt(sums["lap_sq"].sum() / n)) if n else np.nan,
    }


def summarise(preds: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Metrics with 95% race-bootstrap intervals. `preds` needs race_id,
    compound, tyre_age_at_lap, pace_loss, pred."""
    sums = _race_sums(preds)
    point = _reduce(sums)
    out = {"n_races": int(len(sums)), "n_laps": int(sums["n_laps"].sum()), **point}
    if len(sums) < 2:
        return out
    rng = np.random.default_rng(seed)
    vals = sums.to_numpy()
    cols = list(sums.columns)
    boots = []
    for _ in range(n_boot):
        pick = vals[rng.integers(0, len(vals), len(vals))]
        boots.append(_reduce(pd.DataFrame(pick, columns=cols)))
    boots = pd.DataFrame(boots)
    for k in ("curve_mae", "lap_mae", "lap_rmse"):
        out[f"{k}_ci"] = (float(boots[k].quantile(0.025)), float(boots[k].quantile(0.975)))
    return out


def breakdown(preds: pd.DataFrame, split: dict, tags: dict, **kw) -> dict[str, dict]:
    """Overall plus unseen/seen circuit, wet/dry, compound and age slices.

    Slices by compound/age filter laps, then aggregate the same way, so
    their curve MAE is over the cells that fall inside the slice.
    """
    unseen = preds["circuit"].isin(split["held_out_circuits"])
    wet = preds["race_id"].isin(split["wet_test_races"])
    slices = {"all": preds.index == preds.index,
              "unseen_circuit": unseen, "seen_circuit": ~unseen,
              "wet_races": wet, "dry_races": ~wet,
              "unseen_circuit_dry": unseen & ~wet}
    for c in sorted(preds["compound"].unique()):
        slices[f"compound_{c}"] = preds["compound"] == c
    for lo, hi, name in AGE_BUCKETS:
        slices[name] = preds["tyre_age_at_lap"].between(lo, hi)
    out = {name: summarise(preds[np.asarray(m)], **kw) for name, m in slices.items()
           if np.asarray(m).any()}
    # Per-race curve MAE, for eyeballing which races drive the average.
    for race_id, p in preds.groupby("race_id"):
        out[f"race:{race_id}"] = {k: v for k, v in summarise(p, n_boot=0).items()
                                  if k in ("curve_mae", "lap_mae", "n_laps")}
    return out


# --- Inner CV (train side only; use for model selection) ---------------------

def leave_one_circuit_out(train: pd.DataFrame, make_model: Callable[[], Model],
                          min_races: int = 2) -> pd.DataFrame:
    """Predictions for each circuit from a model that never saw that circuit.
    Circuits with fewer than `min_races` races in `train` are skipped."""
    parts = []
    races_per = train.groupby("circuit")["race_id"].nunique()
    for circuit in races_per[races_per >= min_races].index:
        held = train["circuit"] == circuit
        model = make_model().fit(train[~held])
        test = train[held]
        parts.append(test.assign(pred=model.predict_frame(test)))
    return pd.concat(parts)


def leave_one_race_out(train: pd.DataFrame, make_model: Callable[[], Model],
                       races: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for race_id in races or sorted(train["race_id"].unique()):
        held = train["race_id"] == race_id
        model = make_model().fit(train[~held])
        test = train[held]
        parts.append(test.assign(pred=model.predict_frame(test)))
    return pd.concat(parts)


# --- The one look at the frozen test set -------------------------------------

def evaluate_on_test(frame: pd.DataFrame, make_model: Callable[[], Model], tags: dict,
                     split: dict | None = None, **kw) -> dict[str, dict]:
    split = split or load_split()
    train, test = split_frame(frame, split)
    model = make_model().fit(train)
    preds = test.assign(pred=model.predict_frame(test))
    return breakdown(preds, split, tags, **kw)
