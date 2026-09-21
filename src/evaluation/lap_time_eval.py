"""Held-out evaluation for lap-time models (PRD §8 phase 3 / FR-3).

Compares:
  1. Persistence baseline: repeat the driver's last-5 clean lap median (ref_pace)
  2. XGBoost baseline: independent tabular regressor (src.models.lap_time)
  3. GRU baseline: independent sequential regressor (src.models.lap_time_gru)
  4. GNN model: interaction-aware relational graph model (src.models.lap_time_gnn)

Whole races and whole circuits are held out using the frozen split in
configs/tyre_split.toml.

Headline metrics:
  - Lap MAE and RMSE (seconds)
  - Next-lap (horizon = 1) MAE and RMSE
  - Stratified slices:
      * clean_air: car is > 3.0s behind car ahead
      * in_traffic: car is <= 2.0s behind car ahead (dirty air wake)
      * drs_range: car is <= 1.0s behind car ahead
      * seen_circuit vs unseen_circuit
      * dry_races vs wet_races
  - 95% bootstrap confidence intervals resampling whole RACES.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.tyre import REPO_ROOT

SPLIT_CONFIG = REPO_ROOT / "configs" / "tyre_split.toml"


def load_split(path: Path = SPLIT_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def split_laps(laps: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split lap dataframe into train and test according to the frozen split."""
    test_mask = laps["race_id"].isin(split["test_races"])
    unseen_circuit = laps["circuit"].isin(split["held_out_circuits"]) if "circuit" in laps else False
    train = laps[~test_mask & ~unseen_circuit].copy()
    test = laps[test_mask].copy()
    return train, test


# --- Metric Computations ---------------------------------------------------

def _race_sums(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-race summary statistics for bootstrap resampling."""
    err = preds["pred"] - preds["lap_time"]
    abs_err = err.abs()
    sq_err = err ** 2

    df = pd.DataFrame({
        "race_id": preds["race_id"],
        "abs_err": abs_err,
        "sq_err": sq_err,
        "n": 1,
    })

    if "ref_pace" in preds:
        base_err = (preds["ref_pace"] - preds["lap_time"]).abs()
        df["base_abs"] = base_err
    else:
        df["base_abs"] = 0.0

    if "horizon" in preds:
        df["is_h1"] = (preds["horizon"] == 1).astype(float)
        df["abs_h1"] = abs_err * df["is_h1"]
        df["sq_h1"] = sq_err * df["is_h1"]
    else:
        df["is_h1"] = 1.0
        df["abs_h1"] = abs_err
        df["sq_h1"] = sq_err

    return df.groupby("race_id").agg({
        "abs_err": "sum",
        "sq_err": "sum",
        "base_abs": "sum",
        "n": "sum",
        "is_h1": "sum",
        "abs_h1": "sum",
        "sq_h1": "sum",
    })


def _reduce(sums: pd.DataFrame) -> dict[str, float]:
    n = float(sums["n"].sum())
    if n == 0:
        return {"lap_mae": np.nan, "lap_rmse": np.nan, "mae_h1": np.nan, "rmse_h1": np.nan, "baseline_mae": np.nan}

    nh1 = float(sums["is_h1"].sum())
    return {
        "lap_mae": float(sums["abs_err"].sum() / n),
        "lap_rmse": float(np.sqrt(sums["sq_err"].sum() / n)),
        "baseline_mae": float(sums["base_abs"].sum() / n),
        "mae_h1": float(sums["abs_h1"].sum() / nh1) if nh1 > 0 else np.nan,
        "rmse_h1": float(np.sqrt(sums["sq_h1"].sum() / nh1)) if nh1 > 0 else np.nan,
    }


def summarise(preds: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Calculate lap metrics and 95% race-level bootstrap intervals."""
    if preds.empty:
        return {"n_races": 0, "n_laps": 0, "lap_mae": np.nan, "lap_rmse": np.nan}

    sums = _race_sums(preds)
    point = _reduce(sums)
    out: dict[str, Any] = {
        "n_races": int(len(sums)),
        "n_laps": int(sums["n"].sum()),
        **point,
    }

    if len(sums) < 2 or n_boot <= 0:
        return out

    rng = np.random.default_rng(seed)
    vals = sums.to_numpy()
    cols = list(sums.columns)
    boots = []
    for _ in range(n_boot):
        pick = vals[rng.integers(0, len(vals), len(vals))]
        boots.append(_reduce(pd.DataFrame(pick, columns=cols)))

    boots_df = pd.DataFrame(boots)
    for k in ("lap_mae", "lap_rmse", "mae_h1"):
        if k in boots_df:
            out[f"{k}_ci"] = (
                round(float(boots_df[k].quantile(0.025)), 3),
                round(float(boots_df[k].quantile(0.975)), 3),
            )

    return out


def breakdown(preds: pd.DataFrame, split: dict, tags: dict | None = None, **kw) -> dict[str, Any]:
    """Overall metrics plus slices by circuit familiarity, weather, and traffic density."""
    unseen = preds["circuit"].isin(split["held_out_circuits"]) if "circuit" in preds else pd.Series(False, index=preds.index)
    wet = preds["race_id"].isin(split["wet_test_races"])

    slices: dict[str, Any] = {
        "all": preds.index == preds.index,
        "unseen_circuit": unseen,
        "seen_circuit": ~unseen,
        "wet_races": wet,
        "dry_races": ~wet,
    }

    # Horizon slice
    if "horizon" in preds:
        slices["next_lap_h1"] = preds["horizon"] == 1
        slices["multi_lap_h_gt_1"] = preds["horizon"] > 1

    # Traffic & aerodynamic interaction slices (The Core FR-3 Test)
    if "prev_gap_ahead" in preds:
        gap_ahead = preds["prev_gap_ahead"].fillna(999.0)
        slices["clean_air"] = gap_ahead > 3.0
        slices["in_traffic_wake"] = (gap_ahead > 0.0) & (gap_ahead <= 2.0)
        slices["drs_range"] = (gap_ahead > 0.0) & (gap_ahead <= 1.0)

    out: dict[str, Any] = {}
    for name, mask in slices.items():
        sub = preds[np.asarray(mask)]
        if not sub.empty:
            out[name] = summarise(sub, **kw)

    # Per-race breakdown
    for race_id, p in preds.groupby("race_id"):
        out[f"race:{race_id}"] = {
            k: v for k, v in summarise(p, n_boot=0).items()
            if k in ("lap_mae", "lap_rmse", "mae_h1", "n_laps")
        }

    return out
