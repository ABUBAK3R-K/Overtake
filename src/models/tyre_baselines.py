"""Reference models for the tyre evaluation: anything fancier has to beat these."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.preprocessing.cleaning import DRY_COMPOUNDS


class ZeroModel:
    """Predicts no tyre wear at all."""

    def fit(self, train: pd.DataFrame) -> "ZeroModel":
        return self

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame))


class MeanCurveModel:
    """One monotone pace-loss curve per compound, pooled over every circuit.

    No circuit, temperature or race information: this is the "just use the
    average wear curve" baseline. Isotonic fit keeps wear non-decreasing with
    age, matching the XGBoost model's monotone constraint.
    """

    def __init__(self, max_age: int = 40):
        self.max_age = max_age
        self.curves: dict[str, IsotonicRegression] = {}

    def fit(self, train: pd.DataFrame) -> "MeanCurveModel":
        for compound, g in train.groupby("compound"):
            age = g["tyre_age_at_lap"].astype(float).clip(1, self.max_age)
            self.curves[compound] = IsotonicRegression(out_of_bounds="clip").fit(age, g["pace_loss"])
        return self

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(frame))
        age = frame["tyre_age_at_lap"].astype(float).clip(1, self.max_age).to_numpy()
        for compound in DRY_COMPOUNDS:
            m = (frame["compound"] == compound).to_numpy()
            if m.any():
                out[m] = self.curves[compound].predict(age[m])
        return np.maximum(out, 0.0)
