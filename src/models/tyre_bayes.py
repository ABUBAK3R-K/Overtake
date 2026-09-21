"""Hierarchical Bayesian tyre-wear model (PRD FR-2; NumPy/SciPy only).

Model, per lap i on compound c at circuit j with tyre age a:

    pace_loss_i = sum_k (mu[c,k] + delta[j,c,k]) * seg_k(a) + eps_i
    mu[c,k]       ~ N(0, s0^2)        compound-level wear slope, seconds/lap in age segment k
    delta[j,c,k]  ~ N(0, tau_k^2)     circuit deviation from the compound slope
    eps_i         ~ N(0, sigma^2 / w_i)   w_i = race train_weight (0.4 for 2018-2021)

seg_k are piecewise-linear hinge features after the reference age (wear is 0
at the reference tyre age, matching how targets are built in tyre.py). The
model is linear-Gaussian, so the posterior over (mu, delta) is exact. The
noise scale and the circuit spreads tau_k are set by maximising the marginal
likelihood (empirical Bayes).

Why this shape: a circuit seen in training gets a partly-pooled curve (data
pull it away from the compound average); an unseen circuit falls back to the
compound average with its extra uncertainty tau_k included in the intervals.
This is a hierarchical regression, not a filtering state-space model: tyre
age already indexes the state, so nothing is filtered along the stint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import norm

from src.preprocessing.cleaning import DRY_COMPOUNDS

# Age-segment edges, in laps past the reference age; last segment is open-ended.
SEGMENT_EDGES = (0, 5, 10, 20, None)
K = len(SEGMENT_EDGES) - 1


def _segments(age: np.ndarray, ref_age: int, max_age: int) -> np.ndarray:
    past = np.clip(np.clip(age, 1, max_age) - ref_age, 0, None).astype(float)
    cols = []
    for lo, hi in zip(SEGMENT_EDGES[:-1], SEGMENT_EDGES[1:]):
        seg = np.clip(past - lo, 0, None)
        cols.append(seg if hi is None else np.clip(seg, None, hi - lo))
    return np.column_stack(cols)                       # (n, K)


class BayesTyreModel:
    def __init__(self, reference_max_age: int = 3, max_age: int = 40, s0: float = 0.2):
        self.ref_age, self.max_age, self.s0 = reference_max_age, max_age, s0
        self.circuits: list[str] = []

    # --- design ------------------------------------------------------------
    def _design(self, frame: pd.DataFrame, circuit_index: np.ndarray) -> sp.csr_matrix:
        """Columns: mu block (3K) then, per circuit, a delta block (3K).
        circuit_index = -1 means an unseen circuit (delta columns left at 0)."""
        n, nb = len(frame), len(DRY_COMPOUNDS) * K
        seg = _segments(frame["tyre_age_at_lap"].to_numpy(dtype=float), self.ref_age, self.max_age)
        comp = frame["compound"].map({c: i for i, c in enumerate(DRY_COMPOUNDS)}).to_numpy()
        n_cols = nb * (1 + len(self.circuits))
        known = circuit_index >= 0
        rows, cols, vals = [], [], []
        for k in range(K):
            everyone = np.arange(n)
            rows.append(everyone); cols.append(comp * K + k); vals.append(seg[:, k])
            seen = everyone[known]                     # delta block: circuit's own columns
            rows.append(seen); cols.append(nb * (1 + circuit_index[seen]) + comp[seen] * K + k)
            vals.append(seg[seen, k])
        rows, cols, vals = map(np.concatenate, (rows, cols, vals))
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n_cols))

    def _circuit_index(self, frame: pd.DataFrame) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(self.circuits)}
        return frame["circuit"].map(lookup).fillna(-1).astype(int).to_numpy()

    def _prior_precision(self, log_tau: np.ndarray) -> np.ndarray:
        nb = len(DRY_COMPOUNDS) * K
        tau = np.exp(log_tau)                                  # (K,)
        p = [np.full(nb, 1 / self.s0 ** 2)]
        for _ in self.circuits:
            p.append(np.tile(1 / tau ** 2, len(DRY_COMPOUNDS)))
        return np.concatenate(p)

    # --- fitting -----------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "BayesTyreModel":
        self.circuits = sorted(train["circuit"].unique())
        X = self._design(train, self._circuit_index(train))
        y = train["pace_loss"].to_numpy(dtype=float)
        w = (train["train_weight"].to_numpy(dtype=float) if "train_weight" in train
             else np.ones(len(train)))
        XtW = X.T.multiply(w)                                   # X' W
        A = (XtW @ X).toarray()
        b = np.asarray(XtW @ y).ravel()
        yWy, n = float(y @ (w * y)), len(y)
        logw = float(np.log(w).sum())

        def neg_log_evidence(theta):
            log_sigma, log_tau = theta[0], theta[1:]
            s2 = np.exp(2 * log_sigma)
            P = self._prior_precision(log_tau)
            Lam = A / s2 + np.diag(P)
            L = np.linalg.cholesky(Lam)
            z = np.linalg.solve(L, b / s2)
            logdet_lam = 2 * np.log(np.diag(L)).sum()
            ll = (-0.5 * n * np.log(2 * np.pi * s2) + 0.5 * logw - 0.5 * yWy / s2
                  + 0.5 * z @ z - 0.5 * logdet_lam + 0.5 * np.log(P).sum())
            return -ll

        x0 = np.concatenate([[np.log(0.7)], np.full(K, np.log(0.05))])
        res = minimize(neg_log_evidence, x0, method="L-BFGS-B",
                       bounds=[(-3, 1)] + [(-8, 0)] * K)
        self.sigma = float(np.exp(res.x[0]))
        self.tau = np.exp(res.x[1:])
        P = self._prior_precision(res.x[1:])
        self._Lam = A / self.sigma ** 2 + np.diag(P)
        self._cov = np.linalg.inv(self._Lam)
        self.theta = self._cov @ b / self.sigma ** 2
        return self

    # --- prediction --------------------------------------------------------
    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.reset_index(drop=True)
        X = self._design(frame, self._circuit_index(frame))
        return np.maximum(np.asarray(X @ self.theta).ravel(), 0.0)

    def predict_interval(self, frame: pd.DataFrame, level: float = 0.9,
                         include_lap_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Central interval for the mean wear curve (or a single lap if
        include_lap_noise). Unseen circuits add the circuit spread tau_k."""
        frame = frame.reset_index(drop=True)
        idx = self._circuit_index(frame)
        X = self._design(frame, idx)
        mean = np.asarray(X @ self.theta).ravel()
        var = np.asarray(X.multiply(X @ self._cov).sum(axis=1)).ravel()
        unseen = idx < 0
        if unseen.any():
            seg = _segments(frame["tyre_age_at_lap"].to_numpy(dtype=float), self.ref_age, self.max_age)
            var[unseen] += ((seg ** 2) * self.tau ** 2).sum(axis=1)[unseen]
        if include_lap_noise:
            var = var + self.sigma ** 2
        z = norm.ppf(0.5 + level / 2)
        sd = np.sqrt(var)
        return np.maximum(mean - z * sd, 0.0), np.maximum(mean + z * sd, 0.0)
