"""Sequential GRU Baseline for Lap-Time Modeling (PRD.md FR-3 Track a).

Treats each driver independently without cross-car interaction features.
For a given driver at anchor lap L, feeds their sequence of recent clean laps
into a 2-layer GRU to predict next lap time delta vs their reference pace:
    lap_time(L+1) = ref_pace + GRU(history_sequence)

Usage:
    from src.models.lap_time_gru import LapTimeGRUModel
    model = LapTimeGRUModel()
    model.fit(race_laps)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.tyre import REPO_ROOT, load_lap_frame
from src.preprocessing.cleaning import is_clean_lap
from src.preprocessing.leakage import as_of_lap

MODEL_DIR = REPO_ROOT / "data" / "models" / "lap_time_gru"
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

SEQ_LEN = 5  # Number of past clean laps in driver's sequence
INPUT_DIM = 8  # [vs_field, tyre_age_norm, lap_fraction, 5 one-hot compound]

log = logging.getLogger("overtake.models.lap_time_gru")


class GRUNetwork(nn.Module):
    """2-layer GRU with projection head predicting next lap delta vs reference pace."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, seq_len, input_dim) -> (batch_size, 1)"""
        out, _ = self.gru(x)
        # Take the hidden representation of the final timestep
        last_step = out[:, -1, :]
        return self.head(last_step)


def extract_driver_sequences(
    race_laps: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Extract (sequence, target_delta, ref_pace, metadata) for all drivers and laps.

    Only clean laps are used. Target is lap_time(L+1) - ref_pace(L).
    """
    clean = race_laps[is_clean_lap(race_laps)].sort_values(["driver", "lap_number"]).copy()
    if clean.empty:
        return np.zeros((0, seq_len, INPUT_DIM)), np.zeros(0), np.zeros(0), []

    # Lap median pace across field
    lap_medians = clean.groupby("lap_number")["lap_time"].median().to_dict()
    total_laps_val = float(clean["total_laps"].iloc[0]) if "total_laps" in clean else float(clean["lap_number"].max())

    sequences, targets, ref_paces, meta = [], [], [], []

    for driver, group in clean.groupby("driver"):
        group = group.sort_values("lap_number").reset_index(drop=True)
        if len(group) < seq_len + 1:
            continue

        laps_num = group["lap_number"].to_numpy()
        lap_times = group["lap_time"].to_numpy()
        tyre_ages = group["tyre_age_at_lap"].to_numpy()
        compounds = group["compound"].to_numpy()

        for i in range(seq_len, len(group)):
            # Past clean laps sequence: i-seq_len to i-1
            seq_window = []
            for k in range(i - seq_len, i):
                lap_n = laps_num[k]
                l_med = lap_medians.get(lap_n, lap_times[k])
                vs_field = lap_times[k] - l_med
                age_norm = float(tyre_ages[k]) / 50.0
                lap_frac = float(lap_n) / total_laps_val if total_laps_val else 0.5

                c_idx = COMPOUND_TO_IDX.get(str(compounds[k]).upper(), 1)
                c_one_hot = [1.0 if j == c_idx else 0.0 for j in range(len(COMPOUNDS))]

                seq_window.append([vs_field, age_norm, lap_frac, *c_one_hot])

            ref_pace = float(np.median(lap_times[i - seq_len : i]))
            target_delta = float(lap_times[i] - ref_pace)

            sequences.append(seq_window)
            targets.append(target_delta)
            ref_paces.append(ref_pace)
            meta.append({
                "driver": driver,
                "anchor_lap": int(laps_num[i - 1]),
                "target_lap": int(laps_num[i]),
                "race_id": str(group["race_id"].iloc[0]) if "race_id" in group else "unknown",
            })

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(ref_paces, dtype=np.float32), meta


class LapTimeGRUModel:
    """Sequential GRU model for driver-independent lap-time forecasting."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        sc_ratio: float = 1.5,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = learning_rate
        self.sc_ratio = sc_ratio
        self.network = GRUNetwork(
            input_dim=INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def fit(
        self,
        train_laps: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> "LapTimeGRUModel":
        """Fit the GRU model on historical lap frame."""
        X, y, _, _ = extract_driver_sequences(train_laps)
        if len(X) == 0:
            log.warning("No sequences extracted for GRU training.")
            return self

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        # Huber / Smooth L1 loss for robustness to lap-time noise and outliers
        criterion = nn.SmoothL1Loss()

        self.network.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.network(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_x)

            if verbose and (epoch + 1) % 5 == 0:
                log.info("GRU Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, total_loss / len(X))

        return self

    def predict_sequence(self, sequence: np.ndarray) -> float:
        """Predict delta vs ref_pace given a single driver's history sequence of shape (SEQ_LEN, INPUT_DIM)."""
        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            pred_delta = float(self.network(x_t).item())
        return pred_delta

    def predict_driver_next_lap(
        self,
        race_laps: pd.DataFrame,
        driver: str,
        current_lap: int,
    ) -> float:
        """Predict driver's next lap time given race laps up to current_lap."""
        visible = as_of_lap(race_laps, "lap_number", current_lap)
        d_clean = visible[(visible["driver"] == driver) & is_clean_lap(visible)].sort_values("lap_number")

        if len(d_clean) < SEQ_LEN:
            # Fallback to recent median pace if insufficient history
            return float(d_clean["lap_time"].median()) if not d_clean.empty else 90.0

        recent = d_clean.tail(SEQ_LEN)
        ref_pace = float(recent["lap_time"].median())

        total_laps = float(visible["total_laps"].iloc[0]) if "total_laps" in visible else 57.0
        lap_medians = visible[is_clean_lap(visible)].groupby("lap_number")["lap_time"].median().to_dict()

        seq = []
        for _, row in recent.iterrows():
            lap_n = row["lap_number"]
            l_med = lap_medians.get(lap_n, row["lap_time"])
            vs_field = row["lap_time"] - l_med
            age_norm = float(row.get("tyre_age_at_lap", 1)) / 50.0
            lap_frac = float(lap_n) / total_laps

            c_idx = COMPOUND_TO_IDX.get(str(row.get("compound", "MEDIUM")).upper(), 1)
            c_one_hot = [1.0 if j == c_idx else 0.0 for j in range(len(COMPOUNDS))]

            seq.append([vs_field, age_norm, lap_frac, *c_one_hot])

        delta = self.predict_sequence(np.array(seq, dtype=np.float32))
        return ref_pace + delta

    def save(self, directory: Path = MODEL_DIR, metrics: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), directory / "network.pt")
        meta = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_len": SEQ_LEN,
            "input_dim": INPUT_DIM,
            "sc_ratio": self.sc_ratio,
            "metrics": metrics or {},
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> "LapTimeGRUModel":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No GRU model metadata at {directory}")
        meta = json.loads(meta_path.read_text())
        model = cls(
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
            sc_ratio=meta.get("sc_ratio", 1.5),
        )
        model.network.load_state_dict(torch.load(directory / "network.pt", weights_only=True))
        model.network.eval()
        return model
