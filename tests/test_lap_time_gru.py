"""Unit tests for the sequential GRU baseline (src/models/lap_time_gru.py)."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.lap_time_gru import (
    INPUT_DIM,
    SEQ_LEN,
    GRUNetwork,
    LapTimeGRUModel,
    extract_driver_sequences,
)
from tests.test_lap_time_model import with_race_state
from tests.test_tyre_model import synthetic_race


@pytest.fixture(scope="module")
def synthetic_laps():
    r1 = with_race_state(synthetic_race("r1", "CircA", wear_scale=1.0, seed=1, n_drivers=6, n_laps=25))
    r2 = with_race_state(synthetic_race("r2", "CircB", wear_scale=1.5, seed=2, n_drivers=6, n_laps=25))
    import pandas as pd
    return pd.concat([r1, r2], ignore_index=True)


def test_gru_network_forward():
    net = GRUNetwork(input_dim=INPUT_DIM, hidden_dim=32, num_layers=1)
    batch_size = 4
    x = torch.randn(batch_size, SEQ_LEN, INPUT_DIM)
    out = net(x)
    assert out.shape == (batch_size, 1)
    assert not torch.isnan(out).any()


def test_extract_driver_sequences(synthetic_laps):
    X, y, ref_paces, meta = extract_driver_sequences(synthetic_laps, seq_len=SEQ_LEN)
    assert len(X) > 0
    assert X.shape[1] == SEQ_LEN
    assert X.shape[2] == INPUT_DIM
    assert len(y) == len(X)
    assert len(ref_paces) == len(X)
    assert len(meta) == len(X)
    assert meta[0]["driver"].startswith("D")


def test_gru_fit_predict_save_load(synthetic_laps, tmp_path):
    model = LapTimeGRUModel(hidden_dim=16, num_layers=1, dropout=0.0)
    model.fit(synthetic_laps, epochs=2, batch_size=32)

    # Predict single sequence
    dummy_seq = np.zeros((SEQ_LEN, INPUT_DIM), dtype=np.float32)
    delta = model.predict_sequence(dummy_seq)
    assert isinstance(delta, float)
    assert not np.isnan(delta)

    # Predict driver next lap
    driver = "D00"
    pred_time = model.predict_driver_next_lap(synthetic_laps, driver=driver, current_lap=15)
    assert isinstance(pred_time, float)
    assert 70.0 < pred_time < 120.0

    # Save and load
    save_dir = tmp_path / "gru_test"
    model.save(save_dir)
    assert (save_dir / "network.pt").exists()
    assert (save_dir / "meta.json").exists()

    loaded = LapTimeGRUModel.load(save_dir)
    delta_loaded = loaded.predict_sequence(dummy_seq)
    assert delta == pytest.approx(delta_loaded, rel=1e-5)
