"""Unit tests for the interaction-aware GNN model (src/models/lap_time_gnn.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lap_time_gnn import (
    LapTimeGNN,
    LapTimeGNNModel,
    RelationalGraphAttentionLayer,
    graph_to_tensors,
    predict_lap_time_gnn,
)
from src.preprocessing.graph import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    RaceGraph,
    build_race_graph,
    build_race_graph_from_state,
)
from tests.test_graph_construction import MockState
from tests.test_lap_time_model import with_race_state
from tests.test_tyre_model import synthetic_race


@pytest.fixture(scope="module")
def synthetic_laps():
    r1 = with_race_state(synthetic_race("r1", "CircA", wear_scale=1.0, seed=10, n_drivers=6, n_laps=25))
    r2 = with_race_state(synthetic_race("r2", "CircB", wear_scale=1.2, seed=11, n_drivers=6, n_laps=25))
    return pd.concat([r1, r2], ignore_index=True)


def test_relational_gat_layer_forward():
    layer = RelationalGraphAttentionLayer(node_dim=16, edge_dim=8, out_dim=16)
    nodes = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 8)

    out = layer(nodes, edge_index, edge_attr)
    assert out.shape == (4, 16)
    assert not torch.isnan(out).any()

    # Empty edges case
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
    empty_edge_attr = torch.zeros((0, 8))
    out_empty = layer(nodes, empty_edge_index, empty_edge_attr)
    assert out_empty.shape == (4, 16)
    assert not torch.isnan(out_empty).any()


def test_gnn_model_forward():
    model = LapTimeGNN(
        node_in_dim=NODE_FEATURE_DIM,
        edge_in_dim=EDGE_FEATURE_DIM,
        hidden_dim=32,
        edge_dim=16,
        num_layers=2,
    )
    nodes = torch.randn(5, NODE_FEATURE_DIM)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.randn(2, EDGE_FEATURE_DIM)

    delta = model(nodes, edge_index, edge_attr)
    assert delta.shape == (5, 1)
    assert not torch.isnan(delta).any()


def test_predict_graph_and_safety_car():
    state = MockState(
        lap=10,
        positions={"VER": 1, "PER": 2, "HAM": 3},
        gaps={"VER": 0.0, "PER": 0.7, "HAM": 1.5},
        tyres={"VER": ("MEDIUM", 8), "PER": ("MEDIUM", 8), "HAM": ("HARD", 4)},
        safety_car=False,
    )
    graph = build_race_graph_from_state(state)

    model = LapTimeGNNModel(hidden_dim=16, edge_dim=8, num_layers=1, sc_ratio=1.5)
    preds = model.predict_graph(graph)

    assert len(preds) == 3
    assert set(preds.keys()) == {"VER", "PER", "HAM"}
    assert all(isinstance(v, float) for v in preds.values())
    assert all(60.0 < v < 130.0 for v in preds.values())

    # Under safety car, pace should be scaled by sc_ratio
    state_sc = MockState(
        lap=10,
        positions={"VER": 1, "PER": 2, "HAM": 3},
        gaps={"VER": 0.0, "PER": 0.7, "HAM": 1.5},
        tyres={"VER": ("MEDIUM", 8), "PER": ("MEDIUM", 8), "HAM": ("HARD", 4)},
        safety_car=True,
    )
    graph_sc = build_race_graph_from_state(state_sc)
    preds_sc = model.predict_graph(graph_sc)

    for d in preds:
        assert preds_sc[d] > preds[d]
        assert preds_sc[d] == pytest.approx(preds[d] * 1.5, rel=0.05)


def test_contract_predict_lap_time_gnn():
    state = MockState(
        lap=12,
        positions={"LEC": 1, "SAI": 2},
        gaps={"LEC": 0.0, "SAI": 0.9},
        tyres={"LEC": ("SOFT", 5), "SAI": ("SOFT", 5)},
    )
    graph = build_race_graph_from_state(state)
    result = predict_lap_time_gnn(graph)
    assert isinstance(result, dict)
    assert "LEC" in result and "SAI" in result
    assert isinstance(result["LEC"], float)


def test_gnn_fit_save_load(synthetic_laps, tmp_path):
    model = LapTimeGNNModel(hidden_dim=16, edge_dim=8, num_layers=1)
    model.fit(synthetic_laps, epochs=2)

    save_dir = tmp_path / "gnn_test"
    model.save(save_dir)
    assert (save_dir / "network.pt").exists()
    assert (save_dir / "meta.json").exists()

    loaded = LapTimeGNNModel.load(save_dir)
    g = build_race_graph(synthetic_laps, current_lap=15)
    preds = loaded.predict_graph(g)
    assert len(preds) > 0
