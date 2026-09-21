"""Tests for RaceGraph data structures, interaction edge physics, and leakage safety."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.graph import (
    DRS_GAP_THRESHOLD,
    MAX_INTERACTION_GAP,
    WAKE_GAP_THRESHOLD,
    RaceGraph,
    build_race_graph,
    build_race_graph_from_state,
    compute_wake_effect,
    edge_to_features,
    node_to_features,
)
from tests.test_lap_time_model import with_race_state
from tests.test_tyre_model import synthetic_race


class MockState:
    def __init__(self, lap, positions, gaps, tyres, safety_car=False, total_laps=50):
        self.lap = lap
        self.positions = positions
        self.gaps = gaps
        self.tyres = tyres
        self.safety_car = safety_car
        self.total_laps = total_laps
        self.last_lap_times = {d: 92.0 for d in positions}


def test_wake_effect_physics():
    assert compute_wake_effect(0.0) == pytest.approx(1.0)
    assert 0.0 < compute_wake_effect(1.0) < 1.0
    assert 0.0 < compute_wake_effect(2.0) < compute_wake_effect(1.0)
    assert compute_wake_effect(WAKE_GAP_THRESHOLD + 0.1) == 0.0
    assert compute_wake_effect(-0.5) == 0.0


def test_build_race_graph_from_mock_state():
    positions = {"VER": 1, "PER": 2, "HAM": 3, "ALO": 4}
    gaps = {"VER": 0.0, "PER": 0.8, "HAM": 2.2, "ALO": 10.5}
    tyres = {"VER": ("MEDIUM", 10), "PER": ("MEDIUM", 10), "HAM": ("HARD", 5), "ALO": ("HARD", 12)}
    state = MockState(lap=15, positions=positions, gaps=gaps, tyres=tyres)

    graph = build_race_graph_from_state(state)
    assert graph.lap == 15
    assert graph.n_nodes == 4
    assert graph.drivers == ["VER", "PER", "HAM", "ALO"]

    # PER is 0.8s behind VER -> within DRS and wake
    # HAM is 2.2s - 0.8s = 1.4s behind PER -> wake > 0, drs = False
    # ALO is 10.5s - 2.2s = 8.3s behind HAM -> > MAX_INTERACTION_GAP (no edge)
    ver_per_edges = [e for e in graph.edges if e[0] == "VER" and e[1] == "PER"]
    assert len(ver_per_edges) == 1
    edge_attr = ver_per_edges[0][2]
    assert edge_attr["gap"] == pytest.approx(0.8)
    assert edge_attr["drs_range"] is True
    assert edge_attr["wake_effect"] > 0

    # ALO is far behind -> no edges connected to ALO
    alo_edges = [e for e in graph.edges if e[0] == "ALO" or e[1] == "ALO"]
    assert len(alo_edges) == 0


def test_race_graph_no_leakage():
    # Build synthetic race
    race = with_race_state(synthetic_race("r_leak", "TestCircuit", wear_scale=1.0, seed=42, n_drivers=6, n_laps=30))

    # Graph at lap 12
    g1 = build_race_graph(race, current_lap=12)

    # Cut off future laps (> 12) completely
    race_truncated = race[race["lap_number"] <= 12].copy()
    g2 = build_race_graph(race_truncated, current_lap=12)

    assert g1.lap == g2.lap
    assert g1.drivers == g2.drivers
    assert g1.n_nodes == g2.n_nodes
    assert g1.n_edges == g2.n_edges

    for d in g1.drivers:
        assert g1.nodes[d]["position"] == g2.nodes[d]["position"]
        assert g1.nodes[d]["gap"] == pytest.approx(g2.nodes[d]["gap"])
        assert g1.nodes[d]["pace"] == pytest.approx(g2.nodes[d]["pace"])


def test_feature_conversion():
    node = {
        "position": 3,
        "compound": "SOFT",
        "tyre_age": 12,
        "gap": 4.5,
        "driver_vs_field": -0.2,
        "laps_since_ref": 2.0,
        "ref_compound": "SOFT",
        "lap_fraction": 0.3,
    }
    feats = node_to_features(node)
    assert len(feats) > 0
    assert all(isinstance(x, float) for x in feats)

    edge = {
        "gap": 0.7,
        "drs_range": True,
        "wake_effect": 0.62,
        "relative_pos": 1,
        "direction": "ahead_to_follower",
    }
    edge_feats = edge_to_features(edge)
    assert len(edge_feats) > 0
    assert all(isinstance(x, float) for x in edge_feats)
