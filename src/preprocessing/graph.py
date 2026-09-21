"""RaceGraph data structures and graph construction for interaction-aware modeling.

Follows Design.md Section 5:
    @dataclass
    class RaceGraph:
        lap: int
        nodes: dict[str, dict]                    # driver -> {position, tyre, pace, gap}
        edges: list[tuple[str, str, dict]]         # (driver_a, driver_b, {gap, drs_range, wake_effect})

Physics & Interaction Rules:
- Cars within MAX_INTERACTION_GAP (3.0s) form interaction edges.
- DRS range: trailing car is within 1.0s of car ahead (gain top-speed / DRS boost).
- Wake effect: exponential turbulence decay exp(-gap / WAKE_DECAY_SCALE) when gap <= 2.5s.
  Reflects downforce loss, tyre overheating, and cornering lap-time penalty in dirty air.
- Bidirectional awareness: trailing car suffers wake turbulence; leading car faces defensive pressure.
- No future data leakage: strictly routes through as_of_lap().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import clean_gap_to_leader, is_clean_lap
from src.preprocessing.leakage import as_of_lap

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}

# Physical thresholds in seconds
DRS_GAP_THRESHOLD = 1.0        # FIA DRS detection window (s)
WAKE_GAP_THRESHOLD = 2.5       # Dirty air downforce loss threshold (s)
WAKE_DECAY_SCALE = 1.5         # Characteristic decay scale for turbulence
MAX_INTERACTION_GAP = 3.0      # Spatial cutoff for graph edges (s)
REFERENCE_LAPS = 5


@dataclass
class RaceGraph:
    """Graph representation of the field at a single race lap."""
    lap: int
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    @property
    def drivers(self) -> list[str]:
        """List of drivers in position order."""
        return sorted(self.nodes.keys(), key=lambda d: self.nodes[d].get("position", 999))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def compute_wake_effect(gap: float) -> float:
    """Aerodynamic wake turbulence intensity factor in [0, 1].

    Maximum at gap=0, decaying exponentially to 0 beyond WAKE_GAP_THRESHOLD.
    """
    if gap < 0 or gap > WAKE_GAP_THRESHOLD:
        return 0.0
    return float(np.exp(-gap / WAKE_DECAY_SCALE))


def build_race_graph(
    race_laps: pd.DataFrame,
    current_lap: int,
    total_laps: int | None = None,
) -> RaceGraph:
    """Build a RaceGraph as of current_lap strictly from historical laps.

    Leakage-safe: only rows with lap_number <= current_lap are visible.
    State at current_lap defines positions, tyres, and gaps; past clean laps
    define each driver's reference pace.
    """
    visible = as_of_lap(race_laps, "lap_number", current_lap)
    if visible.empty:
        return RaceGraph(lap=current_lap)

    if total_laps is None:
        total_laps = int(visible["total_laps"].iloc[0]) if "total_laps" in visible else int(visible["lap_number"].max())

    # 1. State on current_lap (track position, tyres, gaps)
    cur_lap_df = visible[visible["lap_number"] == current_lap].copy()
    if cur_lap_df.empty:
        # If current_lap has no rows (e.g. lap 0), use the latest available lap
        latest_lap = int(visible["lap_number"].max())
        cur_lap_df = visible[visible["lap_number"] == latest_lap].copy()

    cur_lap_df["gap_clean"] = clean_gap_to_leader(cur_lap_df)

    # 2. Historical clean-lap reference pace
    clean = visible[is_clean_lap(visible)].sort_values("lap_number")
    if not clean.empty:
        lap_median = clean.groupby("lap_number")["lap_time"].median()
        clean = clean.assign(vs_field=clean["lap_time"] - clean["lap_number"].map(lap_median))
        recent = clean.groupby("driver").tail(REFERENCE_LAPS).groupby("driver")
        ref_pace_map = recent["lap_time"].median().to_dict()
        ref_lap_map = recent["lap_number"].mean().to_dict()
        driver_vs_field_map = recent["vs_field"].median().to_dict()
        ref_compound_map = recent["compound"].last().to_dict()
        field_pace = float(lap_median.iloc[-1]) if len(lap_median) else 90.0
    else:
        ref_pace_map, ref_lap_map, driver_vs_field_map, ref_compound_map = {}, {}, {}, {}
        field_pace = 90.0

    # 3. Assemble nodes
    nodes: dict[str, dict[str, Any]] = {}
    cur_lap_df = cur_lap_df.sort_values("position" if "position" in cur_lap_df else "lap_number")

    for _, row in cur_lap_df.iterrows():
        driver = str(row["driver"])
        pos = int(row.get("position", len(nodes) + 1))
        comp = str(row.get("compound", "MEDIUM")).upper()
        age = int(row.get("tyre_age_at_lap", 1))
        gap = float(row.get("gap_clean", row.get("gap_to_leader", 0.0)))
        if np.isnan(gap):
            gap = 0.0

        ref_p = ref_pace_map.get(driver, field_pace)
        dvf = driver_vs_field_map.get(driver, 0.0)
        ref_l = ref_lap_map.get(driver, float(current_lap))
        ref_c = ref_compound_map.get(driver, comp)

        nodes[driver] = {
            "position": pos,
            "compound": comp,
            "tyre_age": age,
            "tyre": (comp, age),
            "pace": float(ref_p),
            "gap": float(gap),
            "driver_vs_field": float(dvf),
            "laps_since_ref": float(current_lap - ref_l),
            "ref_compound": ref_c,
            "lap_fraction": float(current_lap / total_laps) if total_laps else 0.5,
        }

    # 4. Assemble interaction edges
    edges = _build_interaction_edges(nodes)
    return RaceGraph(lap=current_lap, nodes=nodes, edges=edges)


def build_race_graph_from_state(state: Any, history: dict | None = None) -> RaceGraph:
    """Build a RaceGraph directly from a simulation or replay RaceState.

    `state` has: lap, positions (dict), gaps (dict), tyres (dict), safety_car (bool).
    `history` is optional dict containing driver reference pace and baseline metrics.
    """
    nodes: dict[str, dict[str, Any]] = {}
    total_laps = getattr(state, "total_laps", 57)
    lap_fraction = float(state.lap / total_laps) if total_laps else 0.5

    for driver, pos in state.positions.items():
        comp, age = state.tyres.get(driver, ("MEDIUM", 1))
        comp_str = str(comp).upper()
        gap = state.gaps.get(driver, 0.0)
        gap_val = float(gap) if gap is not None and not np.isnan(gap) else 0.0

        hist_driver = history.get("drivers", {}).get(driver, {}) if history else {}
        ref_pace = hist_driver.get("ref_pace", getattr(state, "last_lap_times", {}).get(driver, 90.0))
        dvf = hist_driver.get("driver_vs_field", 0.0)
        ref_c = hist_driver.get("ref_compound", comp_str)

        nodes[driver] = {
            "position": int(pos),
            "compound": comp_str,
            "tyre_age": int(age),
            "tyre": (comp_str, int(age)),
            "pace": float(ref_pace) if ref_pace is not None else 90.0,
            "gap": gap_val,
            "driver_vs_field": float(dvf) if dvf is not None else 0.0,
            "laps_since_ref": 1.0,
            "ref_compound": ref_c,
            "lap_fraction": lap_fraction,
            "safety_car": bool(getattr(state, "safety_car", False)),
        }

    edges = _build_interaction_edges(nodes)
    return RaceGraph(lap=state.lap, nodes=nodes, edges=edges)


def _build_interaction_edges(nodes: dict[str, dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    """Form relational edges between cars within MAX_INTERACTION_GAP on track.

    For drivers A and B sorted by track position (A ahead of B):
    gap_ab = gap_to_leader(B) - gap_to_leader(A) >= 0
    If gap_ab <= MAX_INTERACTION_GAP:
      - (A, B): forward wake edge (A casts dirty air on follower B)
      - (B, A): reverse pressure edge (follower B pressures leader A)
    """
    edges: list[tuple[str, str, dict[str, Any]]] = []
    drivers_by_pos = sorted(nodes.keys(), key=lambda d: nodes[d].get("position", 999))

    for i in range(len(drivers_by_pos)):
        d_lead = drivers_by_pos[i]
        gap_lead = nodes[d_lead].get("gap", 0.0)
        pos_lead = nodes[d_lead].get("position", i + 1)

        for j in range(i + 1, len(drivers_by_pos)):
            d_trail = drivers_by_pos[j]
            gap_trail = nodes[d_trail].get("gap", 0.0)
            pos_trail = nodes[d_trail].get("position", j + 1)

            delta_gap = gap_trail - gap_lead
            if delta_gap < 0:
                delta_gap = 0.0

            if delta_gap > MAX_INTERACTION_GAP:
                if j == i + 1 and delta_gap > MAX_INTERACTION_GAP * 1.5:
                    break
                continue

            in_drs = delta_gap <= DRS_GAP_THRESHOLD
            wake = compute_wake_effect(delta_gap)

            # 1. Leader -> Follower edge (Wake / Dirty Air / DRS opportunity)
            edges.append((
                d_lead,
                d_trail,
                {
                    "gap": float(delta_gap),
                    "drs_range": bool(in_drs),
                    "wake_effect": float(wake),
                    "relative_pos": int(pos_trail - pos_lead),
                    "direction": "ahead_to_follower",
                },
            ))

            # 2. Follower -> Leader edge (Rear Pressure / Mirror Watching)
            edges.append((
                d_trail,
                d_lead,
                {
                    "gap": float(delta_gap),
                    "drs_range": bool(in_drs),
                    "wake_effect": 0.0,
                    "relative_pos": int(pos_lead - pos_trail),
                    "direction": "follower_to_ahead",
                },
            ))

    return edges


# --- Tensor Conversion Utilities for PyTorch --------------------------------

def node_to_features(node: dict[str, Any]) -> list[float]:
    """Convert node attributes to a standardized continuous feature vector."""
    comp_idx = COMPOUND_TO_IDX.get(node.get("compound", "MEDIUM"), 1)
    comp_one_hot = [1.0 if i == comp_idx else 0.0 for i in range(len(COMPOUNDS))]
    ref_comp_idx = COMPOUND_TO_IDX.get(node.get("ref_compound", "MEDIUM"), 1)
    same_comp = 1.0 if comp_idx == ref_comp_idx else 0.0

    return [
        float(node.get("position", 10)) / 20.0,
        float(node.get("tyre_age", 1)) / 50.0,
        float(node.get("gap", 0.0)) / 100.0,
        float(node.get("driver_vs_field", 0.0)),
        float(node.get("laps_since_ref", 0.0)) / 20.0,
        float(node.get("lap_fraction", 0.5)),
        same_comp,
        *comp_one_hot,
    ]


def edge_to_features(edge_attr: dict[str, Any]) -> list[float]:
    """Convert edge attributes to a continuous feature vector."""
    gap = float(edge_attr.get("gap", 0.0))
    drs = 1.0 if edge_attr.get("drs_range", False) else 0.0
    wake = float(edge_attr.get("wake_effect", 0.0))
    rel_pos = float(edge_attr.get("relative_pos", 0)) / 20.0
    is_forward = 1.0 if edge_attr.get("direction") == "ahead_to_follower" else 0.0

    return [
        min(gap, MAX_INTERACTION_GAP) / MAX_INTERACTION_GAP,
        drs,
        wake,
        rel_pos,
        is_forward,
    ]


NODE_FEATURE_DIM = len(node_to_features({}))
EDGE_FEATURE_DIM = len(edge_to_features({}))
