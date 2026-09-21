"""Interaction-Aware Graph Neural Network (GNN) for Lap-Time Modeling.

Implements PRD.md FR-3 (Track b) and Design.md Sections 5 & 6.5.

What it does:
Treats each race timestep as a RaceGraph:
- Nodes: Drivers with tyre state, relative pace, track position, gap to leader.
- Edges: Physical relational edges between cars within interaction threshold (3.0s):
  gap, DRS activation window (<= 1.0s), exponential wake turbulence decay (<= 2.5s),
  and defensive pressure.
- Message Passing: 2 relational graph attention layers aggregating wake turbulence
  and DRS assistance directly into driver latent states.
- Readout: Jointly predicts next-lap times for all active drivers at once.

Interface (Design.md Section 5):
    def predict_lap_time_gnn(graph: RaceGraph) -> dict[str, float]
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tyre import REPO_ROOT
from src.preprocessing.cleaning import is_clean_lap
from src.preprocessing.graph import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    RaceGraph,
    build_race_graph,
    build_race_graph_from_state,
    edge_to_features,
    node_to_features,
)

MODEL_DIR = REPO_ROOT / "data" / "models" / "lap_time_gnn"
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EDGE_DIM = 32

log = logging.getLogger("overtake.models.lap_time_gnn")


# --- PyTorch Relational Graph Attention Layer -------------------------------

class RelationalGraphAttentionLayer(nn.Module):
    """Message passing layer with edge conditioning and multi-head attention.

    Computes:
      message m_ij = W_v [h_src, h_e]
      attention a_ij = LeakyReLU(w_att^T [h_src, h_dst, h_e])
      alpha_ij = softmax_j(a_ij)
      M_i = sum_j alpha_ij * m_ij
      h_i' = LayerNorm(h_i + MLP([h_i, M_i]))
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        # Projections
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, out_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim, out_dim),
        )
        self.att_linear = nn.Linear(node_dim * 2 + edge_dim, 1)

        # Update MLP + Normalization
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_feats: torch.Tensor,       # (N, node_dim)
        edge_indices: torch.Tensor,     # (2, M) [src, dst]
        edge_feats: torch.Tensor,       # (M, edge_dim)
    ) -> torch.Tensor:
        N = node_feats.size(0)
        M = edge_indices.size(1) if edge_indices.numel() > 0 else 0

        if M == 0:
            # If no interaction edges, pass through node projection
            residual = node_feats if self.node_dim == self.out_dim else F.pad(node_feats, (0, self.out_dim - self.node_dim))
            return self.norm(residual)

        src_idx, dst_idx = edge_indices[0], edge_indices[1]

        # 1. Compute messages on edges
        h_src = node_feats[src_idx]
        h_dst = node_feats[dst_idx]
        edge_inputs = torch.cat([h_src, edge_feats], dim=-1)
        messages = self.msg_mlp(edge_inputs)  # (M, out_dim)

        # 2. Compute attention coefficients
        att_inputs = torch.cat([h_src, h_dst, edge_feats], dim=-1)
        raw_att = F.leaky_relu(self.att_linear(att_inputs), 0.2).squeeze(-1)  # (M,)

        # 3. Target-wise softmax using scatter/subtraction for numerical stability
        # Group attention by destination node
        exp_att = torch.exp(raw_att - raw_att.max())
        denom = torch.zeros(N, device=node_feats.device, dtype=node_feats.dtype)
        denom.scatter_add_(0, dst_idx, exp_att)
        denom = denom.clamp(min=1e-8)
        alpha = exp_att / denom[dst_idx]  # (M,)

        # 4. Aggregate messages into destination nodes
        weighted_msgs = messages * alpha.unsqueeze(-1)  # (M, out_dim)
        aggregated = torch.zeros(N, self.out_dim, device=node_feats.device, dtype=node_feats.dtype)
        for d in range(self.out_dim):
            aggregated[:, d].scatter_add_(0, dst_idx, weighted_msgs[:, d])

        # 5. Node state update
        node_context = torch.cat([node_feats, aggregated], dim=-1)
        updated = self.update_mlp(node_context)
        out = self.norm(node_feats + updated)
        return out


class LapTimeGNN(nn.Module):
    """Full Graph Neural Network for interaction-aware lap time prediction."""

    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        edge_in_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        edge_dim: int = DEFAULT_EDGE_DIM,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            RelationalGraphAttentionLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns delta vs reference pace for each node: (N, 1)."""
        h_nodes = self.node_encoder(node_feats)
        if edge_feats.numel() > 0:
            h_edges = self.edge_encoder(edge_feats)
        else:
            h_edges = torch.zeros((0, self.edge_encoder[0].out_features), device=node_feats.device)

        for layer in self.layers:
            h_nodes = layer(h_nodes, edge_indices, h_edges)

        delta = self.head(h_nodes)
        return delta


# --- Graph Batching & Processing Helpers ------------------------------------

def graph_to_tensors(
    graph: RaceGraph,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], dict[str, float]]:
    """Convert RaceGraph into PyTorch tensors for model inference.

    Returns:
      node_tensors: (N, node_in_dim)
      edge_index: (2, M)
      edge_tensors: (M, edge_in_dim)
      drivers: list of driver codes in node index order
      ref_paces: dict of driver -> reference pace
    """
    drivers = graph.drivers
    driver_to_idx = {d: i for i, d in enumerate(drivers)}
    ref_paces = {d: float(graph.nodes[d].get("pace", 90.0)) for d in drivers}

    # 1. Node features
    node_list = [node_to_features(graph.nodes[d]) for d in drivers]
    node_tensor = torch.tensor(node_list, dtype=torch.float32, device=device)

    # 2. Edges
    src_list, dst_list, edge_list = [], [], []
    for src_d, dst_d, attr in graph.edges:
        if src_d in driver_to_idx and dst_d in driver_to_idx:
            src_list.append(driver_to_idx[src_d])
            dst_list.append(driver_to_idx[dst_d])
            edge_list.append(edge_to_features(attr))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        edge_tensor = torch.tensor(edge_list, dtype=torch.float32, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_tensor = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)

    return node_tensor, edge_index, edge_tensor, drivers, ref_paces


# --- Model High-Level Wrapper ----------------------------------------------

class LapTimeGNNModel:
    """Wrapper managing training, evaluation, saving, and inference for the GNN model."""

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        edge_dim: int = DEFAULT_EDGE_DIM,
        num_layers: int = 2,
        lr: float = 0.001,
        sc_ratio: float = 1.5,
    ):
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.lr = lr
        self.sc_ratio = sc_ratio
        self.network = LapTimeGNN(
            node_in_dim=NODE_FEATURE_DIM,
            edge_in_dim=EDGE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
        )

    def fit(
        self,
        race_laps: pd.DataFrame,
        epochs: int = 20,
        verbose: bool = False,
    ) -> "LapTimeGNNModel":
        """Train the GNN across timesteps of historical race laps."""
        clean = race_laps[is_clean_lap(race_laps)].copy()
        if clean.empty:
            log.warning("No clean laps available for GNN training.")
            return self

        # Extract graph samples at each lap
        samples = []
        for race_id, r_df in clean.groupby("race_id"):
            max_lap = int(r_df["lap_number"].max())
            for lap in range(2, max_lap):
                # Target lap is lap + 1
                target_laps = r_df[r_df["lap_number"] == lap + 1]
                if target_laps.empty:
                    continue

                graph = build_race_graph(r_df, current_lap=lap)
                if graph.n_nodes < 2:
                    continue

                # Collect target deltas
                targets = {}
                for _, row in target_laps.iterrows():
                    d = str(row["driver"])
                    if d in graph.nodes:
                        ref_p = graph.nodes[d]["pace"]
                        targets[d] = float(row["lap_time"] - ref_p)

                if targets:
                    samples.append((graph, targets))

        if not samples:
            log.warning("No valid (graph, target) pairs constructed for GNN.")
            return self

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        criterion = nn.SmoothL1Loss()

        self.network.train()
        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(samples)

            for graph, targets in samples:
                nodes_t, edges_idx, edges_t, drivers, _ = graph_to_tensors(graph)
                y_list = [targets[d] for d in drivers if d in targets]
                valid_indices = [i for i, d in enumerate(drivers) if d in targets]

                if not valid_indices:
                    continue

                optimizer.zero_grad()
                pred_deltas = self.network(nodes_t, edges_idx, edges_t)
                pred_sub = pred_deltas[valid_indices].squeeze(-1)
                y_sub = torch.tensor(y_list, dtype=torch.float32)

                loss = criterion(pred_sub, y_sub)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            if verbose and (epoch + 1) % 5 == 0:
                log.info("GNN Epoch %d/%d - Average Lap Loss: %.4f", epoch + 1, epochs, total_loss / len(samples))

        return self

    def predict_graph(self, graph: RaceGraph) -> dict[str, float]:
        """Predict next-lap time (in seconds) for every driver in graph."""
        if graph.n_nodes == 0:
            return {}

        self.network.eval()
        with torch.no_grad():
            nodes_t, edges_idx, edges_t, drivers, ref_paces = graph_to_tensors(graph)
            pred_deltas = self.network(nodes_t, edges_idx, edges_t).squeeze(-1).tolist()
            if isinstance(pred_deltas, float):
                pred_deltas = [pred_deltas]

        # If race state is under safety car, scale lap time
        is_sc = any(graph.nodes[d].get("safety_car", False) for d in drivers)

        predictions: dict[str, float] = {}
        for d, delta in zip(drivers, pred_deltas):
            base_pace = ref_paces[d] + float(delta)
            if is_sc:
                base_pace *= self.sc_ratio
            predictions[d] = round(base_pace, 3)

        return predictions

    def predict_state(self, state: Any, history: dict | None = None) -> dict[str, float]:
        """Convenience method accepting a RaceState dataclass."""
        graph = build_race_graph_from_state(state, history=history)
        return self.predict_graph(graph)

    def save(self, directory: Path = MODEL_DIR, metrics: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), directory / "network.pt")
        meta = {
            "hidden_dim": self.hidden_dim,
            "edge_dim": self.edge_dim,
            "num_layers": self.num_layers,
            "sc_ratio": self.sc_ratio,
            "node_in_dim": NODE_FEATURE_DIM,
            "edge_in_dim": EDGE_FEATURE_DIM,
            "metrics": metrics or {},
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> "LapTimeGNNModel":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            # Return fresh initialized model if no checkpoint saved yet
            return cls()
        meta = json.loads(meta_path.read_text())
        model = cls(
            hidden_dim=meta["hidden_dim"],
            edge_dim=meta["edge_dim"],
            num_layers=meta["num_layers"],
            sc_ratio=meta.get("sc_ratio", 1.5),
        )
        model.network.load_state_dict(torch.load(directory / "network.pt", weights_only=True))
        model.network.eval()
        return model


# --- Interface Contracts (Design.md Section 5) -----------------------------

@lru_cache(maxsize=1)
def _default_gnn_model() -> LapTimeGNNModel:
    return LapTimeGNNModel.load(MODEL_DIR)


def predict_lap_time_gnn(graph: RaceGraph) -> dict[str, float]:
    """Interaction-aware prediction for all drivers at once (Design.md Section 5)."""
    return _default_gnn_model().predict_graph(graph)
