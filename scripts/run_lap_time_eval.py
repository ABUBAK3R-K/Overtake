"""Score the frozen lap-time test set across all four competing tracks (PRD FR-3).

Compares:
  1. Persistence baseline (repeat reference pace)
  2. Independent XGBoost baseline (src.models.lap_time)
  3. Independent sequential GRU baseline (src.models.lap_time_gru)
  4. Interaction-aware Graph Neural Network (src.models.lap_time_gnn)

Writes data/models/lap_time/test_report.json and prints comparative summary tables.

Usage:
    python scripts/run_lap_time_eval.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.lap_time_eval import breakdown, load_split, split_laps  # noqa: E402
from src.models.lap_time import (  # noqa: E402
    MODEL_DIR,
    LapTimeModel,
    assemble_features,
    build_training_rows,
    history_features,
    safety_car_ratio,
)
from src.models.lap_time_gnn import LapTimeGNNModel  # noqa: E402
from src.models.lap_time_gru import LapTimeGRUModel  # noqa: E402
from src.models.tyre import load_lap_frame  # noqa: E402
from src.preprocessing.graph import build_race_graph  # noqa: E402

log = logging.getLogger("overtake.evaluation.lap_time")


def evaluate_all(
    laps: pd.DataFrame,
    split: dict,
    tags: dict | None = None,
) -> dict[str, Any]:
    """Train candidates on the train split and evaluate on the frozen test split."""
    train_laps, test_laps = split_laps(laps, split)
    print(f"Train set: {len(train_laps)} laps across {train_laps.race_id.nunique()} races")
    print(f"Test set:  {len(test_laps)} laps across {test_laps.race_id.nunique()} races")

    sc_ratio = safety_car_ratio(train_laps) if not train_laps.empty else 1.5

    # 1. Fit Baselines
    print("\n[1/3] Fitting XGBoost baseline...")
    train_rows = build_training_rows(train_laps)
    xgb_model = LapTimeModel(sc_ratio=sc_ratio).fit(train_rows)

    print("[2/3] Fitting Sequential GRU baseline...")
    gru_model = LapTimeGRUModel(hidden_dim=32, num_layers=2, sc_ratio=sc_ratio)
    gru_model.fit(train_laps, epochs=15, verbose=False)

    print("[3/3] Fitting Interaction-Aware GNN model...")
    gnn_model = LapTimeGNNModel(hidden_dim=48, edge_dim=24, num_layers=2, sc_ratio=sc_ratio)
    gnn_model.fit(train_laps, epochs=15, verbose=False)

    # 2. Generate Predictions on Test Laps
    print("\nGenerating predictions on held-out test races...")
    test_rows = build_training_rows(test_laps, horizons=(1,))
    if test_rows.empty:
        print("Warning: No test rows generated. Exiting evaluation.")
        return {}

    # XGBoost Predictions
    xgb_preds = xgb_model.predict_rows(test_rows)
    test_rows["pred_xgb"] = xgb_preds
    test_rows["pred_persistence"] = test_rows["ref_pace"]

    # GRU Predictions
    gru_preds = []
    for _, row in test_rows.iterrows():
        r_id = row["race_id"]
        d = row["driver"]
        anchor = int(row["anchor_lap"])
        r_laps = test_laps[test_laps["race_id"] == r_id]
        p = gru_model.predict_driver_next_lap(r_laps, driver=d, current_lap=anchor)
        gru_preds.append(p)
    test_rows["pred_gru"] = gru_preds

    # GNN Predictions
    gnn_preds = []
    # Cache graphs per (race_id, anchor_lap)
    graph_cache = {}
    for _, row in test_rows.iterrows():
        r_id = row["race_id"]
        d = row["driver"]
        anchor = int(row["anchor_lap"])
        key = (r_id, anchor)
        if key not in graph_cache:
            r_laps = test_laps[test_laps["race_id"] == r_id]
            graph = build_race_graph(r_laps, current_lap=anchor)
            graph_cache[key] = gnn_model.predict_graph(graph)
        pred_dict = graph_cache[key]
        gnn_preds.append(pred_dict.get(d, row["ref_pace"]))
    test_rows["pred_gnn"] = gnn_preds

    # 3. Evaluate Breakdowns
    models = {
        "persistence": test_rows["pred_persistence"],
        "xgboost": test_rows["pred_xgb"],
        "gru": test_rows["pred_gru"],
        "gnn": test_rows["pred_gnn"],
    }

    report = {}
    for name, p_col in models.items():
        df = test_rows.copy()
        df["pred"] = p_col
        report[name] = breakdown(df, split, tags, n_boot=500)

    # Save models
    xgb_model.save(MODEL_DIR / "xgboost", metrics=report["xgboost"])
    gru_model.save(MODEL_DIR / "gru", metrics=report["gru"])
    gnn_model.save(MODEL_DIR / "gnn", metrics=report["gnn"])

    (MODEL_DIR / "test_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def print_summary_table(report: dict[str, Any]) -> None:
    slices = [
        ("all", "All test laps"),
        ("clean_air", "Clean air (>3s)"),
        ("in_traffic_wake", "In wake (<=2s)"),
        ("drs_range", "In DRS window (<=1s)"),
        ("unseen_circuit", "Unseen circuit"),
        ("seen_circuit", "Seen circuit"),
    ]

    print("\n" + "=" * 80)
    print(f"{'Model':16s} | {'Slice':22s} | {'Lap MAE (s)':>11s} | {'Lap RMSE (s)':>12s} | {'95% CI':>16s}")
    print("-" * 80)

    for m_name in ("persistence", "xgboost", "gru", "gnn"):
        m_rep = report.get(m_name, {})
        for s_key, s_label in slices:
            if s_key in m_rep:
                d = m_rep[s_key]
                mae = d.get("lap_mae", np.nan)
                rmse = d.get("lap_rmse", np.nan)
                ci = d.get("lap_mae_ci", (np.nan, np.nan))
                ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if not np.isnan(ci[0]) else "N/A"
                print(f"{m_name:16s} | {s_label:22s} | {mae:11.3f} | {rmse:12.3f} | {ci_str:>16s}")
        print("-" * 80)
    print("=" * 80 + "\n")


def main() -> int:
    split = load_split()
    tags = None
    tags_path = ROOT / "configs" / "race_tags.json"
    if tags_path.exists():
        tags = json.loads(tags_path.read_text(encoding="utf-8"))

    laps = load_lap_frame()
    if laps.empty:
        print("No laps found in data/processed/laps. Run ingestion or test on synthetic data.")
        return 1

    report = evaluate_all(laps, split, tags)
    if report:
        print_summary_table(report)
        print(f"Results written to {MODEL_DIR / 'test_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
