"""Score the frozen tyre test set ONCE and write data/models/tyre/test_report.json.

Run this after model selection (inner CV in src.evaluation.tyre_eval) is done.
Every candidate is scored in the same pass; nothing here should be used to
tune a model. Usage:  python scripts/run_tyre_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.tyre_eval import breakdown, load_split, split_frame  # noqa: E402
from src.models.tyre import (MODEL_DIR, TyreModel, build_training_frame,  # noqa: E402
                             load_config, load_lap_frame)
from src.models.tyre_baselines import MeanCurveModel, ZeroModel  # noqa: E402
from src.models.tyre_bayes import BayesTyreModel  # noqa: E402


def main() -> int:
    cfg, split = load_config(), load_split()
    tags = json.loads((ROOT / "configs" / "race_tags.json").read_text(encoding="utf-8"))
    frame = build_training_frame(load_lap_frame(), cfg)
    frame["train_weight"] = frame["race_id"].map(lambda r: tags[r]["train_weight"])
    train, test = split_frame(frame, split)
    print(f"train {len(train)} laps / {train.race_id.nunique()} races; "
          f"test {len(test)} laps / {test.race_id.nunique()} races")

    xgb = lambda **kw: TyreModel("pooled", cfg["xgboost"], cfg["max_age"], **kw)  # noqa: E731
    candidates = {
        "zero": ZeroModel(), "mean_curve": MeanCurveModel(), "bayes": BayesTyreModel(),
        "xgb_current": xgb(), "xgb_no_temp": xgb(use_temp=False),
        "xgb_no_circuit_no_temp": xgb(use_circuit=False, use_temp=False),
    }
    report, fitted = {}, {}
    for name, model in candidates.items():
        fitted[name] = model.fit(train)
        preds = test.assign(pred=model.predict_frame(test))
        report[name] = breakdown(preds, split, tags)

    # Bayes: does the 90% interval on a (race, compound, age) mean cover the
    # observed mean? Split by unseen vs seen circuit.
    bayes = fitted["bayes"]
    lo, hi = bayes.predict_interval(test, level=0.9)
    cells = (test.assign(lo=lo, hi=hi, p=bayes.predict_frame(test))
             .groupby(["race_id", "circuit", "compound", "tyre_age_at_lap"])
             .agg(t=("pace_loss", "mean"), n=("pace_loss", "size"), lo=("lo", "mean"), hi=("hi", "mean")))
    cells = cells[cells["n"] >= 5].reset_index()
    # Widen by the lap noise averaged over n laps so it is a fair target for a cell mean.
    half_noise = 1.645 * bayes.sigma / np.sqrt(cells["n"])
    inside = (cells["t"] >= cells["lo"] - half_noise) & (cells["t"] <= cells["hi"] + half_noise)
    unseen = cells["circuit"].isin(split["held_out_circuits"])
    report["bayes_interval_coverage_90"] = {
        "all": float(inside.mean()), "unseen_circuit": float(inside[unseen].mean()),
        "seen_circuit": float(inside[~unseen].mean()), "n_cells": int(len(cells)),
        "mean_half_width_s": float(((cells["hi"] - cells["lo"]) / 2).mean())}

    # SHAP importance (mean |contribution|, seconds) for the XGBoost variants.
    seen_rows = test[~test["circuit"].isin(split["held_out_circuits"])]
    report["shap_mean_abs"] = {
        name: fitted[name].shap_values(seen_rows).drop(columns="bias").abs().mean().round(4).to_dict()
        for name in ("xgb_current", "xgb_no_temp", "xgb_no_circuit_no_temp")}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "test_report.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"\n{'model':26s}{'slice':16s}{'curve MAE':>10s}{'95% CI':>16s}{'lap MAE':>9s}{'lap RMSE':>9s}")
    for name in candidates:
        for sl in ("all", "unseen_circuit", "seen_circuit", "wet_races", "dry_races"):
            r = report[name].get(sl)
            if r:
                ci = r.get("curve_mae_ci", (np.nan, np.nan))
                print(f"{name:26s}{sl:16s}{r['curve_mae']:10.3f}{f'[{ci[0]:.2f},{ci[1]:.2f}]':>16s}"
                      f"{r['lap_mae']:9.3f}{r['lap_rmse']:9.3f}")
    print("\nBayes 90% interval coverage:", report["bayes_interval_coverage_90"])
    print("SHAP mean |contribution| (s):", json.dumps(report["shap_mean_abs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
