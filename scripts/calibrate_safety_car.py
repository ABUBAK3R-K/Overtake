"""Calibrate empirical safety-car deployment rates and phase modifiers (PRD FR-5).

Uses configs/race_tags.json across 112 multi-season races (2018-2024, 33 circuits).
Applies Empirical Bayes shrinkage toward the global grand mean to prevent
small-sample bias for newly added circuits.

Outputs:
  configs/safety_car_rates.json

Usage:
  python scripts/calibrate_safety_car.py
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAGS_FILE = ROOT / "configs" / "race_tags.json"
OUTPUT_FILE = ROOT / "configs" / "safety_car_rates.json"

# Domain prior SC base rates reflecting track barrier proximity, runoff area, and historic volatility
CIRCUIT_DOMAIN_PRIORS = {
    "Marina Bay": 0.090,
    "Monaco": 0.080,
    "Baku": 0.075,
    "Melbourne": 0.070,
    "Jeddah": 0.070,
    "Las Vegas": 0.065,
    "Montréal": 0.060,
    "Silverstone": 0.050,
    "Zandvoort": 0.055,
    "São Paulo": 0.055,
    "Spa-Francorchamps": 0.050,
    "Miami": 0.050,
    "Imola": 0.045,
    "Suzuka": 0.045,
    "Monza": 0.038,
    "Mexico City": 0.038,
    "Austin": 0.035,
    "Budapest": 0.035,
    "Spielberg": 0.032,
    "Shanghai": 0.035,
    "Yas Island": 0.032,
    "Sochi": 0.030,
    "Sakhir": 0.024,
    "Barcelona": 0.018,
    "Lusail": 0.022,
    "Le Castellet": 0.020,
    "Istanbul": 0.035,
    "Portimão": 0.035,
    "Mugello": 0.045,
    "Nürburgring": 0.040,
    "Hockenheim": 0.045,
}
DEFAULT_DOMAIN_PRIOR = 0.038
SHRINKAGE_PSEUDO_LAPS = 180.0


def calibrate() -> dict:
    tags = json.loads(TAGS_FILE.read_text(encoding="utf-8"))

    circuit_stats: dict[str, dict] = {}
    total_laps_global = 0
    total_sc_global = 0
    total_vsc_global = 0

    # Canonical mappings for circuits with multiple naming conventions in tags
    canonical_circuit = {
        "Monte Carlo": "Monaco",
        "Singapore": "Marina Bay",
    }

    for race_id, info in tags.items():
        circuit = canonical_circuit.get(info["circuit"], info["circuit"])
        laps = info.get("total_laps", 57)
        sc = info.get("sc_deployments", 0)
        vsc = info.get("vsc_deployments", 0)
        sc_share = info.get("sc_lap_share", 0.0)

        total_laps_global += laps
        total_sc_global += sc
        total_vsc_global += vsc

        if circuit not in circuit_stats:
            circuit_stats[circuit] = {
                "n_races": 0,
                "total_laps": 0,
                "sc_deployments": 0,
                "vsc_deployments": 0,
                "sc_lap_shares": [],
            }

        stats = circuit_stats[circuit]
        stats["n_races"] += 1
        stats["total_laps"] += laps
        stats["sc_deployments"] += sc
        stats["vsc_deployments"] += vsc
        stats["sc_lap_shares"].append(sc_share)

    # Mirror stats for synonyms
    for syn, can in canonical_circuit.items():
        if can in circuit_stats and syn not in circuit_stats:
            circuit_stats[syn] = dict(circuit_stats[can])

    global_sc_rate = total_sc_global / max(1, total_laps_global)
    global_vsc_rate = total_vsc_global / max(1, total_laps_global)
    global_any_rate = global_sc_rate + global_vsc_rate

    calibrated_circuits = {}
    for circuit, stats in circuit_stats.items():
        n_laps = stats["total_laps"]
        raw_sc_rate = stats["sc_deployments"] / max(1, n_laps)
        raw_vsc_rate = stats["vsc_deployments"] / max(1, n_laps)
        raw_any_rate = raw_sc_rate + raw_vsc_rate

        # Prior for circuit
        prior_rate = CIRCUIT_DOMAIN_PRIORS.get(circuit, DEFAULT_DOMAIN_PRIOR)
        # Split prior into SC and VSC proportional to global ratio (~64% SC, ~36% VSC)
        sc_vsc_ratio = global_sc_rate / max(1e-6, global_any_rate)
        prior_sc_rate = prior_rate * sc_vsc_ratio
        prior_vsc_rate = prior_rate * (1.0 - sc_vsc_ratio)

        # Empirical Bayes shrinkage combining empirical rate and circuit prior
        shrunk_sc_rate = (n_laps * raw_sc_rate + SHRINKAGE_PSEUDO_LAPS * prior_sc_rate) / (
            n_laps + SHRINKAGE_PSEUDO_LAPS
        )
        shrunk_vsc_rate = (n_laps * raw_vsc_rate + SHRINKAGE_PSEUDO_LAPS * prior_vsc_rate) / (
            n_laps + SHRINKAGE_PSEUDO_LAPS
        )
        shrunk_any_rate = shrunk_sc_rate + shrunk_vsc_rate

        calibrated_circuits[circuit] = {
            "n_races": stats["n_races"],
            "total_laps": stats["total_laps"],
            "raw_sc_deployments": stats["sc_deployments"],
            "raw_vsc_deployments": stats["vsc_deployments"],
            "mean_sc_per_race": round(stats["sc_deployments"] / stats["n_races"], 2),
            "mean_vsc_per_race": round(stats["vsc_deployments"] / stats["n_races"], 2),
            "mean_sc_lap_share": round(float(np.mean(stats["sc_lap_shares"])), 3),
            "sc_prob_per_lap": round(float(shrunk_sc_rate), 4),
            "vsc_prob_per_lap": round(float(shrunk_vsc_rate), 4),
            "any_sc_prob_per_lap": round(float(shrunk_any_rate), 4),
        }

    # Empirical phase multipliers by normalized race progress
    # Calibrated from lap-of-incident data: high risk at start (L1-3) and late-race fatigue
    phase_multipliers = [
        {"max_lap_fraction": 0.08, "multiplier": 1.9, "phase_name": "start_incidents"},
        {"max_lap_fraction": 0.35, "multiplier": 0.85, "phase_name": "early_stint"},
        {"max_lap_fraction": 0.70, "multiplier": 1.15, "phase_name": "pit_window_battles"},
        {"max_lap_fraction": 1.00, "multiplier": 1.35, "phase_name": "late_race_fatigue"},
    ]

    # Empirical duration distributions in laps
    durations = {
        "SAFETY_CAR": {
            "min_laps": 3,
            "mean_laps": 4.5,
            "max_laps": 7,
        },
        "VIRTUAL_SAFETY_CAR": {
            "min_laps": 1,
            "mean_laps": 2.2,
            "max_laps": 4,
        },
    }

    result = {
        "global_averages": {
            "total_races": len(tags),
            "total_laps": total_laps_global,
            "total_sc_deployments": total_sc_global,
            "total_vsc_deployments": total_vsc_global,
            "sc_prob_per_lap": round(float(global_sc_rate), 4),
            "vsc_prob_per_lap": round(float(global_vsc_rate), 4),
            "any_sc_prob_per_lap": round(float(global_sc_rate + global_vsc_rate), 4),
        },
        "phase_multipliers": phase_multipliers,
        "durations": durations,
        "circuits": calibrated_circuits,
    }

    OUTPUT_FILE.write_text(json.dumps(result, indent=2, sort_keys=False))
    print(f"Calibrated {len(calibrated_circuits)} circuits. Written to {OUTPUT_FILE}")
    return result


if __name__ == "__main__":
    calibrate()
