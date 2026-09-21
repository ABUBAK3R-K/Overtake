"""Safety-Car ("Ghost Car") Model & Gap Compression (Design.md Section 6.5, PRD FR-5).

Provides empirical safety-car deployment probabilities by circuit and race phase,
calibrated across 112 multi-season races (2018-2024, 33 circuits) with Empirical Bayes
shrinkage, plus field compression ('bunching') when neutralised.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

from src.simulation.state import RaceState

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "safety_car_rates.json"

# Circuit aliases mapping common variations and informal track names to canonical names
CIRCUIT_ALIASES: dict[str, str] = {
    "bahrain": "Sakhir",
    "sakhir": "Sakhir",
    "singapore": "Marina Bay",
    "marina bay": "Marina Bay",
    "albert park": "Melbourne",
    "melbourne": "Melbourne",
    "australia": "Melbourne",
    "monaco": "Monaco",
    "monte carlo": "Monaco",
    "silverstone": "Silverstone",
    "great britain": "Silverstone",
    "britain": "Silverstone",
    "barcelona": "Barcelona",
    "catalunya": "Barcelona",
    "spain": "Barcelona",
    "monza": "Monza",
    "italy": "Monza",
    "zandvoort": "Zandvoort",
    "netherlands": "Zandvoort",
    "dutch": "Zandvoort",
    "spa": "Spa-Francorchamps",
    "spa-francorchamps": "Spa-Francorchamps",
    "belgium": "Spa-Francorchamps",
    "baku": "Baku",
    "azerbaijan": "Baku",
    "suzuka": "Suzuka",
    "japan": "Suzuka",
    "austin": "Austin",
    "cota": "Austin",
    "united states": "Austin",
    "usa": "Austin",
    "mexico": "Mexico City",
    "mexico city": "Mexico City",
    "interlagos": "São Paulo",
    "sao paulo": "São Paulo",
    "são paulo": "São Paulo",
    "brazil": "São Paulo",
    "yas marina": "Yas Island",
    "yas island": "Yas Island",
    "abu dhabi": "Yas Island",
    "jeddah": "Jeddah",
    "saudi arabia": "Jeddah",
    "las vegas": "Las Vegas",
    "vegas": "Las Vegas",
    "miami": "Miami",
    "budapest": "Budapest",
    "hungary": "Budapest",
    "hungaroring": "Budapest",
    "spielberg": "Spielberg",
    "austria": "Spielberg",
    "red bull ring": "Spielberg",
    "imola": "Imola",
    "lusail": "Lusail",
    "qatar": "Lusail",
    "le castellet": "Le Castellet",
    "paul ricard": "Le Castellet",
    "france": "Le Castellet",
    "shanghai": "Shanghai",
    "china": "Shanghai",
    "sochi": "Sochi",
    "russia": "Sochi",
}

# Default global rates when config is unavailable
DEFAULT_GLOBAL_RATES = {
    "sc_prob_per_lap": 0.0136,
    "vsc_prob_per_lap": 0.0077,
    "any_sc_prob_per_lap": 0.0214,
}

DEFAULT_PHASE_MULTIPLIERS = [
    {"max_lap_fraction": 0.08, "multiplier": 1.9, "phase_name": "start_incidents"},
    {"max_lap_fraction": 0.35, "multiplier": 0.85, "phase_name": "early_stint"},
    {"max_lap_fraction": 0.70, "multiplier": 1.15, "phase_name": "pit_window_battles"},
    {"max_lap_fraction": 1.00, "multiplier": 1.35, "phase_name": "late_race_fatigue"},
]

DEFAULT_DURATION_SPECS = {
    "SAFETY_CAR": {"min_laps": 3, "mean_laps": 4.5, "max_laps": 7},
    "VIRTUAL_SAFETY_CAR": {"min_laps": 1, "mean_laps": 2.2, "max_laps": 4},
}


def _load_calibrated_rates() -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            globals_ = data.get("global_averages", DEFAULT_GLOBAL_RATES)
            circuits_ = data.get("circuits", {})
            phases_ = data.get("phase_multipliers", DEFAULT_PHASE_MULTIPLIERS)
            durations_ = data.get("durations", DEFAULT_DURATION_SPECS)
            return globals_, circuits_, phases_, durations_
        except Exception as err:
            log.warning("Failed to load safety car rates config: %s", err)

    return DEFAULT_GLOBAL_RATES, {}, DEFAULT_PHASE_MULTIPLIERS, DEFAULT_DURATION_SPECS


GLOBAL_AVERAGES, CALIBRATED_CIRCUITS, PHASE_MULTIPLIERS, DURATION_SPECS = _load_calibrated_rates()

# Backward-compatible dictionary of base rates
CIRCUIT_SC_BASE_RATES: dict[str, float] = {
    c: data["any_sc_prob_per_lap"] for c, data in CALIBRATED_CIRCUITS.items()
}
DEFAULT_SC_BASE_RATE: float = GLOBAL_AVERAGES.get("any_sc_prob_per_lap", 0.0214)


def resolve_circuit_name(circuit: str) -> str:
    """Normalize circuit name string using aliases.

    Args:
        circuit: Input circuit name or alias.

    Returns:
        Canonical circuit name matching calibration tables.
    """
    if not circuit:
        return ""
    clean = circuit.strip()
    # Check exact match
    if clean in CALIBRATED_CIRCUITS:
        return clean
    # Check alias match
    lower = clean.lower()
    if lower in CIRCUIT_ALIASES:
        return CIRCUIT_ALIASES[lower]
    # Check partial contains in aliases
    for alias, canonical in CIRCUIT_ALIASES.items():
        if alias in lower or lower in alias:
            return canonical
    return clean


def safety_car_probability(circuit: str, lap_fraction: float, event_type: str = "ANY") -> float:
    """Empirical probability of a Safety Car deployment on a given lap.

    Calibrated against 112 multi-season races with Empirical Bayes shrinkage
    and piecewise race-phase modifiers.

    Args:
        circuit: Name of circuit (e.g. 'Monaco', 'Sakhir', 'Melbourne')
        lap_fraction: Normalized race progress lap / total_laps in [0.0, 1.0]
        event_type: 'ANY' (SC or VSC), 'SAFETY_CAR', or 'VIRTUAL_SAFETY_CAR'

    Returns:
        Probability in [0.0, 1.0] per lap.
    """
    canonical = resolve_circuit_name(circuit)
    circuit_data = CALIBRATED_CIRCUITS.get(canonical)

    if circuit_data is not None:
        if event_type == "SAFETY_CAR":
            base_rate = circuit_data.get("sc_prob_per_lap", GLOBAL_AVERAGES["sc_prob_per_lap"])
        elif event_type == "VIRTUAL_SAFETY_CAR":
            base_rate = circuit_data.get("vsc_prob_per_lap", GLOBAL_AVERAGES["vsc_prob_per_lap"])
        else:
            base_rate = circuit_data.get("any_sc_prob_per_lap", GLOBAL_AVERAGES["any_sc_prob_per_lap"])
    else:
        # Unknown circuit: fallback to CIRCUIT_SC_BASE_RATES or global average
        base_rate = CIRCUIT_SC_BASE_RATES.get(circuit, DEFAULT_SC_BASE_RATE)

    # Piecewise phase modifier
    phase_mod = 1.0
    for phase in PHASE_MULTIPLIERS:
        if lap_fraction <= phase["max_lap_fraction"]:
            phase_mod = phase["multiplier"]
            break

    return float(np.clip(base_rate * phase_mod, 0.005, 0.25))


def sample_safety_car_duration(circuit: str = "", event_type: str = "SAFETY_CAR") -> int:
    """Sample the duration of a Safety Car or VSC period in laps.

    Empirically, full Safety Car periods last 3 to 7 laps (mean 4.5),
    while VSC neutralisations last 1 to 4 laps (mean 2.2).

    Args:
        circuit: Optional circuit name for future track-specific duration tuning.
        event_type: 'SAFETY_CAR' or 'VIRTUAL_SAFETY_CAR'

    Returns:
        Integer number of laps for the incident period.
    """
    spec = DURATION_SPECS.get(event_type, DURATION_SPECS.get("SAFETY_CAR", {"min_laps": 3, "mean_laps": 4.5, "max_laps": 7}))
    mean_laps = float(spec.get("mean_laps", 4.5))
    min_laps = int(spec.get("min_laps", 3))
    max_laps = int(spec.get("max_laps", 7))
    std = 1.1 if event_type == "SAFETY_CAR" else 0.7

    sampled = int(round(np.random.normal(mean_laps, std)))
    return int(np.clip(sampled, min_laps, max_laps))


def get_circuit_safety_car_profile(circuit: str) -> dict[str, Any]:
    """Retrieve calibrated safety car statistics and deployment profile for a circuit.

    Args:
        circuit: Name of circuit or alias.

    Returns:
        Dictionary containing historical deployments, per-lap rates, and phase curves.
    """
    canonical = resolve_circuit_name(circuit)
    circuit_data = CALIBRATED_CIRCUITS.get(canonical, {})

    return {
        "circuit": canonical if canonical else circuit,
        "n_races": circuit_data.get("n_races", 0),
        "total_laps": circuit_data.get("total_laps", 0),
        "mean_sc_per_race": circuit_data.get("mean_sc_per_race", round(GLOBAL_AVERAGES.get("total_sc_deployments", 0) / max(1, GLOBAL_AVERAGES.get("total_races", 1)), 2)),
        "mean_vsc_per_race": circuit_data.get("mean_vsc_per_race", round(GLOBAL_AVERAGES.get("total_vsc_deployments", 0) / max(1, GLOBAL_AVERAGES.get("total_races", 1)), 2)),
        "mean_sc_lap_share": circuit_data.get("mean_sc_lap_share", 0.0),
        "sc_prob_per_lap": circuit_data.get("sc_prob_per_lap", GLOBAL_AVERAGES.get("sc_prob_per_lap", 0.0136)),
        "vsc_prob_per_lap": circuit_data.get("vsc_prob_per_lap", GLOBAL_AVERAGES.get("vsc_prob_per_lap", 0.0077)),
        "any_sc_prob_per_lap": circuit_data.get("any_sc_prob_per_lap", GLOBAL_AVERAGES.get("any_sc_prob_per_lap", 0.0214)),
        "phase_multipliers": PHASE_MULTIPLIERS,
        "durations": DURATION_SPECS,
    }


def apply_safety_car_bunching(state: RaceState, interval_spacing: float = 0.8) -> RaceState:
    """Compress gaps towards the leader when the Safety Car is deployed.

    During a Safety Car period, cars bunch up behind the leader with roughly
    0.5s - 1.2s intervals between successive positions. Updates both `gaps`
    and `intervals`.

    Args:
        state: Current RaceState
        interval_spacing: Target spacing in seconds between adjacent cars

    Returns:
        New RaceState with compressed gaps and intervals.
    """
    new_state = copy.deepcopy(state)
    new_state.safety_car = True
    if new_state.status == "RACING":
        new_state.status = "SAFETY_CAR"

    # Separate active and retired drivers
    active_drivers = [
        (driver, pos) for driver, pos in state.positions.items()
        if driver not in state.retired
    ]
    # Sort active drivers by current position
    sorted_drivers = sorted(active_drivers, key=lambda item: item[1])

    compressed_gaps: dict[str, float] = {}
    compressed_intervals: dict[str, float] = {}
    current_gap = 0.0

    for idx, (driver, pos) in enumerate(sorted_drivers):
        if idx == 0:
            compressed_gaps[driver] = 0.0
            compressed_intervals[driver] = 0.0
        else:
            # Natural small reaction-time jitter in SC queue
            jitter = float(np.clip(np.random.normal(0, 0.08), -0.2, 0.3))
            step_interval = max(0.4, interval_spacing + jitter)
            current_gap += step_interval

            # Cannot be larger than the original uncompressed gap
            orig_gap = state.gaps.get(driver, current_gap)
            final_gap = round(float(min(orig_gap, current_gap)), 2)

            prev_driver = sorted_drivers[idx - 1][0]
            step = max(0.2, final_gap - compressed_gaps[prev_driver])

            compressed_gaps[driver] = final_gap
            compressed_intervals[driver] = round(float(step), 2)

    # Preserve retired drivers
    for driver in state.retired:
        if driver in state.gaps:
            compressed_gaps[driver] = state.gaps[driver]
        if driver in state.intervals:
            compressed_intervals[driver] = state.intervals[driver]

    new_state.gaps = compressed_gaps
    new_state.intervals = compressed_intervals
    return new_state
