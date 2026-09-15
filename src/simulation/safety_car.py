"""Safety-Car ("Ghost Car") Model & Gap Compression (Design.md Section 6.5).

Provides empirical safety-car deployment probabilities by circuit and race phase,
plus field compression ('bunching') when neutralised.
"""

from __future__ import annotations

import copy
import numpy as np

from src.simulation.state import RaceState

# Baseline empirical per-lap safety car probabilities by circuit
# (Synthesised from historical 2021-2023 FIA data)
CIRCUIT_SC_BASE_RATES = {
    "Sakhir": 0.025,        # Bahrain: relatively low
    "Melbourne": 0.075,     # Albert Park: high SC frequency
    "Monaco": 0.085,        # Monaco: very high SC frequency
    "Barcelona": 0.020,     # Spain: low SC frequency
    "Silverstone": 0.055,   # Britain: medium-high SC frequency
    "Zandvoort": 0.060,     # Netherlands: medium-high SC frequency
    "Monza": 0.040,         # Italy: medium
    "Marina Bay": 0.095,    # Singapore: highest SC frequency (~100% of races have SC)
}
DEFAULT_SC_BASE_RATE = 0.040


def safety_car_probability(circuit: str, lap_fraction: float) -> float:
    """Empirical probability of a Safety Car deployment on a given lap.

    Args:
        circuit: Name of circuit (e.g. 'Monaco', 'Sakhir')
        lap_fraction: Normalized race progress lap / total_laps in [0.0, 1.0]

    Returns:
        Probability in [0.0, 1.0] per lap.
    """
    base_rate = CIRCUIT_SC_BASE_RATES.get(circuit, DEFAULT_SC_BASE_RATE)

    # Phase modifier: Lap 1-3 (start incidents) and late race (fatigue/battles) have higher rates
    if lap_fraction <= 0.08:
        phase_mod = 1.8   # First ~4-5 laps
    elif lap_fraction <= 0.35:
        phase_mod = 0.9   # Early stint
    elif lap_fraction <= 0.70:
        phase_mod = 1.1   # Mid race pit window & battles
    else:
        phase_mod = 1.3   # Late race

    return float(np.clip(base_rate * phase_mod, 0.005, 0.25))


def apply_safety_car_bunching(state: RaceState, interval_spacing: float = 0.8) -> RaceState:
    """Compress gaps towards the leader when the Safety Car is deployed.

    During a Safety Car period, cars bunch up behind the leader with roughly
    0.5s - 1.2s intervals between successive positions.

    Args:
        state: Current RaceState
        interval_spacing: Target spacing in seconds between adjacent cars

    Returns:
        New RaceState with compressed gaps.
    """
    new_state = copy.deepcopy(state)
    new_state.safety_car = True
    new_state.status = "SAFETY_CAR"

    # Sort drivers by their current position
    sorted_drivers = sorted(state.positions.items(), key=lambda item: item[1])
    
    compressed_gaps = {}
    current_gap = 0.0
    for idx, (driver, pos) in enumerate(sorted_drivers):
        if idx == 0:
            compressed_gaps[driver] = 0.0
        else:
            # Natural small jitter in SC queue
            jitter = np.clip(np.random.normal(0, 0.1), -0.2, 0.3)
            current_gap += interval_spacing + jitter
            # Cannot be larger than the original uncompressed gap
            orig_gap = state.gaps.get(driver, current_gap)
            compressed_gaps[driver] = round(float(min(orig_gap, current_gap)), 2)

    new_state.gaps = compressed_gaps
    return new_state
