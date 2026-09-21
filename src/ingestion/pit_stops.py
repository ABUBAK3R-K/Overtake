"""PitStop table: one row per pit stop event.

Schema (Design.md Section 4, "PitStop"):
    race_id, driver, lap, pit_duration, compound_before, compound_after
Added:
    stint, tyre_age_before, fresh_tyre_after

LEAKAGE SAFETY:
Each pit stop is stamped with `lap` (the in-lap where the driver entered the pits)
so that `as_of_lap()` can strictly exclude future pit stops.
"""

from __future__ import annotations

import pandas as pd

from src.ingestion.tyres import normalise_compounds

PIT_STOP_COLUMNS = [
    "race_id", "driver", "lap", "stint", "pit_duration",
    "compound_before", "compound_after", "tyre_age_before", "fresh_tyre_after",
]


def _seconds(td: pd.Series) -> pd.Series:
    """Convert timedelta series to float seconds."""
    if td is None:
        return pd.Series(dtype=float)
    return td.dt.total_seconds()


def build_pit_stops_table(laps: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Turn FastF1 session laps into the PitStop table."""
    if laps is None or laps.empty:
        return pd.DataFrame(columns=PIT_STOP_COLUMNS)

    df = laps.sort_values(["Driver", "LapNumber"]).copy()
    # Same relative SOFT/MEDIUM/HARD scale as the tyres table (matters for 2018).
    df["Compound"] = normalise_compounds(df["Compound"])  # missing -> real null -> "UNKNOWN" below
    pit_in_mask = df["PitInTime"].notna()

    pit_laps = df[pit_in_mask].copy()
    if pit_laps.empty:
        return pd.DataFrame(columns=PIT_STOP_COLUMNS)

    rows = []
    # Group by driver to determine before/after compounds and stint info
    for driver, d_laps in df.groupby("Driver"):
        d_laps = d_laps.sort_values("LapNumber").reset_index(drop=True)
        for idx, row in d_laps.iterrows():
            if pd.isna(row["PitInTime"]):
                continue

            lap_num = int(row["LapNumber"])
            stint = int(row["Stint"]) if pd.notna(row["Stint"]) else 1
            compound_before = str(row["Compound"]) if pd.notna(row["Compound"]) else "UNKNOWN"
            age_before = int(row["TyreLife"]) if pd.notna(row["TyreLife"]) else 0

            # Find next lap for compound_after and pit duration / out time
            compound_after = "UNKNOWN"
            fresh_after = True
            pit_duration = float("nan")  # unknown unless the source gives pit-in/out times

            if idx + 1 < len(d_laps):
                next_row = d_laps.iloc[idx + 1]
                if pd.notna(next_row["Compound"]):
                    compound_after = str(next_row["Compound"])
                if pd.notna(next_row["FreshTyre"]):
                    fresh_after = bool(next_row["FreshTyre"])
                
                # If PitOutTime is available on next lap or same lap
                if pd.notna(next_row["PitOutTime"]) and pd.notna(row["PitInTime"]):
                    duration = (next_row["PitOutTime"] - row["PitInTime"]).total_seconds()
                    if 10.0 <= duration <= 120.0:
                        pit_duration = float(duration)
                elif pd.notna(row["PitOutTime"]) and pd.notna(row["PitInTime"]):
                    duration = (row["PitOutTime"] - row["PitInTime"]).total_seconds()
                    if 10.0 <= duration <= 120.0:
                        pit_duration = float(duration)

            rows.append({
                "race_id": race_id,
                "driver": str(driver),
                "lap": lap_num,
                "stint": stint,
                "pit_duration": float(pit_duration),
                "compound_before": compound_before,
                "compound_after": compound_after,
                "tyre_age_before": age_before,
                "fresh_tyre_after": fresh_after,
            })

    if not rows:
        return pd.DataFrame(columns=PIT_STOP_COLUMNS)

    table = pd.DataFrame(rows)
    return table[PIT_STOP_COLUMNS].sort_values(["lap", "driver"]).reset_index(drop=True)


def load_pit_stops(session, race_id: str) -> pd.DataFrame:
    """PitStop table for a loaded FastF1 race session."""
    return build_pit_stops_table(session.laps, race_id)
