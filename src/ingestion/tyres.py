"""TyreStint table: one row per (race_id, driver, lap_number).

Schema (Design.md Section 4, "TyreStint"):
    race_id, driver, compound, stint_start_lap, tyre_age_at_lap
Added:
    lap_number  - the schema lists no lap key, but without one as_of_lap()
                  cannot filter this table. Same grain as the Lap table.
    stint       - FastF1 stint counter (1, 2, 3...) per driver.
    fresh_tyre  - False if the set was already used (e.g. in qualifying).
    compound_raw - FastF1's name as reported. `compound` is that name mapped to
                  the relative SOFT/MEDIUM/HARD scale (they differ only for
                  2018, which uses Pirelli brand names like ULTRASOFT).

Why one row per lap rather than one per stint: "tyre age at lap N" is
per-lap by definition, and a per-stint row would have to store the stint's
*end* lap, which is future information at any lap inside the stint.
"""

import pandas as pd

TYRE_COLUMNS = [
    "race_id", "driver", "lap_number", "stint", "compound", "compound_raw",
    "stint_start_lap", "tyre_age_at_lap", "fresh_tyre",
]


# Softest -> hardest. 2018 reports these brand names; 2019+ already reports
# relative SOFT/MEDIUM/HARD, which pass through unchanged.
_SOFTNESS_ORDER = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"]


def normalise_compounds(raw: pd.Series, relative: bool = True) -> pd.Series:
    """Map brand-name dry compounds onto the relative SOFT/MEDIUM/HARD scale.

    Within one race the observed dry compounds are ranked by softness: softest
    -> SOFT, hardest -> HARD, anything between -> MEDIUM (Pirelli nominates
    three per race, so this mirrors its own C-scale naming). Races that already
    use SOFT/MEDIUM/HARD, and INTERMEDIATE/WET/unknown, are returned untouched.
    """
    # Missing compounds arrive as NaN, None, or the strings "nan"/"None"; make
    # them all a real null so downstream code sees one representation.
    raw = raw.astype(object).where(~raw.astype(str).isin(["nan", "None", "<NA>", ""]), None)
    if not relative:
        return raw
    observed = [c for c in _SOFTNESS_ORDER if c in set(raw.dropna())]
    if not (set(observed) - {"SOFT", "MEDIUM", "HARD"}):
        return raw.where(raw.notna(), None)
    mapping = {c: "MEDIUM" for c in observed}
    mapping[observed[0]] = "SOFT"
    if len(observed) > 1:
        mapping[observed[-1]] = "HARD"
    return raw.map(lambda c: mapping.get(c, c))


def build_tyre_table(laps: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Turn a FastF1 `session.laps` frame into the TyreStint table."""
    table = pd.DataFrame({
        "race_id": race_id,
        "driver": laps["Driver"],
        "lap_number": laps["LapNumber"].astype("int64"),
        "stint": laps["Stint"].astype("Int64"),
        "compound": normalise_compounds(laps["Compound"]).astype("string"),
        "compound_raw": normalise_compounds(laps["Compound"], relative=False).astype("string"),
        # Tyre age comes from FastF1's TyreLife, NOT "laps since stint start":
        # a used set of softs can start the race already 3-4 laps old.
        "tyre_age_at_lap": laps["TyreLife"].astype("Int64"),
        "fresh_tyre": laps["FreshTyre"].astype(bool),
    })
    # First lap of each stint. At any lap inside the stint this lap is already
    # in the past, so this is safe (unlike stint *length*, which would leak).
    table["stint_start_lap"] = (
        table.groupby(["driver", "stint"])["lap_number"].transform("min")
    )
    return table[TYRE_COLUMNS].sort_values(["lap_number", "driver"]).reset_index(drop=True)


def load_tyres(session, race_id: str) -> pd.DataFrame:
    """TyreStint table for a loaded FastF1 race session."""
    return build_tyre_table(session.laps, race_id)
