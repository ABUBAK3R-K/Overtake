"""Telemetry table: one row per (race_id, driver, lap_number).

Telemetry is not in Design.md Section 4, so this grain is our choice:
raw car data is ~4 Hz per car (~35k samples per driver per race), far more
than the tyre/lap-time models need. We summarise each lap into a handful of
interpretable numbers instead:

    race_id, driver, lap_number,
    mean_speed, max_speed    - km/h
    mean_throttle            - 0-100 (%)
    full_throttle_pct        - share of samples with throttle >= 99%
    brake_pct                - share of samples with the brake pressed
    drs_open_pct             - share of samples with DRS open
    n_samples                - samples used (low values = patchy data)

Each summary only uses samples recorded *during* that lap, so a lap's row
never contains information from any later lap.
"""

import pandas as pd

TELEMETRY_COLUMNS = [
    "race_id", "driver", "lap_number",
    "mean_speed", "max_speed", "mean_throttle", "full_throttle_pct",
    "brake_pct", "drs_open_pct", "n_samples",
]

# FastF1 DRS codes: 0/1 = off, 8 = eligible but closed, 10/12/14 = open.
DRS_OPEN_CODES = {10, 12, 14}


def assign_samples_to_laps(samples: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Tag each telemetry sample with the lap it was recorded on.

    samples: one driver's car data, needs a `SessionTime` column.
    laps:    that driver's laps, needs `LapNumber`, `LapStartTime`, `Time`
             (Time = session clock when the lap was completed).

    A sample belongs to lap N if LapStartTime(N) <= SessionTime <= Time(N).
    merge_asof finds the latest lap that started before each sample; we then
    drop samples after that lap's end (e.g. after the chequered flag).
    """
    lap_windows = (
        laps[["LapNumber", "LapStartTime", "Time"]]
        .dropna()
        .rename(columns={"Time": "LapEndTime"})
        .sort_values("LapStartTime")
    )
    tagged = pd.merge_asof(
        samples.sort_values("SessionTime"),
        lap_windows,
        left_on="SessionTime",
        right_on="LapStartTime",
        direction="backward",
    )
    inside_lap = tagged["SessionTime"] <= tagged["LapEndTime"]
    return tagged[inside_lap]


def summarise_laps(tagged: pd.DataFrame) -> pd.DataFrame:
    """Collapse tagged samples to one summary row per lap."""
    tagged = tagged.assign(
        full_throttle=tagged["Throttle"] >= 99,
        brake_on=tagged["Brake"].astype(bool),
        drs_open=tagged["DRS"].isin(DRS_OPEN_CODES),
    )
    summary = tagged.groupby("LapNumber").agg(
        mean_speed=("Speed", "mean"),
        max_speed=("Speed", "max"),
        mean_throttle=("Throttle", "mean"),
        full_throttle_pct=("full_throttle", "mean"),
        brake_pct=("brake_on", "mean"),
        drs_open_pct=("drs_open", "mean"),
        n_samples=("Speed", "size"),
    )
    return summary.reset_index().rename(columns={"LapNumber": "lap_number"})


def build_telemetry_table(car_data: dict, laps: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Build the Telemetry table for every driver in a race.

    car_data: FastF1 `session.car_data`, a dict of driver number -> samples.
    laps:     FastF1 `session.laps` (all drivers).
    """
    per_driver = []
    for driver_number, driver_laps in laps.groupby("DriverNumber"):
        if driver_number not in car_data:
            continue  # no telemetry for this car; keep the rest of the race
        tagged = assign_samples_to_laps(car_data[driver_number], driver_laps)
        if tagged.empty:
            continue
        summary = summarise_laps(tagged)
        summary["driver"] = driver_laps["Driver"].iloc[0]
        per_driver.append(summary)

    table = pd.concat(per_driver, ignore_index=True)
    table["race_id"] = race_id
    table["lap_number"] = table["lap_number"].astype("int64")
    return table[TELEMETRY_COLUMNS].sort_values(["lap_number", "driver"]).reset_index(drop=True)


def load_telemetry(session, race_id: str) -> pd.DataFrame:
    """Telemetry table for a loaded FastF1 race session."""
    return build_telemetry_table(session.car_data, session.laps, race_id)
