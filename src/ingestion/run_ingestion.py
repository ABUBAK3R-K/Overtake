"""Ingest the Prediction Lane tables for every race in configs/races.toml.

Usage (from the repo root):
    python -m src.ingestion.run_ingestion                 # all races
    python -m src.ingestion.run_ingestion 2023_bahrain    # specific races
    python -m src.ingestion.run_ingestion --force         # rebuild existing

Writes, per race: races, laps, tyres, telemetry (see storage.py for layout).
Races whose tables all exist are skipped, so a failed run can simply be
re-run and only the missing races are fetched. One race failing (e.g. a
network timeout) is reported but does not stop the others.
"""

import argparse
import logging
import sys
import time

from src.ingestion.laps import load_laps
from src.ingestion.pit_stops import load_pit_stops
from src.ingestion.race_control import load_race_control
from src.ingestion.race_info import build_race_table
from src.ingestion.session import load_race_list, load_race_session
from src.ingestion.storage import table_exists, write_table
from src.ingestion.telemetry import load_telemetry
from src.ingestion.tyres import load_tyres
from src.ingestion.weather import load_weather

TABLES = ["races", "laps", "tyres", "telemetry", "weather", "race_control", "pit_stops"]

log = logging.getLogger("overtake.ingestion")


def ingest_race(race: dict) -> dict[str, int]:
    """Load one race session and write all Prediction & Decision lane tables.

    Returns row counts per table, for the run summary.
    """
    race_id = race["race_id"]
    session = load_race_session(race["season"], race["round"])
    tables = {
        "races": build_race_table(session, race_id, race["season"], race["round"]),
        "laps": load_laps(session, race_id),
        "tyres": load_tyres(session, race_id),
        "telemetry": load_telemetry(session, race_id),
        "weather": load_weather(session, race_id),
        "race_control": load_race_control(session, race_id),
        "pit_stops": load_pit_stops(session, race_id),
    }
    for name, df in tables.items():
        write_table(df, name, race_id)
    return {name: len(df) for name, df in tables.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("race_ids", nargs="*", help="race_ids to ingest (default: all)")
    parser.add_argument("--force", action="store_true", help="re-ingest even if files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fastf1").setLevel(logging.WARNING)  # FastF1 is very chatty

    races = load_race_list()
    if args.race_ids:
        unknown = set(args.race_ids) - {r["race_id"] for r in races}
        if unknown:
            parser.error(f"not in configs/races.toml: {sorted(unknown)}")
        races = [r for r in races if r["race_id"] in args.race_ids]

    failed = []
    for race in races:
        race_id = race["race_id"]
        if not args.force and all(table_exists(t, race_id) for t in TABLES):
            log.info("%s: already ingested, skipping", race_id)
            continue
        start = time.time()
        try:
            counts = ingest_race(race)
        except Exception:
            log.exception("%s: FAILED", race_id)
            failed.append(race_id)
            continue
        log.info("%s: done in %.0fs %s", race_id, time.time() - start, counts)

    if failed:
        log.error("Failed races (re-run to retry): %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
