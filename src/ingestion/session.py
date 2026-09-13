"""FastF1 setup and race-session loading.

Every ingestion function takes an already-loaded FastF1 session, so the
(slow) network/cache load happens exactly once per race, here.
"""

import tomllib
from pathlib import Path

import fastf1

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RACES_CONFIG = REPO_ROOT / "configs" / "races.toml"


def enable_cache() -> None:
    """Point FastF1 at data/cache/ so each race is downloaded only once.

    After the first download, loading a race works fully offline, which is
    what makes results reproducible (PRD Section 7).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_race_list() -> list[dict]:
    """Read the fixed race list from configs/races.toml."""
    with open(RACES_CONFIG, "rb") as f:
        return tomllib.load(f)["races"]


def load_race_session(season: int, round_number: int, telemetry: bool = True):
    """Load a race ("R") session with laps, telemetry, weather and messages.

    We load weather and race-control messages too even though those tables
    are ingested on the Decision-Making lane: they live in the same cached
    session, and loading them here means both lanes read one identical copy.
    """
    enable_cache()
    session = fastf1.get_session(season, round_number, "R")
    session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
    return session
