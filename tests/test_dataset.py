"""Dataset-level checks on the ingested library (skipped if data isn't present)."""
import json

import pytest

from src.ingestion.run_ingestion import MANIFEST, SCHEMA_VERSION
from src.ingestion.session import load_race_list
from src.ingestion.storage import connect
from src.preprocessing.leakage import as_of_lap

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="dataset not ingested")


def test_every_configured_race_ingested_ok_at_current_schema():
    manifest = json.loads(MANIFEST.read_text())
    for race in load_race_list():
        entry = manifest.get(race["race_id"])
        assert entry and entry["status"] == "ok", race["race_id"]
        assert entry["schema_version"] == SCHEMA_VERSION, race["race_id"]


def test_library_meets_fr1_breadth():
    races = connect().sql("select season, circuit from races").df()
    assert len(races) >= 60
    assert races.circuit.nunique() >= 15
    assert races.season.nunique() >= 3


def test_compounds_share_one_dry_vocabulary():
    con = connect()
    vals = set(con.sql("select distinct compound from tyres where compound is not null").df().compound)
    assert vals <= {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"} | {"UNKNOWN", "TEST_UNKNOWN"}


def test_guard_holds_on_older_season_table():
    laps = connect().sql("select * from laps where race_id = '2018_azerbaijan'").df()
    cut = as_of_lap(laps, "lap_number", 10)
    assert cut.lap_number.max() == 10 and len(cut) < len(laps)
