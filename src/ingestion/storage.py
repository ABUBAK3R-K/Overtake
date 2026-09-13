"""Parquet storage and DuckDB access for every ingested table.

Layout: data/processed/<table>/<race_id>.parquet

One folder per table and one file per race means:
  - re-ingesting a race only rewrites that race's files;
  - DuckDB reads a whole table across races with one glob, e.g.
        SELECT * FROM read_parquet('data/processed/laps/*.parquet')

Both lanes write here with the same layout (the Decision-Making lane's
pit_stops / race_control / weather tables included), so connect() exposes
every table from either lane as a view.

Note: this is plain storage. It does no lap filtering. Code that builds
features or race state must still go through as_of_lap().
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.ingestion.session import PROCESSED_DIR


def table_path(table: str, race_id: str, base_dir: Path = PROCESSED_DIR) -> Path:
    return base_dir / table / f"{race_id}.parquet"


def write_table(df: pd.DataFrame, table: str, race_id: str,
                base_dir: Path = PROCESSED_DIR) -> Path:
    """Write one race's rows for one table, replacing any previous file."""
    path = table_path(table, race_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def table_exists(table: str, race_id: str, base_dir: Path = PROCESSED_DIR) -> bool:
    return table_path(table, race_id, base_dir).exists()


def connect(base_dir: Path = PROCESSED_DIR) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with one view per table folder.

    Example:
        con = connect()
        con.sql("SELECT driver, AVG(lap_time) FROM laps "
                "WHERE race_id = '2023_bahrain' GROUP BY driver").df()
    """
    con = duckdb.connect()
    if not base_dir.exists():
        return con
    for table_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        if not any(table_dir.glob("*.parquet")):
            continue
        glob = (table_dir / "*.parquet").as_posix()
        # union_by_name: tolerate a column added to newer files later on.
        con.execute(
            f"CREATE VIEW {table_dir.name} AS "
            f"SELECT * FROM read_parquet('{glob}', union_by_name = true)"
        )
    return con
