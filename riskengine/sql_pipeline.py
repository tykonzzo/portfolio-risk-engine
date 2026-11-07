"""Build the SQLite database by running the numbered .sql files in order."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))

    def _ln(x):
        return None if x is None or x <= 0 else math.log(x)

    def _sqrt(x):
        return None if x is None or x < 0 else math.sqrt(x)

    # SQLite only ships math functions when compiled with them; register portable versions.
    conn.create_function("LN", 1, _ln, deterministic=True)
    conn.create_function("EXP", 1, lambda x: None if x is None else math.exp(x), deterministic=True)
    conn.create_function("SQRT", 1, _sqrt, deterministic=True)
    conn.create_function("POWER", 2, lambda a, b: None if a is None or b is None else a ** b,
                         deterministic=True)
    return conn


def _run(conn: sqlite3.Connection, name: str) -> None:
    conn.executescript((SQL_DIR / name).read_text())


def build_database(db_path: Path | str, universe: pd.DataFrame, prices: pd.DataFrame,
                   actions: pd.DataFrame) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    _run(conn, "01_schema.sql")
    universe.to_sql("universe", conn, if_exists="append", index=False)
    prices.to_sql("raw_prices", conn, if_exists="append", index=False)
    actions.to_sql("corporate_actions", conn, if_exists="append", index=False)
    for f in ("02_adjust_prices.sql", "03_returns_panel.sql", "04_risk_stats.sql"):
        _run(conn, f)
    conn.commit()
    return conn
