"""Raw NSE CSVs -> cleaned, adjusted prices in SQLite, plus SQL helpers."""
import math
import os
import re
import sqlite3

import numpy as np
import pandas as pd

from corporate_actions import ACTIONS, apply_actions, validate_actions
from load_data import load_prices

SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql")


def _read_sql(name: str) -> str:
    with open(os.path.join(SQL_DIR, name), encoding="utf-8") as f:
        return f.read()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # Math functions are not compiled into every SQLite build; register them explicitly.
    conn.create_function("sqrt", 1, lambda x: None if x is None else math.sqrt(x), deterministic=True)
    conn.create_function("ln", 1, lambda x: None if x is None or x <= 0 else math.log(x), deterministic=True)
    conn.create_function("exp", 1, lambda x: None if x is None else math.exp(x), deterministic=True)
    return conn


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    text = _read_sql("analytics.sql")
    blocks = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    queries = {blocks[i]: blocks[i + 1].strip() for i in range(1, len(blocks), 2)}
    return pd.read_sql_query(queries[name], conn)


def fill_gaps(prices: pd.DataFrame):
    """Align every ticker to the shared trading calendar.

    Any interior gap is filled by log-linear interpolation and flagged. This keeps
    portfolio returns aligned; the flagged rows are excluded from risk statistics.
    """
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    parts, gaps = [], []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        span = calendar[(calendar >= g.index.min()) & (calendar <= g.index.max())]
        missing = span.difference(g.index)
        full = g.reindex(span)
        full["is_interpolated"] = full["close_adj"].isna().astype(int)
        if len(missing):
            logp = np.log(full["close_adj"])
            pos = pd.Series(np.arange(len(full)), index=full.index, dtype=float)
            known = logp.notna()
            full["close_adj"] = np.exp(np.interp(pos, pos[known], logp[known]))
            full["adj_factor"] = full["adj_factor"].ffill().bfill()
            gaps.append(
                {
                    "ticker": ticker,
                    "missing_trading_days": int(len(missing)),
                    "gap_start": str(missing.min().date()),
                    "gap_end": str(missing.max().date()),
                }
            )
        full["ticker"] = ticker
        full["company"] = g["company"].iloc[0]
        parts.append(full.reset_index().rename(columns={"index": "date"}))
    out = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return out, gaps


def build(raw_root: str, db_path: str) -> dict:
    raw = load_prices(raw_root)
    actions = validate_actions(raw)
    adj = apply_actions(raw)
    adj = adj.rename(columns={"close": "close_raw"})
    adj["close_raw"] = adj["close_raw"].astype(float)
    filled, gaps = fill_gaps(adj)

    if os.path.exists(db_path):
        os.remove(db_path)
    conn = connect(db_path)
    conn.executescript(_read_sql("schema.sql"))
    rows = filled.assign(date=filled["date"].dt.strftime("%Y-%m-%d"))[
        ["date", "ticker", "company", "close_raw", "adj_factor", "close_adj", "is_interpolated"]
    ]
    # close_raw is NaN for interpolated rows -> NULL in SQLite
    conn.executemany(
        "INSERT INTO prices VALUES (?,?,?,?,?,?,?)",
        [
            (r.date, r.ticker, r.company, None if pd.isna(r.close_raw) else float(r.close_raw),
             float(r.adj_factor), float(r.close_adj), int(r.is_interpolated))
            for r in rows.itertuples(index=False)
        ],
    )
    conn.executemany(
        "INSERT INTO corporate_actions VALUES (?,?,?,?)",
        [(r.ticker, r.ex_date, float(r.factor), float(r.observed)) for r in actions.itertuples(index=False)],
    )
    conn.executescript(_read_sql("views.sql"))
    conn.commit()

    quality = {
        "tickers": int(filled["ticker"].nunique()),
        "price_rows_loaded": int(len(raw)),
        "price_rows_after_gap_fill": int(len(filled)),
        "first_date": str(filled["date"].min().date()),
        "last_date": str(filled["date"].max().date()),
        "non_eq_rows_dropped": int(raw.attrs.get("non_eq_dropped", 0)),
        "overlapping_rows_dropped": int(raw.attrs.get("duplicates_dropped", 0)),
        "interpolated_rows": int(filled["is_interpolated"].sum()),
        "gaps": gaps,
        "corporate_actions": actions.to_dict("records"),
    }
    conn.close()
    return quality
