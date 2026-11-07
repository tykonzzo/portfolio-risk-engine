"""Read the raw NSE price files into one clean long table.

Handles the quirks found in the source files:
  * Indian digit grouping inside quoted numbers ("1,02,40,982")
  * two date formats (31-Mar-2022 and 31-Mar-22)
  * one stock (Bajaj Auto) delivered as a close-only 5-year file plus yearly files
  * series: EQ (normal) and BE (trade-for-trade, same stock) are both regular sessions and are kept.
    An EQ-only export drops every day a stock sits in BE (Tata Investment: 29 Feb to 31 May 2024),
    which is why data/raw/<company>/*_supplement.csv exists. BL (block deals) rows disagree with the
    regular session and are dropped.
  * the same trading day appearing in more than one file
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    out = out.fillna(pd.to_datetime(s, format="%d-%b-%y", errors="coerce"))
    if out.isna().any():
        bad = s[out.isna()].head(3).tolist()
        raise ValueError(f"Unparseable dates, e.g. {bad}")
    return out


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", thousands=",", quotechar='"')
    df.columns = [" ".join(c.split()).upper() for c in df.columns]
    for col in ("VWAP", "VOLUME"):
        if col not in df.columns:
            df[col] = float("nan")
    return df[["DATE", "SERIES", "CLOSE", "VWAP", "VOLUME"]]


def load_raw(raw_dir: Path, universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (prices, stats). prices columns: ticker, trade_date, close, vwap, volume."""
    company_to_ticker = dict(zip(universe["company"], universe["ticker"]))
    frames, files = [], 0
    for folder in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        if folder.name not in company_to_ticker:
            raise KeyError(f"Folder '{folder.name}' is not listed in data/universe.csv")
        for f in sorted(folder.glob("*.csv")):
            df = _read_one(f)
            df["ticker"] = company_to_ticker[folder.name]
            frames.append(df)
            files += 1
    raw = pd.concat(frames, ignore_index=True)
    raw["trade_date"] = _parse_dates(raw["DATE"])
    stats = {"files_read": files, "rows_read": int(len(raw))}

    series = raw["SERIES"].str.strip().str.upper()
    non_regular = ~series.isin(["EQ", "BE"])
    stats["rows_dropped_non_regular_series"] = int(non_regular.sum())
    stats["rows_be_series"] = int((series == "BE").sum())
    raw = raw[~non_regular]

    # Same day in several files: keep the row that carries the most fields.
    raw = raw.assign(_fields=raw[["VWAP", "VOLUME"]].notna().sum(axis=1))
    raw = raw.sort_values(["ticker", "trade_date", "_fields"], ascending=[True, True, False])
    dup = raw.duplicated(["ticker", "trade_date"], keep="first")
    stats["rows_dropped_duplicate_day"] = int(dup.sum())
    raw = raw[~dup]

    prices = raw.rename(columns={"CLOSE": "close", "VWAP": "vwap", "VOLUME": "volume"})[
        ["ticker", "trade_date", "close", "vwap", "volume"]
    ].sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    if prices["close"].isna().any() or (prices["close"] <= 0).any():
        raise ValueError("Found missing or non-positive closing prices")

    stats["rows_final"] = int(len(prices))
    stats["stocks"] = int(prices["ticker"].nunique())
    stats["first_date"] = prices["trade_date"].min().strftime("%Y-%m-%d")
    stats["last_date"] = prices["trade_date"].max().strftime("%Y-%m-%d")
    stats["rows_by_ticker"] = prices.groupby("ticker").size().astype(int).to_dict()
    prices["trade_date"] = prices["trade_date"].dt.strftime("%Y-%m-%d")
    return prices, stats
