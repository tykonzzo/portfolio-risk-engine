"""Rebuild a daily Nifty 50 close series from 1-minute index bars (github.com/technovusin/nifty50-historical-data, MIT).

NSE's official Nifty close is the average of the index over the last 30 minutes of trading, not the last tick.
The mean of (open+high+low+close)/4 over the last 30 one-minute bars reproduces published closes to within
about 1 point (0.01%); the last bar alone can be off by 100+ points on a volatile day. The method is checked
against six official closes in tests/test_pipeline.py.

    python tools/build_nifty_daily.py <path-to-nifty-clone> --raw data/raw --out data/benchmark/NIFTY50_daily.csv
Output days are exactly the trading days present in the stock files, so the benchmark shares their calendar.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riskengine.ingest import load_raw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clone")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--universe", default="data/universe.csv")
    ap.add_argument("--out", default="data/benchmark/NIFTY50_daily.csv")
    ap.add_argument("--window", type=int, default=30)
    a = ap.parse_args()

    prices, _ = load_raw(Path(a.raw), pd.read_csv(a.universe))
    cal = pd.to_datetime(sorted(prices["trade_date"].unique()))

    files = sorted(glob.glob(f"{a.clone}/1min/20*/*.csv"))
    bars = pd.concat([pd.read_csv(f, usecols=["Timestamp", "Open", "High", "Low", "Close"]) for f in files])
    bars["ts"] = pd.to_datetime(bars["Timestamp"])
    bars = bars.sort_values("ts")
    bars["d"] = bars["ts"].dt.normalize()
    bars["ohlc4"] = bars[["Open", "High", "Low", "Close"]].mean(axis=1)
    tail = bars.groupby("d").tail(a.window)
    daily = tail.groupby("d").agg(Close=("ohlc4", "mean"), Bars=("ohlc4", "size"))
    n_bars = bars.groupby("d").size().rename("SessionBars")

    # Days in the stock calendar with no index bars (e.g. the evening Diwali Muhurat session) carry the
    # previous close forward, so the benchmark return is 0 that day. They are flagged in the file.
    filled = [d for d in cal if d not in daily.index]
    if len(filled) > 3:
        raise SystemExit(f"{len(filled)} stock-calendar days have no Nifty bars, e.g. {[d.date() for d in filled[:5]]}")
    out = daily.reindex(cal)
    out["Close"] = out["Close"].ffill()
    out = out.join(n_bars).fillna({"SessionBars": 0})
    short = out[(out.SessionBars < 300) & (out.SessionBars > 0)]
    out.insert(0, "Date", out.index.strftime("%d-%b-%Y"))
    out["Close"] = out["Close"].round(2)
    out["Source"] = "technovusin/nifty50-historical-data, mean OHLC/4 of last 30 one-minute bars"
    out.loc[out.SessionBars == 0, "Source"] = "carried forward: no index bars in source for this session"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out[["Date", "Close", "SessionBars", "Source"]].to_csv(a.out, index=False)
    print(f"{len(out)} days {out.Date.iloc[0]} -> {out.Date.iloc[-1]} -> {a.out}")
    print("carried forward:", [d.strftime("%Y-%m-%d") for d in filled])
    print(f"short sessions (<300 bars): {len(short)}", [d.strftime('%Y-%m-%d') for d in short.index])


if __name__ == "__main__":
    main()
