"""Recover days that an EQ-only NSE export drops because the stock traded in another series.

Tata Investment Corporation sat in NSE's BE (trade-for-trade) series from 29 Feb to 31 May 2024.
An export filtered to series EQ silently omits those 62 trading days. This script rebuilds them
from NSE's official daily "sec_bhavdata_full" files (mirrored in github.com/tilak999/NSE-Data-bank).

    python tools/build_bhavcopy_supplement.py <path-to-NSE-Data-bank-clone> \
        --symbol TATAINVEST --start 2024-02-29 --end 2024-05-31 --out "data/raw/<company>/..."

A day whose file is absent from the mirror is filled only when the NEXT session's PREV_CLOSE
proves the close exactly (NSE records the last close as the next day's previous close). Such
rows carry no open/high/low/VWAP/volume and are labelled in the SOURCE column.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

COLS = ["DATE", "SERIES", "PREV. CLOSE", "OPEN", "HIGH", "LOW", "CLOSE", "VWAP", "VOLUME", "SOURCE"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--series", default="BE")
    ap.add_argument("--expected-dates", help="CSV with a trade_date column: every day that must be present")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = pd.Timestamp(a.start), pd.Timestamp(a.end)

    rows = []
    for f in sorted(Path(a.bank, "data").glob("sec_bhavdata_full_*.csv")):
        m = re.search(r"_(\d{2})(\d{2})(\d{4})\.csv$", f.name)
        if not m or not (lo <= pd.Timestamp(f"{m[3]}-{m[2]}-{m[1]}") <= hi):
            continue
        df = pd.read_csv(f, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        for c in ("SYMBOL", "SERIES"):
            df[c] = df[c].str.strip()
        df = df[(df.SYMBOL == a.symbol) & (df.SERIES == a.series)]
        rows.append(df)
    b = pd.concat(rows, ignore_index=True)
    b["DATE"] = pd.to_datetime(b["DATE1"].str.strip(), format="%d-%b-%Y")
    for c in ("PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.drop_duplicates("DATE").sort_values("DATE").set_index("DATE")
    out = pd.DataFrame({
        "SERIES": a.series, "PREV. CLOSE": b.PREV_CLOSE, "OPEN": b.OPEN_PRICE, "HIGH": b.HIGH_PRICE,
        "LOW": b.LOW_PRICE, "CLOSE": b.CLOSE_PRICE, "VWAP": b.AVG_PRICE, "VOLUME": b.TTL_TRD_QNTY,
        "SOURCE": "NSE sec_bhavdata_full"})

    if a.expected_dates:
        want = pd.to_datetime(pd.read_csv(a.expected_dates)["trade_date"])
        want = want[(want >= lo) & (want <= hi)]
        for d in sorted(set(want) - set(out.index)):
            nxt = out.index[out.index > d]
            if len(nxt) == 0:
                raise SystemExit(f"{d.date()}: absent and no later session to derive it from")
            nd = nxt[0]
            close = float(out.loc[nd, "PREV. CLOSE"])
            prev = out.index[out.index < d]
            out.loc[d] = [a.series, float(out.loc[prev[-1], "CLOSE"]) if len(prev) else None,
                          None, None, None, close, None, None,
                          f"derived: PREV_CLOSE of {nd.date()} session (exact close; no OHLC/volume)"]
            out = out.sort_index()           # keep the index ordered so the next lookup is correct
        absent = sorted(set(want) - set(out.index))
        assert not absent, absent

    out.insert(0, "DATE", out.index.strftime("%d-%b-%Y"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out[COLS].to_csv(a.out, index=False)
    print(f"{len(out)} rows -> {a.out}  ({(out.SOURCE.str.startswith('derived')).sum()} derived)")


if __name__ == "__main__":
    main()
