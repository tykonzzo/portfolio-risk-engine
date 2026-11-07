"""Load and clean NSE daily price CSVs into a tidy DataFrame.

Handles the quirks found in the raw files:
  * UTF-8 BOM and padded column names
  * Indian digit grouping in numbers ("1,02,40,982")
  * Mixed date formats (31-Mar-2022 and 31-Mar-22)
  * Non-EQ series rows (e.g. BL block deals) that carry off-market prices
  * Overlapping files (a file can span more than one financial year)
  * Files that hold only DATE/SERIES/CLOSE
"""
import glob
import os
import sys

import pandas as pd

TICKERS = {
    "Bajaj Auto Limited": "BAJAJ-AUTO",
    "Bharat Heavy Electricals Limited": "BHEL",
    "Dr. Lal PathLabs Limited": "LALPATHLAB",
    "HDFC Bank Limited": "HDFCBANK",
    "Hindustan Unilever Limited": "HINDUNILVR",
    "Indian Oil Corporation Limited": "IOC",
    "Kotak Mahindra Bank Limited": "KOTAKBANK",
    "Larsen & Toubro Limited": "LT",
    "Mahindra & Mahindra Limited": "M&M",
    "Shriram Finance Limited": "SHRIRAMFIN",
    "State Bank of India": "SBIN",
    "Sun Pharmaceutical Industries Limited": "SUNPHARMA",
    "Tata Investment Corporation Limited": "TATAINVEST",
    "United Spirits Limited": "MCDOWELL-N",
    "Vedanta Limited": "VEDL",
}


def _parse_dates(s: pd.Series) -> pd.Series:
    full = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d-%b-%y", errors="coerce")
    return full.fillna(short)


def load_prices(root: str) -> pd.DataFrame:
    frames = []
    non_eq_dropped = 0
    for path in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
        company = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path, encoding="utf-8-sig", thousands=",")
        df.columns = [c.strip() for c in df.columns]
        df["date"] = _parse_dates(df["DATE"])
        keep = df["SERIES"].str.strip() == "EQ"
        non_eq_dropped += int((~keep).sum())
        df = df[keep]
        out = df[["date", "CLOSE"]].rename(columns={"CLOSE": "close"})
        out["company"] = company
        out["ticker"] = TICKERS.get(company, company)
        out["source_file"] = os.path.basename(path)
        frames.append(out)
    allp = pd.concat(frames, ignore_index=True)
    bad = int(allp["date"].isna().sum())
    if bad:
        raise ValueError(f"{bad} rows have unparseable dates")
    before = len(allp)
    allp = (
        allp.sort_values(["ticker", "date", "source_file"])
        .drop_duplicates(["ticker", "date"], keep="first")
        .reset_index(drop=True)
    )
    result = allp[["date", "ticker", "company", "close"]].copy()
    result.attrs["non_eq_dropped"] = non_eq_dropped
    result.attrs["duplicates_dropped"] = before - len(allp)
    return result


if __name__ == "__main__":
    p = load_prices(sys.argv[1])
    print(p.groupby("ticker").agg(n=("close", "size"), first=("date", "min"), last=("date", "max")))
    print("total rows:", len(p))
