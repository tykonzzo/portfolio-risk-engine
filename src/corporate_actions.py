"""Split / bonus adjustments for the raw (unadjusted) NSE closing prices.

The downloaded NSE files are NOT adjusted for corporate actions: on the ex-date the
close simply drops (e.g. HDFC Bank roughly halves). Left alone, each drop would be
read as a -50% to -90% daily return and wreck volatility, drawdown and Sharpe.

The factors below were inferred from the data itself: each is the clean ratio nearest
to the observed one-day price drop. They are NOT taken from an official corporate-action
feed, so confirm them against NSE announcements before relying on exact figures.
`validate_actions` re-checks every factor against the data on each run.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Action:
    ticker: str
    ex_date: str
    factor: float  # multiplier applied to every close BEFORE ex_date


ACTIONS = [
    Action("IOC", "2022-06-30", 2 / 3),
    Action("SHRIRAMFIN", "2025-01-10", 0.2),
    Action("HDFCBANK", "2025-08-26", 0.5),
    Action("LALPATHLAB", "2025-12-19", 0.5),
    Action("TATAINVEST", "2025-10-14", 0.1),
    Action("KOTAKBANK", "2026-01-14", 0.2),
]

TOLERANCE = 0.15  # observed ratio may differ from the clean ratio by market move on the day


def observed_ratio(prices: pd.DataFrame, action: Action) -> float:
    s = prices[prices["ticker"] == action.ticker].sort_values("date").set_index("date")["close"]
    ex = pd.Timestamp(action.ex_date)
    if ex not in s.index:
        raise ValueError(f"{action.ticker}: ex-date {action.ex_date} not in data")
    prev = s[s.index < ex].iloc[-1]
    return float(s.loc[ex] / prev)


def validate_actions(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ACTIONS:
        obs = observed_ratio(prices, a)
        if abs(obs / a.factor - 1.0) > TOLERANCE:
            raise ValueError(
                f"{a.ticker} {a.ex_date}: observed ratio {obs:.4f} is not within "
                f"{TOLERANCE:.0%} of factor {a.factor:.4f}"
            )
        rows.append({"ticker": a.ticker, "ex_date": a.ex_date, "factor": a.factor, "observed": obs})
    return pd.DataFrame(rows)


def apply_actions(prices: pd.DataFrame) -> pd.DataFrame:
    """Add adj_factor and close_adj columns."""
    out = prices.copy()
    out["adj_factor"] = 1.0
    for a in ACTIONS:
        mask = (out["ticker"] == a.ticker) & (out["date"] < pd.Timestamp(a.ex_date))
        out.loc[mask, "adj_factor"] *= a.factor
    out["close_adj"] = out["close"] * out["adj_factor"]
    return out
