"""Walk-forward portfolio construction and performance metrics.

Three strategies, all long-only and fully invested, rebalanced at every month-end close
using ONLY data available at that close (no look-ahead):

  equal_weight   1/N
  min_variance   minimise w' S w on the trailing `lookback` days, weight cap `max_weight`
  momentum_tilt  equal-weight x 1.5 / 1.0 / 0.5 for the top / middle / bottom third of
                 stocks ranked on trailing 12-month return, skipping the latest month
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
STRATEGIES = ["equal_weight", "min_variance", "momentum_tilt"]
LABELS = {
    "equal_weight": "Equal-weight",
    "min_variance": "Minimum-variance",
    "momentum_tilt": "Momentum-tilted",
    "benchmark": "Nifty 50",
}


@dataclass
class Config:
    lookback: int = 252        # trading days of history used at each rebalance
    mom_skip: int = 21         # most recent days skipped in the momentum signal
    max_weight: float = 0.25   # per-stock cap for min-variance
    cost_bps: float = 10.0     # one-way transaction cost on traded notional
    rf: float = 0.06           # annual risk-free rate used in Sharpe

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- data access
def load_panel(conn) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (returns, gap_flag, prices) on the aligned panel calendar."""
    px = pd.read_sql(
        "SELECT a.ticker, a.trade_date, a.adj_close FROM adj_prices a "
        "JOIN panel_dates d USING (trade_date)", conn, parse_dates=["trade_date"])
    px = px.pivot(index="trade_date", columns="ticker", values="adj_close").sort_index()
    rt = pd.read_sql("SELECT ticker, trade_date, ret, is_gap FROM daily_returns",
                     conn, parse_dates=["trade_date"])
    R = rt.pivot(index="trade_date", columns="ticker", values="ret").reindex(px.index)
    gap = (rt.groupby("trade_date")["is_gap"].max().reindex(px.index).fillna(0)
           .astype(bool))
    return R[px.columns], gap, px


def load_benchmark(bench_dir: Path) -> pd.Series | None:
    """Optional NSE index history (Date, Close). Returns None if no file is present."""
    files = sorted(Path(bench_dir).glob("*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[0], encoding="utf-8-sig", thousands=",")
    df.columns = [" ".join(c.split()).upper() for c in df.columns]
    date_col = next(c for c in df.columns if "DATE" in c)
    close_col = next(c for c in df.columns if c in ("CLOSE", "CLOSE PRICE", "INDEX CLOSE"))
    d = pd.to_datetime(df[date_col], format="mixed", dayfirst=True, errors="coerce")
    s = pd.Series(pd.to_numeric(df[close_col], errors="coerce").values, index=d).dropna()
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated()].rename("benchmark")


# ------------------------------------------------------------ weight functions
def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance_weights(cov: np.ndarray, max_w: float | None = None) -> np.ndarray:
    n = cov.shape[0]
    hi = 1.0 if max_w is None else max_w
    res = minimize(
        lambda w: w @ cov @ w, np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * cov @ w,
        bounds=[(0.0, hi)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"min-variance optimisation failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def momentum_tilt_weights(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    rank = signal.argsort().argsort()            # 0 = weakest
    third = n / 3.0
    mult = np.where(rank >= n - third, 1.5, np.where(rank < third, 0.5, 1.0))
    w = mult / mult.sum()
    return w


# -------------------------------------------------------------- simulation
def month_end_indices(dates: pd.DatetimeIndex, first_allowed: int) -> list[int]:
    out = []
    for k in range(first_allowed, len(dates) - 1):
        if dates[k].month != dates[k + 1].month:
            out.append(k)
    return out


def simulate(weight_fn, R: pd.DataFrame, reb: list[int], cost: float):
    """Run one strategy. Returns (nav Series, weights DataFrame at rebalances, turnover Series)."""
    dates = R.index
    start = reb[0]
    reb_set = set(reb[1:])
    n = R.shape[1]
    w = weight_fn(start)
    nav = 1.0 - cost * np.abs(w).sum()           # buy the initial portfolio
    navs = {dates[start]: nav}
    weights = {dates[start]: w.copy()}
    turnover = {dates[start]: float(np.abs(w).sum())}
    Rv = R.values
    for k in range(start + 1, len(dates)):
        r = Rv[k]
        pr = float(w @ r)
        nav *= 1.0 + pr
        w = w * (1.0 + r) / (1.0 + pr)           # weights drift with prices
        if k in reb_set:
            target = weight_fn(k)
            t = float(np.abs(target - w).sum())
            nav *= 1.0 - cost * t
            w = target
            weights[dates[k]] = w.copy()
            turnover[dates[k]] = t
        navs[dates[k]] = nav
    cols = list(R.columns)
    return (pd.Series(navs, name="nav"),
            pd.DataFrame(weights, index=cols).T,
            pd.Series(turnover, name="turnover"))


def run_backtest(R: pd.DataFrame, gap: pd.Series, px: pd.DataFrame, cfg: Config):
    cols = list(R.columns)
    n = len(cols)
    L = cfg.lookback
    reb = month_end_indices(R.index, first_allowed=L)
    ok = ~gap.values

    def w_equal(k):
        return equal_weights(n)

    def w_minvar(k):
        win = R.iloc[k - L + 1:k + 1]
        win = win[ok[k - L + 1:k + 1]]
        return min_variance_weights(win.cov().values * TRADING_DAYS, cfg.max_weight)

    def w_mom(k):
        sig = (px.iloc[k - cfg.mom_skip] / px.iloc[k - L] - 1.0).values
        return momentum_tilt_weights(sig)

    fns = {"equal_weight": w_equal, "min_variance": w_minvar, "momentum_tilt": w_mom}
    cost = cfg.cost_bps / 1e4
    navs, weights, turns = {}, {}, {}
    for name in STRATEGIES:
        navs[name], weights[name], turns[name] = simulate(fns[name], R, reb, cost)
    return pd.DataFrame(navs), weights, turns


# ----------------------------------------------------------------- metrics
def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def performance(nav: pd.Series, gap: pd.Series, rf: float, turnover: pd.Series | None = None,
                bench_nav: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25
    cagr = nav.iloc[-1] ** (1.0 / years) / nav.iloc[0] ** (1.0 / years) - 1.0
    ret = nav.pct_change().dropna()
    ret = ret[~gap.reindex(ret.index).fillna(False).values]   # drop the gap-spanning return
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    dd = drawdown_series(nav)
    out = {
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1.0,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": (cagr - rf) / vol,
        "max_drawdown": dd.min(),
        "max_drawdown_date": dd.idxmin().strftime("%Y-%m-%d"),
        "calmar": cagr / abs(dd.min()),
        "best_day": ret.max(),
        "worst_day": ret.min(),
        "pct_days_up": (ret > 0).mean(),
    }
    if turnover is not None:
        out["avg_annual_turnover"] = float(turnover.iloc[1:].sum() / 2.0 / (years or 1))
    if bench_nav is not None:
        b = bench_nav.reindex(nav.index).ffill()
        br = b.pct_change().dropna()
        pr = nav.pct_change().dropna()
        keep = ~gap.reindex(pr.index).fillna(False).values
        pr, br = pr[keep], br.reindex(pr.index)[keep]
        cov = np.cov(pr.values, br.values, ddof=1)
        beta = cov[0, 1] / cov[1, 1]
        te = (pr - br).std(ddof=1) * np.sqrt(TRADING_DAYS)
        bcagr = b.iloc[-1] ** (1.0 / years) / b.iloc[0] ** (1.0 / years) - 1.0
        out.update({"beta": beta, "correlation_to_benchmark": float(np.corrcoef(pr, br)[0, 1]),
                    "tracking_error": te,
                    "information_ratio": (cagr - bcagr) / te if te > 0 else np.nan})
    return out


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, points: int = 30) -> pd.DataFrame:
    """Long-only minimum-variance frontier from the min-variance point to the best single asset."""
    n = len(mu)
    w0 = min_variance_weights(cov)
    lo, hi = float(w0 @ mu), float(mu.max())
    rows = []
    for target in np.linspace(lo, hi, points):
        res = minimize(
            lambda w: w @ cov @ w, w0, jac=lambda w: 2.0 * cov @ w,
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                         {"type": "eq", "fun": lambda w, t=target: w @ mu - t}],
            method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        if res.success:
            rows.append({"ret": float(res.x @ mu), "vol": float(np.sqrt(res.x @ cov @ res.x))})
    return pd.DataFrame(rows)
