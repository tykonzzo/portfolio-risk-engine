"""End-to-end run: raw NSE files -> SQLite -> backtest -> outputs/ (CSV, JSON, charts, dashboard).

    python run_pipeline.py                      # defaults
    python run_pipeline.py --rf 0.065 --cost-bps 20 --max-weight 0.20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine import backtest as bt
from riskengine.ingest import load_raw
from riskengine.sql_pipeline import build_database

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def corr_matrix(pairs: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    for a, b, c in pairs[["ticker_a", "ticker_b", "corr"]].itertuples(index=False):
        m.loc[a, b] = m.loc[b, a] = c
    return m


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    d = bt.Config()
    ap.add_argument("--lookback", type=int, default=d.lookback)
    ap.add_argument("--max-weight", type=float, default=d.max_weight)
    ap.add_argument("--cost-bps", type=float, default=d.cost_bps)
    ap.add_argument("--rf", type=float, default=d.rf)
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    cfg = bt.Config(lookback=a.lookback, max_weight=a.max_weight, cost_bps=a.cost_bps, rf=a.rf)

    OUT.mkdir(exist_ok=True)
    universe = pd.read_csv(ROOT / "data/universe.csv")
    actions = pd.read_csv(ROOT / "data/corporate_actions.csv")
    prices, ingest_stats = load_raw(ROOT / "data/raw", universe)
    conn = build_database(OUT / "risk_engine.db", universe, prices, actions)

    # ---- SQL-computed statistics
    stock_stats = pd.read_sql("SELECT * FROM stock_stats ORDER BY ticker", conn)
    pairs = pd.read_sql("SELECT * FROM correlation_pairs ORDER BY ticker_a, ticker_b", conn)
    tickers = sorted(universe["ticker"])
    cmat = corr_matrix(pairs, tickers)
    rvol = pd.read_sql("SELECT * FROM rolling_vol_63d", conn, parse_dates=["trade_date"])

    # ---- backtest
    R, gap, px = bt.load_panel(conn)
    navs, weights, turns = bt.run_backtest(R, gap, px, cfg)
    bench = bt.load_benchmark(ROOT / "data/benchmark")
    bench_nav = None
    if bench is not None:
        b = bench.reindex(navs.index).ffill()
        bench_nav = b / b.iloc[0]
        navs["benchmark"] = bench_nav

    rows = {}
    for s in bt.STRATEGIES:
        rows[s] = bt.performance(navs[s], gap, cfg.rf, turns[s], bench_nav)
    if bench_nav is not None:
        rows["benchmark"] = bt.performance(bench_nav, gap, cfg.rf)
    metrics = pd.DataFrame(rows).T
    metrics.insert(0, "strategy", [bt.LABELS[i] for i in metrics.index])

    dd = navs.apply(bt.drawdown_series)

    # ---- in-sample efficient frontier (reference only: uses the full sample)
    ok = ~gap.values
    full = R.iloc[1:][ok[1:]]
    mu, cov = full.mean().values * bt.TRADING_DAYS, full.cov().values * bt.TRADING_DAYS
    frontier = bt.efficient_frontier(mu, cov)
    mv = bt.min_variance_weights(cov)
    insample_minvar = {"ret": float(mv @ mu), "vol": float(np.sqrt(mv @ cov @ mv)),
                       "weights": dict(zip(R.columns, mv.round(4)))}
    stock_stats["sharpe_arith"] = (stock_stats["ann_mean_return"] - cfg.rf) / stock_stats["ann_vol"]

    # ---- write CSV outputs
    navs.round(6).rename_axis("date").to_csv(OUT / "nav.csv")
    dd.round(6).rename_axis("date").to_csv(OUT / "drawdown.csv")
    metrics.rename_axis("key").to_csv(OUT / "metrics.csv", float_format="%.6f")
    for s in bt.STRATEGIES:
        weights[s].round(6).rename_axis("rebalance_date").to_csv(OUT / f"weights_{s}.csv")
    cmat.round(6).to_csv(OUT / "correlation_matrix.csv")
    pairs.round(6).to_csv(OUT / "correlation_pairs.csv", index=False)
    stock_stats.round(6).to_csv(OUT / "stock_stats.csv", index=False)
    frontier.round(6).to_csv(OUT / "efficient_frontier.csv", index=False)

    gap_rows = pd.read_sql(
        "SELECT trade_date, gap_days FROM daily_returns WHERE is_gap = 1 GROUP BY trade_date", conn)
    quality = {
        "ingest": ingest_stats,
        "panel_trading_days": int(len(px)),
        "panel_first": px.index[0].strftime("%Y-%m-%d"),
        "panel_last": px.index[-1].strftime("%Y-%m-%d"),
        "excluded_non_common_days": int(prices["trade_date"].nunique() - len(px)),
        "gap_return_dates": gap_rows.to_dict("records"),
        "corporate_actions": actions.to_dict("records"),
        "unique_correlation_pairs": int(len(pairs)),
        "benchmark_present": bench is not None,
        "config": cfg.as_dict(),
    }
    (OUT / "data_quality.json").write_text(json.dumps(quality, indent=2, default=str))

    results = {
        "meta": quality | {"labels": bt.LABELS},
        "tickers": tickers,
        "sectors": dict(zip(universe["ticker"], universe["sector"])),
        "companies": dict(zip(universe["ticker"], universe["company"])),
        "dates": [d.strftime("%Y-%m-%d") for d in navs.index],
        "nav": {c: navs[c].round(5).tolist() for c in navs.columns},
        "drawdown": {c: dd[c].round(5).tolist() for c in dd.columns},
        "metrics": json.loads(metrics.reset_index(names="key").to_json(orient="records")),
        "corr": {"tickers": tickers, "matrix": cmat.round(3).values.tolist()},
        "stocks": json.loads(stock_stats.round(5).to_json(orient="records")),
        "frontier": frontier.round(5).to_dict("records"),
        "insample_minvar": insample_minvar,
        "weights": {s: {"dates": [d.strftime("%Y-%m-%d") for d in weights[s].index],
                        "tickers": list(weights[s].columns),
                        "w": weights[s].round(4).values.tolist()} for s in bt.STRATEGIES},
        "vol_rolling": {"dates": [], "series": {}},
    }
    rv = rvol.pivot(index="trade_date", columns="ticker", values="vol_63d").iloc[::5]
    results["vol_rolling"] = {"dates": [d.strftime("%Y-%m-%d") for d in rv.index],
                              "series": {c: rv[c].round(4).where(rv[c].notna(), None).tolist()
                                         for c in rv.columns}}
    (OUT / "results.json").write_text(json.dumps(results, separators=(",", ":")))

    if not a.no_charts:
        from riskengine import charts, static_dashboard
        charts.make_all(results, ROOT / "docs/img")
        static_dashboard.build(OUT / "results.json", ROOT / "docs/index.html")
    conn.close()
    print(metrics.set_index("strategy")[["start", "end", "total_return", "cagr", "ann_vol",
                                         "sharpe", "max_drawdown"]].to_string(
        float_format=lambda x: f"{x:,.4f}"))
    return results


if __name__ == "__main__":
    main()
