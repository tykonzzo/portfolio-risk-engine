"""Verification suite. Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from riskengine import backtest as bt  # noqa: E402
from riskengine.ingest import load_raw  # noqa: E402
from riskengine.sql_pipeline import build_database  # noqa: E402


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    universe = pd.read_csv(ROOT / "data/universe.csv")
    actions = pd.read_csv(ROOT / "data/corporate_actions.csv")
    prices, stats = load_raw(ROOT / "data/raw", universe)
    db = tmp_path_factory.mktemp("db") / "t.db"
    conn = build_database(db, universe, prices, actions)
    R, gap, px = bt.load_panel(conn)
    return {"conn": conn, "R": R, "gap": gap, "px": px, "stats": stats,
            "actions": actions, "universe": universe}


def q(env, sql):
    return pd.read_sql(sql, env["conn"])


# ------------------------------------------------------------------ data layer
def test_universe_and_row_counts(env):
    s = env["stats"]
    assert s["stocks"] == 15
    assert s["rows_dropped_non_regular_series"] == 2      # two Dr. Lal block-deal rows
    assert s["rows_be_series"] == 62                       # Tata Investment trade-for-trade days
    assert s["rows_final"] == 15 * 1239                    # every stock has every trading day
    assert set(s["rows_by_ticker"].values()) == {1239}


def test_no_duplicate_stock_days(env):
    d = q(env, "SELECT ticker, trade_date, COUNT(*) c FROM raw_prices GROUP BY 1,2 HAVING c > 1")
    assert d.empty


def test_corporate_actions_remove_artificial_crashes(env):
    raw = q(env, "SELECT ticker, trade_date, raw_close, adj_close FROM adj_prices "
                 "ORDER BY ticker, trade_date")
    for a in env["actions"].itertuples():
        t = raw[raw.ticker == a.ticker].reset_index(drop=True)
        i = t.index[t.trade_date >= a.ex_date][0]
        raw_move = t.raw_close[i] / t.raw_close[i - 1] - 1
        adj_move = t.adj_close[i] / t.adj_close[i - 1] - 1
        assert raw_move < -0.30, f"{a.ticker}: expected a raw crash"
        assert abs(adj_move) < 0.08, f"{a.ticker}: adjusted move {adj_move:.3f} still too large"


def test_latest_prices_are_unadjusted(env):
    d = q(env, "SELECT COUNT(*) c FROM adj_prices WHERE trade_date = '2026-03-30' "
               "AND ABS(adj_close / raw_close - 1) > 1e-12")
    assert d.c[0] == 0


def test_no_unexplained_extreme_returns(env):
    r = q(env, "SELECT ticker, trade_date, ret FROM daily_returns WHERE is_gap = 0")
    assert r.ret.abs().max() < 0.25


def test_complete_calendar_after_supplement(env):
    assert env["gap"].sum() == 0
    assert len(env["px"]) == 1239
    assert q(env, "SELECT COUNT(*) c FROM daily_returns WHERE is_gap = 1")["c"][0] == 0


def test_recovered_be_series_prices_are_consistent(env):
    """NSE caps BE-series stocks at +/-5% a day, so every recovered return must sit inside that band,
    and the series must join the surrounding EQ prices without a jump."""
    r = q(env, "SELECT trade_date, ret FROM daily_returns WHERE ticker = 'TATAINVEST' "
               "AND trade_date BETWEEN '2024-02-29' AND '2024-05-31'")
    assert len(r) == 62
    assert r.ret.abs().max() <= 0.0501
    edge = q(env, "SELECT trade_date, ret FROM daily_returns WHERE ticker = 'TATAINVEST' "
                  "AND trade_date IN ('2024-02-29', '2024-06-03')")
    assert edge.ret.abs().max() < 0.06                      # NSE-recorded closes chain across both edges


def test_gap_handling_still_works_if_an_export_has_a_hole(env, tmp_path):
    """The supplement fixes today's data; the engine must still cope with a future hole."""
    raw = q(env, "SELECT * FROM raw_prices")
    holed = raw[~((raw.ticker == "TATAINVEST") & raw.trade_date.between("2024-02-29", "2024-05-31"))]
    conn = build_database(tmp_path / "gap.db", env["universe"], holed, env["actions"])
    R, gap, px = bt.load_panel(conn)
    assert gap.sum() == 1 and len(px) == 1239 - 62
    navs, _, _ = bt.run_backtest(R, gap, px, bt.Config())
    m = bt.performance(navs["equal_weight"], gap, 0.06)
    assert np.isfinite(m["sharpe"]) and m["ann_vol"] > 0


# ------------------------------------------------- SQL results vs independent pandas
def test_sql_correlation_matches_pandas(env):
    pairs = q(env, "SELECT * FROM correlation_pairs")
    assert len(pairs) == 105
    R = env["R"][~env["gap"].values].iloc[1:]
    pc = R.corr()
    diff = max(abs(pc.loc[a, b] - c) for a, b, c in
               pairs[["ticker_a", "ticker_b", "corr"]].itertuples(index=False))
    assert diff < 1e-9
    assert pairs["corr"].between(-1, 1).all()


def test_sql_volatility_matches_pandas(env):
    s = q(env, "SELECT ticker, ann_vol, ann_mean_return FROM stock_stats").set_index("ticker")
    R = env["R"][~env["gap"].values].iloc[1:]
    pd.testing.assert_series_equal(s.ann_vol.sort_index(), R.std(ddof=1).sort_index() * np.sqrt(252),
                                   check_names=False, atol=1e-10)
    pd.testing.assert_series_equal(s.ann_mean_return.sort_index(), R.mean().sort_index() * 252,
                                   check_names=False, atol=1e-10)


def test_sql_rolling_vol_matches_pandas(env):
    rv = q(env, "SELECT trade_date, vol_63d FROM rolling_vol_63d WHERE ticker = 'SBIN' "
                "ORDER BY trade_date").set_index("trade_date")["vol_63d"]
    x = env["R"]["SBIN"].where(~env["gap"]).iloc[1:]
    chk = x.rolling(63, min_periods=60).std() * np.sqrt(252)
    chk.index = chk.index.strftime("%Y-%m-%d")
    both = pd.concat([rv, chk], axis=1, join="inner").dropna()
    assert len(both) > 1000
    assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() < 1e-9


def test_sql_returns_match_adjusted_prices(env):
    px = env["px"]
    pd.testing.assert_frame_equal(env["R"].iloc[1:], px.pct_change().iloc[1:], check_names=False,
                                  atol=1e-12, check_freq=False)


# ------------------------------------------------------------------ portfolio engine
def test_min_variance_constraints_and_optimality():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(15, 15))
    cov = A @ A.T / 15 + np.eye(15) * 0.05
    w = bt.min_variance_weights(cov, max_w=0.25)
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= -1e-12 and w.max() <= 0.25 + 1e-9
    assert w @ cov @ w <= bt.equal_weights(15) @ cov @ bt.equal_weights(15)
    # no random feasible portfolio beats it
    for _ in range(500):
        x = rng.dirichlet(np.ones(15))
        if x.max() <= 0.25:
            assert w @ cov @ w <= x @ cov @ x + 1e-9


def test_momentum_tilt_weights():
    sig = np.arange(15, dtype=float)
    w = bt.momentum_tilt_weights(sig)
    assert abs(w.sum() - 1) < 1e-12
    assert w[-1] > w[7] > w[0] and np.isclose(w[-1] / w[0], 3.0)


def test_backtest_has_no_lookahead(env):
    """Weights chosen up to date T must not change if every return after T is scrambled."""
    R, gap, px = env["R"], env["gap"], env["px"]
    cfg = bt.Config()
    _, w_base, _ = bt.run_backtest(R, gap, px, cfg)
    T = 700
    rng = np.random.default_rng(1)
    R2 = R.copy()
    R2.iloc[T + 1:] = rng.normal(0, 0.03, size=R2.iloc[T + 1:].shape)
    px2 = (1 + R2.fillna(0)).cumprod()
    px2.iloc[:T + 1] = px.iloc[:T + 1]
    px2.iloc[T + 1:] = px.iloc[T] .values * (1 + R2.iloc[T + 1:]).cumprod().values
    _, w_alt, _ = bt.run_backtest(R2, gap, px2, cfg)
    cutoff = R.index[T]
    for s in ("min_variance", "momentum_tilt"):
        a = w_base[s][w_base[s].index <= cutoff]
        b = w_alt[s][w_alt[s].index <= cutoff]
        assert len(a) > 5
        np.testing.assert_allclose(a.values, b.values, atol=1e-9)


def test_backtest_outputs_are_sane(env):
    cfg = bt.Config()
    navs, weights, turns = bt.run_backtest(env["R"], env["gap"], env["px"], cfg)
    assert navs.notna().all().all()
    for s in bt.STRATEGIES:
        w = weights[s]
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)
        assert (w >= -1e-12).all().all()
        assert turns[s].iloc[0] == pytest.approx(1.0)       # first trade buys the whole book
    assert (weights["min_variance"].max(axis=1) <= cfg.max_weight + 1e-6).all()
    m = bt.performance(navs["equal_weight"], env["gap"], cfg.rf)
    assert -1 < m["max_drawdown"] < 0 and m["ann_vol"] > 0


def test_equal_weight_nav_matches_manual_calc(env):
    """Independent check: monthly-rebalanced 1/N with zero cost equals a hand-rolled loop."""
    R, gap, px = env["R"], env["gap"], env["px"]
    cfg = bt.Config(cost_bps=0.0)
    navs, _, _ = bt.run_backtest(R, gap, px, cfg)
    reb = set(bt.month_end_indices(R.index, cfg.lookback))
    start = min(reb)
    nav, w = 1.0, np.full(15, 1 / 15)
    for k in range(start + 1, len(R)):
        r = R.iloc[k].values
        pr = w @ r
        nav *= 1 + pr
        w = w * (1 + r) / (1 + pr)
        if k in reb:
            w = np.full(15, 1 / 15)
    assert navs["equal_weight"].iloc[-1] == pytest.approx(nav, rel=1e-12)


# ------------------------------------------------------------------ Nifty 50 benchmark
OFFICIAL_NIFTY_CLOSES = {      # published NSE closing values
    "2021-12-31": 17354.05, "2022-12-30": 18105.30, "2023-12-29": 21731.40,
    "2024-06-03": 23263.90, "2024-06-04": 21884.50, "2024-12-31": 23644.80,
}


def test_benchmark_matches_official_nifty_closes():
    b = pd.read_csv(ROOT / "data/benchmark/NIFTY50_daily.csv")
    b["d"] = pd.to_datetime(b["Date"], format="%d-%b-%Y").dt.strftime("%Y-%m-%d")
    b = b.set_index("d")
    for day, official in OFFICIAL_NIFTY_CLOSES.items():
        assert abs(b.loc[day, "Close"] - official) < 3.0, (day, b.loc[day, "Close"], official)


def test_benchmark_shares_the_stock_calendar(env):
    b = pd.read_csv(ROOT / "data/benchmark/NIFTY50_daily.csv")
    got = pd.to_datetime(b["Date"], format="%d-%b-%Y")
    assert list(got) == list(env["px"].index)
    assert (b["SessionBars"] == 0).sum() == 1              # only the 2022 Diwali Muhurat evening


def test_benchmark_metrics_are_sensible(env):
    bench = bt.load_benchmark(ROOT / "data/benchmark")
    cfg = bt.Config()
    navs, _, turns = bt.run_backtest(env["R"], env["gap"], env["px"], cfg)
    b = bench.reindex(navs.index).ffill()
    b = b / b.iloc[0]
    m = bt.performance(navs["equal_weight"], env["gap"], cfg.rf, turns["equal_weight"], b)
    assert 0.6 < m["beta"] < 1.3 and 0.5 < m["correlation_to_benchmark"] < 1.0


def test_dashboard_loads_plotly_from_a_host_that_serves_it():
    """cdnjs lists plotly.js 2.35.2 with zero files (404), which left the live charts blank."""
    from riskengine import static_dashboard as sd
    html = sd.fragment((ROOT / "outputs/results.json").read_text())
    assert sd.PLOTLY_URL.startswith("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/")
    assert f'<script src="{sd.PLOTLY_URL}"></script>' in html
    assert "cdnjs.cloudflare.com/ajax/libs/plotly.js" not in html


def test_github_pages_page_serves_its_own_plotly_and_reports_failures():
    """docs/index.html must not depend on a third-party CDN, and must show errors on the page."""
    page = (ROOT / "docs/index.html").read_text()
    local = ROOT / "docs/vendor/plotly-2.35.2.min.js"
    assert '<script src="vendor/plotly-2.35.2.min.js"></script>' in page
    assert local.exists() and local.stat().st_size > 3_000_000
    assert local.read_text(errors="ignore").lstrip().startswith("/**\n* plotly.js v2.35.2")
    assert 'id="status"' in page and "Dashboard build 20" in page
