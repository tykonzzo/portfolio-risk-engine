-- 04_risk_stats.sql : volatility, drawdown, correlation - all in SQL.
-- SQLite has no STDDEV or CORR, so both are built from running sums:
--   var  = (sum(x^2) - sum(x)^2 / n) / (n - 1)
--   corr = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x^2) - sum(x)^2) * (n*sum(y^2) - sum(y)^2))
DROP TABLE IF EXISTS rolling_vol_63d;
DROP TABLE IF EXISTS stock_drawdown;
DROP TABLE IF EXISTS stock_stats;
DROP TABLE IF EXISTS correlation_pairs;

-- 63-day rolling annualised volatility (gap returns excluded from the window)
CREATE TABLE rolling_vol_63d AS
WITH x AS (
    SELECT ticker, trade_date, CASE WHEN is_gap = 0 THEN ret END AS r
      FROM daily_returns
),
w AS (
    SELECT ticker, trade_date,
           COUNT(r)   OVER win AS n,
           SUM(r)     OVER win AS s1,
           SUM(r * r) OVER win AS s2
      FROM x
    WINDOW win AS (PARTITION BY ticker ORDER BY trade_date
                   ROWS BETWEEN 62 PRECEDING AND CURRENT ROW)
)
SELECT ticker, trade_date,
       SQRT((s2 - s1 * s1 / n) / (n - 1)) * SQRT(252.0) AS vol_63d
  FROM w
 WHERE n >= 60;

-- Running-peak drawdown on every stock's own full history
CREATE TABLE stock_drawdown AS
WITH x AS (
    SELECT ticker, trade_date, adj_close,
           MAX(adj_close) OVER (PARTITION BY ticker ORDER BY trade_date
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peak
      FROM adj_prices
)
SELECT ticker, trade_date, adj_close / peak - 1.0 AS drawdown FROM x;

-- Per-stock summary statistics
CREATE TABLE stock_stats AS
WITH agg AS (
    SELECT ticker,
           COUNT(*)       AS n_obs,
           AVG(ret)       AS mean_daily,
           SUM(ret)       AS s1,
           SUM(ret * ret) AS s2
      FROM daily_returns
     WHERE is_gap = 0
     GROUP BY ticker
),
ends AS (
    SELECT ticker, MIN(trade_date) AS d0, MAX(trade_date) AS d1
      FROM adj_prices GROUP BY ticker
),
dd AS (
    SELECT ticker, MIN(drawdown) AS max_drawdown FROM stock_drawdown GROUP BY ticker
)
SELECT
    u.ticker, u.company, u.sector,
    agg.n_obs,
    agg.mean_daily * 252.0                                             AS ann_mean_return,
    SQRT((agg.s2 - agg.s1 * agg.s1 / agg.n_obs) / (agg.n_obs - 1))
        * SQRT(252.0)                                                  AS ann_vol,
    p1.adj_close / p0.adj_close - 1.0                                  AS total_return,
    POWER(p1.adj_close / p0.adj_close,
          365.25 / (julianday(ends.d1) - julianday(ends.d0))) - 1.0    AS cagr,
    dd.max_drawdown
FROM universe u
JOIN agg  ON agg.ticker  = u.ticker
JOIN ends ON ends.ticker = u.ticker
JOIN adj_prices p0 ON p0.ticker = u.ticker AND p0.trade_date = ends.d0
JOIN adj_prices p1 ON p1.ticker = u.ticker AND p1.trade_date = ends.d1
JOIN dd   ON dd.ticker   = u.ticker;

-- Pairwise correlation: 15 stocks -> 15*14/2 = 105 unique pairs
CREATE TABLE correlation_pairs AS
SELECT
    a.ticker AS ticker_a,
    b.ticker AS ticker_b,
    COUNT(*) AS n_obs,
    (COUNT(*) * SUM(a.ret * b.ret) - SUM(a.ret) * SUM(b.ret)) /
    SQRT((COUNT(*) * SUM(a.ret * a.ret) - SUM(a.ret) * SUM(a.ret)) *
         (COUNT(*) * SUM(b.ret * b.ret) - SUM(b.ret) * SUM(b.ret))) AS corr
  FROM daily_returns a
  JOIN daily_returns b
    ON a.trade_date = b.trade_date AND a.ticker < b.ticker
 WHERE a.is_gap = 0
 GROUP BY a.ticker, b.ticker;
