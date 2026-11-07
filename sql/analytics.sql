-- Named analytical queries. run via src/build_db.py:run_query(conn, "<name>").

-- name: stock_stats
WITH s AS (
    SELECT ticker,
           COUNT(*)                      AS n,
           AVG(log_ret)                  AS mu,
           AVG(log_ret * log_ret)        AS m2
    FROM daily_returns
    WHERE log_ret IS NOT NULL AND synthetic = 0
    GROUP BY ticker
),
span AS (
    SELECT ticker, MIN(date) AS d0, MAX(date) AS d1 FROM prices GROUP BY ticker
),
ends AS (
    SELECT p.ticker,
           MAX(CASE WHEN p.date = span.d0 THEN p.close_adj END) AS p0,
           MAX(CASE WHEN p.date = span.d1 THEN p.close_adj END) AS p1,
           span.d0, span.d1
    FROM prices p JOIN span ON span.ticker = p.ticker
    GROUP BY p.ticker
),
dd AS (
    SELECT ticker, MIN(drawdown) AS max_drawdown FROM drawdowns GROUP BY ticker
)
SELECT
    s.ticker,
    s.n                                                           AS return_obs,
    e.p1 / e.p0 - 1.0                                             AS total_return,
    EXP(LN(e.p1 / e.p0) / ((julianday(e.d1) - julianday(e.d0)) / 365.25)) - 1.0 AS cagr,
    SQRT(MAX(0.0, s.m2 - s.mu * s.mu) * s.n / (s.n - 1.0)) * SQRT(252.0)        AS ann_vol,
    dd.max_drawdown
FROM s
JOIN ends e ON e.ticker = s.ticker
JOIN dd     ON dd.ticker = s.ticker
ORDER BY s.ticker;

-- name: corr_pairs
-- All unique stock pairs (n * (n - 1) / 2 = 105 for 15 stocks), Pearson correlation of log returns.
SELECT
    a.ticker AS ticker_a,
    b.ticker AS ticker_b,
    COUNT(*) AS n,
    (AVG(a.log_ret * b.log_ret) - AVG(a.log_ret) * AVG(b.log_ret)) /
    SQRT((AVG(a.log_ret * a.log_ret) - AVG(a.log_ret) * AVG(a.log_ret)) *
         (AVG(b.log_ret * b.log_ret) - AVG(b.log_ret) * AVG(b.log_ret))) AS corr
FROM daily_returns a
JOIN daily_returns b ON a.date = b.date AND a.ticker < b.ticker
WHERE a.log_ret IS NOT NULL AND b.log_ret IS NOT NULL
  AND a.synthetic = 0 AND b.synthetic = 0
GROUP BY a.ticker, b.ticker
ORDER BY a.ticker, b.ticker;

-- name: returns_matrix
-- Long format of simple returns used by the backtest (synthetic flag kept).
SELECT date, ticker, ret, synthetic FROM daily_returns WHERE ret IS NOT NULL ORDER BY date, ticker;

-- name: adjusted_prices
SELECT date, ticker, close_adj FROM prices ORDER BY date, ticker;

-- name: rolling_vol
SELECT date, ticker, vol_60d FROM rolling_vol_60d ORDER BY date, ticker;
