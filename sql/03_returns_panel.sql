-- 03_returns_panel.sql : aligned panel + daily returns (window functions).
--
-- panel_dates : trading days on which ALL stocks have a price. Keeping every stock on
--               the same calendar makes returns, covariances and portfolios comparable.
-- daily_returns: simple and log return per stock per panel day using LAG().
--               is_gap = 1 when the return spans more than 10 calendar days (a hole in
--               the data, not a real daily move). Gap rows are kept for wealth/NAV
--               (the price path stays continuous) but excluded from risk statistics.
DROP TABLE IF EXISTS panel_dates;
DROP TABLE IF EXISTS daily_returns;

CREATE TABLE panel_dates AS
SELECT trade_date
  FROM adj_prices
 GROUP BY trade_date
HAVING COUNT(*) = (SELECT COUNT(*) FROM universe);

CREATE TABLE daily_returns AS
WITH base AS (
    SELECT
        a.ticker,
        a.trade_date,
        a.adj_close,
        LAG(a.adj_close) OVER w AS prev_close,
        LAG(a.trade_date) OVER w AS prev_date
    FROM adj_prices a
    JOIN panel_dates d USING (trade_date)
    WINDOW w AS (PARTITION BY a.ticker ORDER BY a.trade_date)
)
SELECT
    ticker,
    trade_date,
    adj_close,
    adj_close / prev_close - 1.0                                   AS ret,
    LN(adj_close / prev_close)                                     AS log_ret,
    CAST(julianday(trade_date) - julianday(prev_date) AS INTEGER)  AS gap_days,
    CASE WHEN julianday(trade_date) - julianday(prev_date) > 10
         THEN 1 ELSE 0 END                                         AS is_gap
FROM base
WHERE prev_close IS NOT NULL;

CREATE UNIQUE INDEX idx_daily_returns ON daily_returns (ticker, trade_date);
CREATE INDEX idx_daily_returns_date ON daily_returns (trade_date);
