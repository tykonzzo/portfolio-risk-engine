-- Daily simple and log returns via window functions.
-- `synthetic` marks returns that fall inside a filled data gap; risk statistics
-- (volatility, correlation) exclude them so filled prices cannot understate risk.

DROP VIEW IF EXISTS daily_returns;
CREATE VIEW daily_returns AS
SELECT
    date,
    ticker,
    close_adj,
    close_adj / LAG(close_adj) OVER w - 1.0      AS ret,
    LN(close_adj / LAG(close_adj) OVER w)        AS log_ret,
    CASE WHEN is_interpolated = 1
           OR COALESCE(LAG(is_interpolated) OVER w, 0) = 1
         THEN 1 ELSE 0 END                       AS synthetic
FROM prices
WINDOW w AS (PARTITION BY ticker ORDER BY date);

-- 60-trading-day rolling annualised volatility (sample standard deviation of log returns).
DROP VIEW IF EXISTS rolling_vol_60d;
CREATE VIEW rolling_vol_60d AS
SELECT date, ticker, vol_60d
FROM (
    SELECT
        date,
        ticker,
        COUNT(*) OVER w AS n,
        SQRT(MAX(0.0, AVG(log_ret * log_ret) OVER w - AVG(log_ret) OVER w * AVG(log_ret) OVER w)
             * 60.0 / 59.0) * SQRT(252.0)  AS vol_60d
    FROM daily_returns
    WHERE log_ret IS NOT NULL AND synthetic = 0
    WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
)
WHERE n = 60;

-- Running drawdown from the prior peak, per stock.
DROP VIEW IF EXISTS drawdowns;
CREATE VIEW drawdowns AS
SELECT
    date,
    ticker,
    close_adj / MAX(close_adj) OVER (PARTITION BY ticker ORDER BY date) - 1.0 AS drawdown
FROM prices;
