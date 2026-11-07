-- 02_adjust_prices.sql : back-adjust closes for splits and bonuses.
-- adj_close = close x product of price_factor for every action with ex_date AFTER the trade date.
-- SQLite has no PRODUCT() aggregate, so the product is EXP(SUM(LN(factor))).
DROP TABLE IF EXISTS adj_prices;

CREATE TABLE adj_prices AS
SELECT
    p.ticker,
    p.trade_date,
    p.close AS raw_close,
    p.close * COALESCE(
        (SELECT EXP(SUM(LN(ca.price_factor)))
           FROM corporate_actions ca
          WHERE ca.ticker = p.ticker
            AND ca.ex_date > p.trade_date),
        1.0) AS adj_close,
    p.volume
FROM raw_prices p;

CREATE UNIQUE INDEX idx_adj_prices ON adj_prices (ticker, trade_date);
