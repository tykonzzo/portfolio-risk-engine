-- Core tables for the Portfolio Construction & Risk-Return Analysis Engine.
-- Prices are stored once, split/bonus-adjusted, one row per ticker per trading day.

DROP TABLE IF EXISTS prices;
CREATE TABLE prices (
    date            TEXT    NOT NULL,   -- ISO yyyy-mm-dd
    ticker          TEXT    NOT NULL,
    company         TEXT    NOT NULL,
    close_raw       REAL,               -- NULL where the price was filled (see is_interpolated)
    adj_factor      REAL    NOT NULL,   -- cumulative split/bonus multiplier
    close_adj       REAL    NOT NULL,   -- close_raw * adj_factor (or interpolated)
    is_interpolated INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, ticker)
);
CREATE INDEX idx_prices_ticker_date ON prices (ticker, date);

DROP TABLE IF EXISTS corporate_actions;
CREATE TABLE corporate_actions (
    ticker      TEXT NOT NULL,
    ex_date     TEXT NOT NULL,
    factor      REAL NOT NULL,          -- multiplier applied to all closes before ex_date
    observed    REAL NOT NULL,          -- close(ex_date) / close(previous day), raw
    PRIMARY KEY (ticker, ex_date)
);
