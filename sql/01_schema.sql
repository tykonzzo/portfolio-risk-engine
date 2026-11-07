-- 01_schema.sql : base tables. Loaded from CSV by riskengine/sql_pipeline.py
DROP TABLE IF EXISTS universe;
DROP TABLE IF EXISTS raw_prices;
DROP TABLE IF EXISTS corporate_actions;

CREATE TABLE universe (
    ticker  TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    sector  TEXT NOT NULL
);

-- One row per stock per trading day (NSE regular "EQ" series only).
CREATE TABLE raw_prices (
    ticker     TEXT NOT NULL REFERENCES universe (ticker),
    trade_date TEXT NOT NULL,            -- ISO yyyy-mm-dd
    close      REAL NOT NULL CHECK (close > 0),
    vwap       REAL,
    volume     REAL,
    PRIMARY KEY (ticker, trade_date)
);

-- Splits / bonuses. price_factor multiplies every close BEFORE ex_date.
CREATE TABLE corporate_actions (
    ticker       TEXT NOT NULL REFERENCES universe (ticker),
    ex_date      TEXT NOT NULL,
    action       TEXT NOT NULL,
    price_factor REAL NOT NULL CHECK (price_factor > 0),
    note         TEXT
);

CREATE INDEX idx_raw_prices_date ON raw_prices (trade_date);
