CREATE TABLE Tickers (
    ticker_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100),
    sector VARCHAR(50)
);

CREATE TABLE Signals (
    signal_id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES Tickers(ticker_id),
    timestamp TIMESTAMP NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    strength FLOAT,
    price FLOAT
);

CREATE TABLE Models (
    model_id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES Tickers(ticker_id),
    model_type VARCHAR(50) NOT NULL,
    parameters JSONB,
    last_updated TIMESTAMP
);

CREATE TABLE Performance (
    performance_id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES Tickers(ticker_id),
    timestamp TIMESTAMP NOT NULL,
    profit_loss FLOAT,
    trade_count INTEGER
);

CREATE TABLE Forecasts (
    forecast_id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES Tickers(ticker_id),
    timestamp TIMESTAMP NOT NULL,
    forecast_value FLOAT,
    confidence FLOAT
);