import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

class Database:
    def __init__(self, connection_params):
        self.conn = psycopg2.connect(**connection_params)
        self.cur = self.conn.cursor()

    # Tickers table functions
    def insert_ticker(self, symbol, name, sector):
        query = """
        INSERT INTO Tickers (symbol, name, sector)
        VALUES (%s, %s, %s)
        RETURNING ticker_id
        """
        self.cur.execute(query, (symbol, name, sector))
        self.conn.commit()
        return self.cur.fetchone()[0]

    def get_ticker(self, symbol):
        query = "SELECT * FROM Tickers WHERE symbol = %s"
        self.cur.execute(query, (symbol,))
        return self.cur.fetchone()

    # Signals table functions
    def insert_signal(self, ticker_id, timestamp, signal_type, strength, price):
        query = """
        INSERT INTO Signals (ticker_id, timestamp, signal_type, strength, price)
        VALUES (%s, %s, %s, %s, %s)
        """
        self.cur.execute(query, (ticker_id, timestamp, signal_type, strength, price))
        self.conn.commit()

    def get_latest_signals(self, ticker_id, limit=100):
        query = """
        SELECT * FROM Signals
        WHERE ticker_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        self.cur.execute(query, (ticker_id, limit))
        return self.cur.fetchall()

    # Models table functions
    def insert_model(self, ticker_id, model_type, parameters):
        query = """
        INSERT INTO Models (ticker_id, model_type, parameters, last_updated)
        VALUES (%s, %s, %s, %s)
        """
        self.cur.execute(query, (ticker_id, model_type, parameters, datetime.now()))
        self.conn.commit()

    def update_model(self, model_id, parameters):
        query = """
        UPDATE Models
        SET parameters = %s, last_updated = %s
        WHERE model_id = %s
        """
        self.cur.execute(query, (parameters, datetime.now(), model_id))
        self.conn.commit()

    def get_latest_model(self, ticker_id, model_type):
        query = """
        SELECT * FROM Models
        WHERE ticker_id = %s AND model_type = %s
        ORDER BY last_updated DESC
        LIMIT 1
        """
        self.cur.execute(query, (ticker_id, model_type))
        return self.cur.fetchone()

    # Performance table functions
    def insert_performance(self, ticker_id, timestamp, profit_loss, trade_count):
        query = """
        INSERT INTO Performance (ticker_id, timestamp, profit_loss, trade_count)
        VALUES (%s, %s, %s, %s)
        """
        self.cur.execute(query, (ticker_id, timestamp, profit_loss, trade_count))
        self.conn.commit()

    def get_performance(self, ticker_id, start_date, end_date):
        query = """
        SELECT * FROM Performance
        WHERE ticker_id = %s AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp
        """
        self.cur.execute(query, (ticker_id, start_date, end_date))
        return self.cur.fetchall()

    # Forecasts table functions
    def insert_forecast(self, ticker_id, timestamp, forecast_value, confidence):
        query = """
        INSERT INTO Forecasts (ticker_id, timestamp, forecast_value, confidence)
        VALUES (%s, %s, %s, %s)
        """
        self.cur.execute(query, (ticker_id, timestamp, forecast_value, confidence))
        self.conn.commit()

    def get_latest_forecast(self, ticker_id):
        query = """
        SELECT * FROM Forecasts
        WHERE ticker_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """
        self.cur.execute(query, (ticker_id,))
        return self.cur.fetchone()

    def close(self):
        self.cur.close()
        self.conn.close()