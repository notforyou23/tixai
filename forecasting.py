from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import os

class ARIMAForecaster:
    def __init__(self, ticker_id):
        self.ticker_id = ticker_id
        self.model = None

    def train(self, data):
        self.model = ARIMA(data['Close'], order=(1,1,1))
        self.model_fit = self.model.fit()

    def forecast(self, steps):
        return self.model_fit.forecast(steps=steps)

class LSTMForecaster:
    def __init__(self, ticker_id, lookback=60):
        self.ticker_id = ticker_id
        self.lookback = lookback
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.lookback, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback), 0])
            y.append(data[i + self.lookback, 0])
        return np.array(X), np.array(y)

    def train(self, data):
        X, y = self.prepare_data(data['Close'].values.reshape(-1, 1))
        self.model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    def forecast(self, data, steps):
        X = data[-self.lookback:].reshape(1, self.lookback, 1)
        predictions = []
        for _ in range(steps):
            prediction = self.model.predict(X)
            predictions.append(prediction[0, 0])
            X = np.roll(X, -1, axis=1)
            X[0, -1, 0] = prediction
        return np.array(predictions)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            print(f"No saved model found at {path}")