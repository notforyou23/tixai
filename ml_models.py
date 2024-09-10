from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os

class TickerModel:
    def __init__(self, ticker_id):
        self.ticker_id = ticker_id
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_importance = None

    def prepare_data(self, data):
        # Implement feature engineering here
        features = [
            'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
            'ADX', 'OBV', 'MOM', 'Stochastic', 'VWAP', 'SAR'
        ]
        X = data[features]
        y = data['signal']  # Assuming 'signal' column exists in data
        return X, y

    def train(self, data):
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.feature_importance = self.model.feature_importances_
        
        return accuracy

    def predict(self, features):
        return self.model.predict_proba(features)

    def get_feature_importance(self):
        return self.feature_importance

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            print(f"No saved model found at {path}")