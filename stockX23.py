import yfinance as yf
import pandas as pd
import numpy as np
import logging
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml_models import TickerModel
from forecasting import ARIMAForecaster, LSTMForecaster
from reinforcement_learning import RLAgent, TradingEnvironment
import talib
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduced verbosity for info logging
def log_info_highlight(message):
    logger.info(f"\033[1m{message}\033[0m")  # This will bold the message in most terminals.

@dataclass
class StockData:
    ticker: str
    data: pd.DataFrame
    dips: pd.DataFrame

class StockAnalyzer(ABC):
    @abstractmethod
    def fetch_data(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        pass

class YahooFinanceAnalyzer(StockAnalyzer):
    def __init__(self, short_window: int = 5, long_window: int = 7, threshold_multiplier: float = 1.5):
        self.short_window = short_window
        self.long_window = long_window
        self.threshold_multiplier = threshold_multiplier

    def fetch_data(self, ticker: str, period: str = '2y') -> Optional[pd.DataFrame]:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                logger.warning(f"No data found for {ticker}.")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def calculate_signals(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if data is None or data.empty:
            logger.warning("Data is empty, skipping signal calculation.")
            return None, None

        available_data_len = len(data)
        effective_short_window = min(self.short_window, available_data_len)
        effective_long_window = min(self.long_window, available_data_len)
        effective_atr_window = min(14, available_data_len)
        effective_recent_days = min(30, available_data_len)

        if available_data_len < max(effective_short_window, effective_long_window, effective_atr_window):
            logger.warning(f"Not enough data to calculate signals for {data.index[-1].strftime('%Y-%m-%d')}.")
            return None, None

        try:
            data = data.copy()
            data['5_day_ema'] = data['Close'].ewm(span=effective_short_window, adjust=False).mean()
            data['7_day_ema'] = data['Close'].ewm(span=effective_long_window, adjust=False).mean()
            data['percentage_drop'] = (data['Close'].shift(1) - data['Close']) / data['Close'].shift(1) * 100

            data['true_range'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            )
            data['ATR'] = data['true_range'].rolling(window=effective_atr_window).mean()

            data['dynamic_threshold'] = np.where(
                data.index >= data.index[-effective_recent_days],
                self.threshold_multiplier * 0.75 * data['ATR'],
                self.threshold_multiplier * data['ATR']
            )

            data['avg_volume'] = data['Volume'].rolling(window=effective_atr_window).mean()
            data['volume_signal'] = data['Volume'] > data['avg_volume']

            data['RSI'] = talib.RSI(data['Close'], timeperiod=effective_atr_window)
            data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

            data['signal'] = (
                (data['percentage_drop'] > data['dynamic_threshold']) &
                data['volume_signal'] &
                (data['RSI'] < 50) &
                (data['MACD'] < data['MACD_signal'])
            )

            signal_count = data['signal'].sum()
            log_info_highlight(f"Signal counts: {signal_count} for {data.index[-1].strftime('%Y-%m-%d')}")

            return data, data[data['signal']]
        except Exception as e:
            logger.error(f"Error calculating signals: {e}")
            return None, None


class EnhancedYahooFinanceAnalyzer(YahooFinanceAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def calculate_signals(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if data is None or data.empty:
            logger.warning("Data is empty, skipping signal calculation.")
            return None, None

        try:
            data = data.copy()
            
            # Calculate additional indicators
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
            
            data['5_day_ema'] = data['Close'].ewm(span=self.short_window, adjust=False).mean()
            data['7_day_ema'] = data['Close'].ewm(span=self.long_window, adjust=False).mean()
            data['percentage_drop'] = (data['Close'].shift(1) - data['Close']) / data['Close'].shift(1) * 100

            data['true_range'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            )
            data['ATR'] = data['true_range'].rolling(window=14).mean()

            data['dynamic_threshold'] = self.threshold_multiplier * data['ATR']

            data['avg_volume'] = data['Volume'].rolling(window=14).mean()
            data['volume_signal'] = data['Volume'] > data['avg_volume']

            data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
            data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

            # Add more technical indicators
            data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
            data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
            data['OBV'] = talib.OBV(data['Close'], data['Volume'])
            data['MOM'] = talib.MOM(data['Close'], timeperiod=14)
            data['Stochastic'], data['Stochastic_signal'] = talib.STOCH(data['High'], data['Low'], data['Close'])
            data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
            data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

            data['signal'] = (
                (data['percentage_drop'] > data['dynamic_threshold']) &
                data['volume_signal'] &
                (data['RSI'] < 50) &
                (data['MACD'] < data['MACD_signal']) &
                # (data['Stochastic'] > data['Stochastic_signal']) &  # Bullish Stochastic crossover
                (data['MOM'] > 0)  # Positive momentum
            )

            # Prepare features for ML model
            features = ['RSI', 'MACD', 'ADX', 'BB_upper', 'BB_middle', 'BB_lower', 'OBV', 'MOM', 'Stochastic', 'VWAP', 'SAR']

            data = data.dropna(subset=features + ['signal'])

            X = data[features].values
            y = data['signal'].values

            if len(X) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
                self.ml_model.fit(X_train, y_train)
                train_accuracy = accuracy_score(y_train, self.ml_model.predict(X_train))
                test_accuracy = accuracy_score(y_test, self.ml_model.predict(X_test))
                logger.info(f"ML model train accuracy: {train_accuracy:.2f}, test accuracy: {test_accuracy:.2f}")

                ml_predictions = self.ml_model.predict(X)
                data['ml_signal'] = ml_predictions
                log_info_highlight(f"ML model predictions made. Positive predictions: {sum(ml_predictions)}")

                feature_importance = pd.Series(self.ml_model.feature_importances_, index=features).sort_values(ascending=False)
                logger.info(f"Top 3 important features: {feature_importance.head(3).to_dict()}")
            else:
                data['ml_signal'] = 0
                logger.warning("Not enough data to train ML model")

            # Enhance buy signal criteria
            data['trend_confirmation'] = (data['Close'] > data['SMA50']) & (data['Close'] > data['SMA200'])
            data['volume_confirmation'] = data['Volume'] > data['Volume_MA20']
            data['rsi_oversold'] = data['RSI'] < 30
            data['macd_crossover'] = (data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1))
            data['bb_squeeze'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle'] < 0.1

            # Combine all buy criteria
            data['strong_buy_signal'] = (
                data['signal'] &
                data['ml_signal'] &
                data['trend_confirmation'] &
                data['volume_confirmation'] &
                (data['rsi_oversold'] | data['macd_crossover'] | data['bb_squeeze'])
            )

            log_info_highlight(f"Strong buy signal counts: {data['strong_buy_signal'].sum()}")

            return data, data[data['strong_buy_signal']]
        except Exception as e:
            logger.error(f"Error calculating signals: {e}")
            return None, None


class TradingSimulator:
    def __init__(self, initial_balance: float = 10000, risk_percentage: float = 0.01,
                 stop_loss_percentage: float = 0.02, profit_target: float = 0.04,
                 max_loss_per_trade: float = 0.01, trailing_stop_activation: float = 0.02):
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.profit_target = profit_target
        self.max_loss_per_trade = max_loss_per_trade
        self.trailing_stop_activation = trailing_stop_activation

    def simulate(self, data: pd.DataFrame, signals: pd.DataFrame) -> Tuple[float, int]:
        balance = self.initial_balance
        position = 0
        trades = 0
        entry_price = 0
        trailing_stop = 0

        for index, row in signals.iterrows():
            if position == 0:
                entry_price = row['Close']
                stop_loss_price = entry_price * (1 - self.stop_loss_percentage)
                take_profit_price = entry_price * (1 + self.profit_target)
                trailing_stop = stop_loss_price

                max_position_size = balance * self.max_loss_per_trade / self.stop_loss_percentage
                position_size = min(balance * self.risk_percentage / self.stop_loss_percentage, max_position_size)
                position = position_size / entry_price
                balance -= position * entry_price
                trades += 1
            else:
                current_price = row['Close']
                if current_price <= trailing_stop or current_price >= take_profit_price:
                    balance += position * current_price
                    position = 0
                elif current_price > entry_price * (1 + self.trailing_stop_activation):
                    new_trailing_stop = current_price * (1 - self.stop_loss_percentage)
                    trailing_stop = max(trailing_stop, new_trailing_stop)

        if position > 0:
            balance += position * data['Close'].iloc[-1]

        return balance - self.initial_balance, trades

class MarketRegimeDetector:
    def __init__(self, short_lookback: int = 50, long_lookback: int = 200):
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback

    def detect_regime(self, data: pd.DataFrame) -> bool:
        if len(data) < self.long_lookback:
            return True  # Default to bullish if not enough data

        short_sma = data['Close'].rolling(window=self.short_lookback).mean().iloc[-1]
        long_sma = data['Close'].rolling(window=self.long_lookback).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]

        return current_price > short_sma and short_sma > long_sma  # Bullish if price is above both SMAs and short SMA is above long SMA

class LeverageManager:
    def __init__(self, max_leverage: float = 3.0, min_leverage: float = 1.0):
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage

    def calculate_optimal_leverage(self, market_regime: str, volatility: float, portfolio_performance: float) -> float:
        base_leverage = 1.5 if market_regime == 'bullish' else 1.0
        volatility_factor = 1 - (volatility / 100)  # Reduce leverage for high volatility
        performance_factor = min(max(portfolio_performance, -0.2), 0.2) + 1  # Adjust leverage based on recent performance

        optimal_leverage = base_leverage * volatility_factor * performance_factor
        return max(min(optimal_leverage, self.max_leverage), self.min_leverage)

class RiskManager:
    def __init__(self, max_portfolio_risk: float = 0.02):
        self.max_portfolio_risk = max_portfolio_risk

    def calculate_position_size(self, current_portfolio_value: float, stock_volatility: float, leverage: float) -> float:
        max_risk_per_trade = self.max_portfolio_risk * current_portfolio_value
        position_size = max_risk_per_trade / (stock_volatility * leverage)
        logger.info(f"Position size calculation: Portfolio value: ${current_portfolio_value:.2f}, "
                    f"Volatility: {stock_volatility:.4f}, Leverage: {leverage:.2f}, "
                    f"Position size: ${position_size:.2f}")
        return position_size

class PerformanceTracker:
    def __init__(self):
        self.daily_returns = []
        self.cumulative_returns = []
        self.peak_value = 1
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.current_drawdown = 0
        self.max_drawdown = 0

    def update(self, daily_return: float, trade_result: float):
        self.daily_returns.append(daily_return)
        if not self.cumulative_returns:
            self.cumulative_returns.append(1 + daily_return)
        else:
            self.cumulative_returns.append(self.cumulative_returns[-1] * (1 + daily_return))

        current_value = self.cumulative_returns[-1]
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value

        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        self.total_trades += 1
        self.total_profit += trade_result
        if trade_result > 0:
            self.winning_trades += 1

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if not self.daily_returns:
            return 0
        returns = np.array(self.daily_returns)
        annualized_return = np.mean(returns) * 252
        annualized_std_dev = np.std(returns) * np.sqrt(252)
        return (annualized_return - risk_free_rate) / annualized_std_dev if annualized_std_dev != 0 else 0

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0

    @property
    def avg_profit_per_trade(self) -> float:
        return self.total_profit / self.total_trades if self.total_trades > 0 else 0

class MacroeconomicAnalyzer:
    def __init__(self):
        pass

    def get_economic_indicators(self) -> Dict[str, float]:
        return {}

    def analyze_macro_environment(self) -> str:
        return "neutral"

class AdvancedTradingBot:
    def __init__(self, initial_balance: float = 100000):
        self.analyzer = EnhancedYahooFinanceAnalyzer()
        self.simulator = TradingSimulator()
        self.regime_detector = MarketRegimeDetector()
        self.leverage_manager = LeverageManager()
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        self.macro_analyzer = MacroeconomicAnalyzer()
        self.initial_balance = initial_balance
        self.portfolio_value = initial_balance
        self.held_positions = {}  # To track held positions
        self.current_portfolio = {
            'TSLA': {'shares': 4.00175472, 'avg_cost': 205.16, 'current_price': 222.41},
            'AMAT': {'shares': 0.57259338, 'avg_cost': 174.64, 'current_price': 206.04},
            'COST': {'shares': 0.25318583, 'avg_cost': 710.94, 'current_price': 875.45},
            'AVGO': {'shares': 2.848094, 'avg_cost': 175.55, 'current_price': 164.96},
            'NVDA': {'shares': 47.74509687, 'avg_cost': 105.07, 'current_price': 127.45},
        }
        
        # Streamlit placeholders for the portfolio
        self.portfolio_value_placeholder = st.empty()
        self.holdings_table_placeholder = st.empty()
        self.signal_placeholder = st.empty()
        
        # Streamlit placeholders for other tickers
        self.other_tickers_value_placeholder = st.empty()
        self.other_tickers_signal_placeholder = st.empty()

        # Initializing new attributes for ML models, forecasters, and RL agents
        self.ml_models = {}
        self.forecasters = {}
        self.rl_agents = {}

    def initialize_models(self, tickers):
        for ticker in tickers:
            self.ml_models[ticker] = TickerModel(ticker)
            self.forecasters[ticker] = LSTMForecaster(ticker)
            env = TradingEnvironment(self.analyzer.fetch_data(ticker, period='1y'))
            self.rl_agents[ticker] = RLAgent(env)

    def calculate_current_portfolio_value(self) -> float:
        total_value = 0
        for ticker, details in self.current_portfolio.items():
            total_value += details['shares'] * details['current_price']
        return total_value

    def run(self, tickers: List[str], period: str = '5y'):
        for ticker in tickers:
            logger.info(f"Processing {ticker}")
            data = self.analyzer.fetch_data(ticker, period)
            if data is None or data.empty:
                logger.warning(f"No data found or data is insufficient for {ticker}, skipping.")
                continue

            market_regime = 'bullish' if self.regime_detector.detect_regime(data) else 'bearish'
            logger.info(f"Market regime for {ticker}: {market_regime}")

            macro_environment = self.macro_analyzer.analyze_macro_environment()
            logger.info(f"Macro environment: {macro_environment}")

            data, signals = self.analyzer.calculate_signals(data)
            if signals is None or signals.empty:
                logger.warning(f"No trading signals generated for {ticker}, skipping.")
                continue

            logger.info(f"Number of trading signals for {ticker}: {len(signals)}")

            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            leverage = self.leverage_manager.calculate_optimal_leverage(market_regime, volatility, self.performance_tracker.cumulative_returns[-1] if self.performance_tracker.cumulative_returns else 0)
            logger.info(f"Calculated leverage for {ticker}: {leverage:.2f}")

            position_size = self.risk_manager.calculate_position_size(self.portfolio_value, volatility, leverage)
            logger.info(f"Calculated position size for {ticker}: {position_size:.2f}")

            profit, trades = self.simulator.simulate(data, signals)
            self.portfolio_value += profit
            logger.info(f"Simulation result for {ticker}: Profit: ${profit:.2f}, Trades: {trades}")

            daily_return = profit / self.portfolio_value
            self.performance_tracker.update(daily_return, profit)

        self.report_performance()

    def report_performance(self):
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        sharpe_ratio = self.performance_tracker.get_sharpe_ratio()
        logger.info(f"Initial Portfolio Value: ${self.initial_balance:.2f}")
        logger.info(f"Final Portfolio Value: ${self.portfolio_value:.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {self.performance_tracker.max_drawdown:.2%}")
        logger.info(f"Win Rate: {self.performance_tracker.win_rate:.2%}")
        logger.info(f"Average Profit per Trade: ${self.performance_tracker.avg_profit_per_trade:.2f}")
        logger.info(f"Total Trades: {self.performance_tracker.total_trades}")

    def track_current_holdings(self):
        portfolio_value = 0
        holdings_data = []

        for ticker, details in self.current_portfolio.items():
            data = self.analyzer.fetch_data(ticker, period='1y')
            if data is None or data.empty:
                continue

            current_price = data['Close'].iloc[-1]
            avg_cost = details['avg_cost']
            shares_held = details['shares']

            unrealized_gain_loss = (current_price - avg_cost) * shares_held
            portfolio_value += shares_held * current_price

            holdings_data.append({
                'Ticker': ticker,
                'Shares Held': shares_held,
                'Avg Cost': avg_cost,
                'Current Price': current_price,
                'Unrealized P/L': unrealized_gain_loss
            })

        # Update the portfolio value metric
        self.portfolio_value_placeholder.metric("Total Current Portfolio Value", f"${portfolio_value:.2f}")

        # Update the table with current holdings
        holdings_df = pd.DataFrame(holdings_data)
        self.holdings_table_placeholder.table(holdings_df)

        return portfolio_value

    def calculate_opportunity_score(self, data: pd.DataFrame, signals: pd.DataFrame) -> float:
        if data.empty:
            return 0

        latest_data = data.iloc[-1]
        score = 0
        
        # Technical factors
        if 'SMA50' in data.columns and 'SMA200' in data.columns:
            if latest_data['Close'] > latest_data['SMA50'] and latest_data['Close'] > latest_data['SMA200']:
                score += 2
        if 'RSI' in data.columns and latest_data['RSI'] < 40:  # Slightly oversold
            score += 1
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            if latest_data['MACD'] > latest_data['MACD_signal']:
                score += 1
        if 'Volume' in data.columns and 'Volume_MA20' in data.columns:
            if latest_data['Volume'] > latest_data['Volume_MA20']:
                score += 1
        
        # Signal strength
        score += len(signals) * 0.5
        
        # Trend strength
        if len(data) >= 20:
            pct_change = (latest_data['Close'] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
            if pct_change > 0.05:
                score += 1
        
        return score

    def track_other_tickers(self, other_tickers: List[str]):
        opportunities = []

        for ticker in other_tickers:
            data = self.analyzer.fetch_data(ticker, period='1y')
            if data is None or data.empty:
                continue

            market_regime = 'bullish' if self.regime_detector.detect_regime(data) else 'bearish'
            macro_environment = self.macro_analyzer.analyze_macro_environment()

            data, signals = self.analyzer.calculate_signals(data)
            if signals is None or signals.empty:
                continue

            opportunity_score = self.calculate_opportunity_score(data, signals)
            
            current_price = data['Close'].iloc[-1]
            rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else None
            macd = data['MACD'].iloc[-1] if 'MACD' in data.columns else None
            signal_line = data['MACD_signal'].iloc[-1] if 'MACD_signal' in data.columns else None
            
            opportunities.append({
                'Ticker': ticker,
                'Score': opportunity_score,
                'Current Price': current_price,
                'RSI': rsi,
                'MACD': macd,
                'Signal Line': signal_line,
                'Market Regime': market_regime,
                'Macro Environment': macro_environment,
                'Signals': len(signals)
            })

        # Sort opportunities by score in descending order and take top 10
        top_opportunities = sorted(opportunities, key=lambda x: x['Score'], reverse=True)[:10]

        # Update the table with top opportunities
        opportunities_df = pd.DataFrame(top_opportunities)
        self.other_tickers_signal_placeholder.table(opportunities_df)

        # Provide detailed analysis for the top 3 opportunities
        for opp in top_opportunities[:3]:
            analysis = f"""
            Top Opportunity: {opp['Ticker']}
            Score: {opp['Score']:.2f}
            Current Price: ${opp['Current Price']:.2f}
            RSI: {opp['RSI']:.2f if opp['RSI'] is not None else 'N/A'}
            MACD: {opp['MACD']:.4f if opp['MACD'] is not None else 'N/A'}
            Signal Line: {opp['Signal Line']:.4f if opp['Signal Line'] is not None else 'N/A'}
            Market Regime: {opp['Market Regime']}
            Macro Environment: {opp['Macro Environment']}
            Number of Signals: {opp['Signals']}
            
            Analysis:
            - {'Potential oversold condition' if opp['RSI'] is not None and opp['RSI'] < 40 else 'RSI in neutral territory' if opp['RSI'] is not None else 'RSI data not available'}
            - {'MACD above signal line, indicating bullish momentum' if opp['MACD'] is not None and opp['Signal Line'] is not None and opp['MACD'] > opp['Signal Line'] else 'MACD below signal line, caution advised' if opp['MACD'] is not None and opp['Signal Line'] is not None else 'MACD data not available'}
            - {'Multiple buy signals detected' if opp['Signals'] > 1 else 'Buy signal detected'}
            - {'Bullish market regime supports potential upside' if opp['Market Regime'] == 'bullish' else 'Bearish market regime, exercise caution'}
            
            Action: Consider opening a long position with a stop loss at $
            {opp['Current Price'] * 0.95:.2f} (5% below current price) and a 
            target price of ${opp['Current Price'] * 1.1:.2f} (10% above current price).
            """
            self.signal_placeholder.info(analysis)

    def run_real_time(self, portfolio_tickers: List[str], other_tickers: List[str], period: str = '1mo', interval: int = 60):
        try:
            while True:
                try:
                    logger.info("Updating current holdings...")
                    self.track_current_holdings()  # Update current portfolio and table

                    logger.info("Analyzing other tickers for opportunities...")
                    self.track_other_tickers(other_tickers)  # Track and update other tickers

                    # Evaluate signals for portfolio tickers
                    for ticker in portfolio_tickers:
                        data = self.analyzer.fetch_data(ticker, period=period)
                        if data is None or data.empty:
                            logger.warning(f"No data found or data is insufficient for {ticker}, skipping.")
                            continue

                        # Update predictions and signals
                        data, signals = self.analyzer.calculate_signals(data)
                        if signals is None or signals.empty:
                            logger.warning(f"No trading signals generated for {ticker}, skipping.")
                            continue

                        self.update_predictions(ticker, data)  # Update predictions for the ticker
                        self.evaluate_signals(data, signals, ticker)

                    # Report actions taken
                    self.report_actions()

                    logger.info(f"Analysis complete. Sleeping for {interval} seconds...")
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"An error occurred during real-time analysis: {str(e)}")
                    logger.info("Continuing to next iteration...")
                    time.sleep(10)  # Short sleep before retrying
        except KeyboardInterrupt:
            logger.info("Real-time trading loop terminated by user.")


    def make_trading_decision(self, ticker, ml_prediction, forecast, rl_action, data):
        # Implement logic to combine different signals and make a trading decision
        # This is a simplified example
        if ml_prediction[0, 1] > 0.7 and forecast[0] > data['Close'].iloc[-1] and rl_action == 1:
            return 'buy'
        elif ml_prediction[0, 0] > 0.7 and forecast[0] < data['Close'].iloc[-1] and rl_action == 2:
            return 'sell'
        else:
            return 'hold'

    def execute_trade(self, ticker, decision):
        # Implement trade execution logic
        if decision == 'buy':
            # Buy logic
            pass
        elif decision == 'sell':
            # Sell logic
            pass

    def update_models(self, ticker, data):
        self.ml_models[ticker].train(data)
        self.forecasters[ticker].train(data)
        self.rl_agents[ticker].train(10)  # Train for 10 iterations

    def evaluate_signals(self, data: pd.DataFrame, signals: pd.DataFrame, ticker: str):
        position = ticker in self.held_positions
        for index, row in signals.iterrows():
            current_price = row['Close']
            avg_cost = self.current_portfolio[ticker]['avg_cost'] if ticker in self.current_portfolio else 0
            current_shares = self.current_portfolio[ticker]['shares'] if ticker in self.current_portfolio else 0

            unrealized_gain_loss = (current_price - avg_cost) * current_shares

            # Streamlit update
            self.signal_placeholder.text(f"{ticker}: Current Price = {current_price}, Avg Cost = {avg_cost}, Unrealized P/L = {unrealized_gain_loss}")

            if not position:  # No position, consider buying
                if current_price < avg_cost * 0.95:
                    action = f"Buy signal detected for {ticker} at {current_price}"
                    self.signal_placeholder.success(action)
                    self.held_positions[ticker] = {
                        'shares': 100,  # Example share amount
                        'entry_price': current_price,
                        'current_price': current_price
                    }
                    position = True
            else:  # Holding position, consider selling or holding
                if current_price < avg_cost * 1.05 or current_price < row['7_day_ema']:
                    action = f"Sell signal detected for {ticker} at {current_price}"
                    self.signal_placeholder.warning(action)
                    del self.held_positions[ticker]
                    position = False
                else:
                    action = f"Holding position for {ticker} at {current_price}"
                    self.signal_placeholder.info(action)
                    self.held_positions[ticker]['current_price'] = current_price

    def update_predictions(self, ticker: str, data: pd.DataFrame):
        if ticker in self.ml_models:
            model = self.ml_models[ticker]
            features = data[['RSI', 'MACD', 'ADX', 'BB_upper', 'BB_middle', 'BB_lower', 'OBV', 'MOM', 'Stochastic', 'VWAP', 'SAR']].dropna().values
            if len(features) > 0:
                model_predictions = model.predict(features)
                data['ml_prediction'] = model_predictions
                self.signal_placeholder.table(data[['ml_prediction', 'Close', 'RSI', 'MACD']].tail(10))
                logger.info(f"Updated predictions for {ticker} with latest data.")
            else:
                logger.warning(f"Not enough features to predict for {ticker}")

    def report_actions(self):
        if self.held_positions:
            logger.info("Currently Held Positions:")
            print("Currently Held Positions:")
            for ticker, details in self.held_positions.items():
                logger.info(f"Ticker: {ticker}, Shares: {details['shares']}, Current Price: ${details['current_price']:.2f}, Entry Price: ${details['entry_price']:.2f}")
                print(f"Ticker: {ticker}, Shares: {details['shares']}, Current Price: ${details['current_price']:.2f}, Entry Price: ${details['entry_price']:.2f}")
        else:
            logger.info("No positions currently held.")
            print("No positions currently held.")


def main():
    st.title("Advanced Trading Bot Dashboard")

    portfolio_tickers = [
        'TSLA', 'AMAT', 'COST', 'AVGO', 'NVDA'  # Your portfolio tickers
    ]

    other_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'LLY', 'UNH', 'V', 'WMT', 'JPM', 'XOM', 'AVGO', 'MA', 'PG', 'JNJ', 'ORCL', 'HD', 'CVX', 'ADBE', 'MRK', 'COST', 'KO', 'ABBV', 'PEP', 'BAC', 'CSCO', 'CRM', 'ACN', 'MCD', 'NFLX', 'LIN', 'AMD', 'TMO', 'TMUS', 'CMCSA', 'ABT', 'DIS', 'PFE', 'INTC', 'NKE', 'VZ', 'INTU', 'WFC', 'DHR', 'AMGN', 'PM', 'QCOM', 'COP', 'IBM', 'TXN', 'NOW', 'UNP', 'GE', 'AMAT', 'SPGI', 'BA', 'MS', 'CAT', 'HON', 'SBUX', 'RTX', 'UPS', 'AXP', 'LOW', 'NEE', 'T', 'LMT', 'BKNG', 'ELV', 'SYK', 'GS', 'DE', 'TJX', 'BMY', 'ISRG', 'BLK', 'MMC', 'VRTX', 'MDT', 'SCHW', 'PGR', 'PLD', 'MDLZ', 'GILD', 'ADP', 'CB', 'ETN', 'LRCX', 'CVS', 'REGN', 'CI', 'AMT', 'ADI', 'MU', 'C', 'SNPS', 'BSX', 'CME', 'ZTS', 'SLB', 'SO', 'CDNS', 'KLAC', 'EOG', 'EQIX', 'MO', 'NOC', 'ITW', 'WM', 'BDX', 'GD', 'DUK', 'AON', 'ANET', 'SHW', 'MCK', 'MCO', 'CL', 'ICE', 'FDX', 'HCA', 'HUM', 'CSX', 'CHTR', 'ORLY', 'CMG', 'APD', 'PYPL', 'MAR', 'MNST', 'MPC', 'ROP', 'PXD', 'TDG', 'CTAS', 'OXY', 'PH', 'AJG', 'USB', 'APH', 'MSI', 'MMM', 'TT', 'ECL', 'PSX', 'RSG', 'EMR', 'TGT', 'FCX', 'AZO', 'PNC', 'NXPI', 'AFL', 'WELL', 'CPRT', 'PCAR', 'NSC', 'ADSK', 'MET', 'AIG', 'STZ', 'SRE', 'HES', 'KDP', 'PSA', 'CARR', 'ODFL', 'WMB', 'VLO', 'HLT', 'ROST', 'MCHP', 'CCI', 'PAYX', 'DHI', 'KMB', 'KHC', 'EL', 'MSCI', 'HSY', 'COF', 'AEP', 'F', 'TEL', 'EW', 'GWW', 'TFC', 'TRV', 'ADM', 'CEG', 'DLR', 'FTNT', 'EXC', 'CNC', 'DXCM', 'GIS', 'SPG', 'OKE', 'NUE', 'D', 'GM', 'KMI', 'LVS', 'O', 'IDXX', 'EA', 'IQV', 'YUM', 'AME', 'BK', 'LHX', 'LEN', 'BKR', 'HAL', 'VRSK', 'JCI', 'DOW', 'FAST', 'ALL', 'AMP', 'SYY', 'OTIS', 'PRU', 'CTSH', 'IT', 'BIIB', 'XEL', 'CTVA', 'KR', 'A', 'URI', 'FIS', 'CMI', 'PEG', 'ED', 'PPG', 'LYB', 'NDAQ', 'ROK', 'DD', 'DVN', 'VICI', 'ON', 'CDW', 'FANG', 'GPN', 'VMC', 'MLM', 'IR', 'HPQ', 'NEM', 'MRNA', 'CAH', 'DG', 'PWR', 'TTWO', 'ANSS', 'WTW', 'RCL', 'WEC', 'WST', 'EXR', 'DLTR', 'MPWR', 'WBD', 'EIX', 'XYL', 'SBAC', 'AWK', 'GLW', 'AVB', 'FTV', 'EFX', 'CHD', 'CBRE', 'HIG', 'GRMN', 'ZBH', 'MTD', 'DAL', 'KEYS', 'WY', 'VRSN', 'APTV', 'TSCO', 'MOH', 'RMD', 'RJF', 'DFS', 'BRO', 'BR', 'HWM', 'STT', 'TROW', 'EBAY', 'EQR', 'HPE', 'CTRA', 'WAB', 'LYV', 'FE', 'DTE', 'ETR', 'STE', 'AEE', 'MTB', 'ULTA', 'NVR', 'GPC', 'ROL', 'ES', 'DOV', 'PPL', 'DRI', 'IFF', 'TDY', 'PTC', 'JBHT', 'WRB', 'HRL', 'PHM', 'TYL', 'HOLX', 'MKC', 'WBA', 'LH', 'IRM', 'FDS', 'VTR', 'BAX', 'FITB', 'CNP', 'J', 'AKAM', 'CLX', 'EXPD', 'FLT', 'PFG', 'ATO', 'EXPE', 'ARE', 'COO', 'TSN', 'CMS', 'NTAP', 'CINF', 'BALL', 'CF', 'STX', 'OMC', 'TXT', 'DGX', 'WAT', 'HBAN', 'L', 'NTRS', 'ILMN', 'MRO', 'ALGN', 'WDC', 'IEX', 'AVY', 'LDOS', 'CCL', 'FOXA', 'SWKS', 'SNA', 'MAA', 'RF', 'LW', 'BBY', 'LUV', 'PKG', 'EPAM', 'ALB', 'FOX', 'TER', 'ESS', 'CAG', 'DPZ', 'MGM', 'AMCR', 'SWK', 'CE', 'NDSN', 'POOL', 'TAP', 'UAL', 'MAS', 'SYF', 'NWS', 'LNT', 'CPB', 'LKQ', 'INCY', 'NWSA', 'CFG', 'HST', 'MOS', 'APA', 'BEN', 'IP', 'SJM', 'EVRG', 'REG', 'IPG', 'GL', 'JKHY', 'AOS', 'KIM', 'VTRS', 'ENPH', 'UDR', 'ZBRA', 'AES', 'NRG', 'PAYC', 'NI', 'KEY', 'TRMB', 'CDAY', 'KMX', 'PNR', 'WRK', 'TFX', 'WYNN', 'FFIV', 'HII', 'CPT', 'CHRW', 'EMN', 'CZR', 'ALLE', 'TECH', 'UHS', 'BIO', 'HSIC', 'QRVO', 'CRL', 'JNPR', 'AIZ', 'PEAK', 'MKTX', 'RHI', 'DVA', 'BXP', 'MTCH', 'PNW', 'PARA', 'AAL', 'BWA', 'ETSY', 'RL', 'FRT', 'BBWI', 'TPR', 'FMC', 'GNRC', 'CTLT', 'HAS', 'XRAY', 'WHR', 'IVZ', 'NCLH', 'VFC', 'CMA', 'MHK', 'ZION', 'PVH', 'SEE', 'ALK', 'DXC', 'SEDG', 'VNO', 'LNC', 'PENN', 'AAP', 'OGN', 'NWL', 'DISH', 'LUMN', 'EMBC', 'K' # Additional tickers to analyze
    ]

    bot = AdvancedTradingBot()

    # Add placeholders for metrics
    initial_value_placeholder = st.empty()
    final_value_placeholder = st.empty()
    total_return_placeholder = st.empty()
    sharpe_ratio_placeholder = st.empty()
    max_drawdown_placeholder = st.empty()
    win_rate_placeholder = st.empty()
    avg_profit_per_trade_placeholder = st.empty()
    total_trades_placeholder = st.empty()

    # Function to update the Streamlit dashboard
    def update_dashboard():
        initial_value_placeholder.metric("Initial Portfolio Value", f"${bot.initial_balance:.2f}")
        final_value_placeholder.metric("Final Portfolio Value", f"${bot.portfolio_value:.2f}")
        total_return_placeholder.metric("Total Return", f"{(bot.portfolio_value - bot.initial_balance) / bot.initial_balance:.2%}")
        sharpe_ratio_placeholder.metric("Sharpe Ratio", f"{bot.performance_tracker.get_sharpe_ratio():.2f}")
        max_drawdown_placeholder.metric("Max Drawdown", f"{bot.performance_tracker.max_drawdown:.2%}")
        win_rate_placeholder.metric("Win Rate", f"{bot.performance_tracker.win_rate:.2%}")
        avg_profit_per_trade_placeholder.metric("Avg Profit per Trade", f"${bot.performance_tracker.avg_profit_per_trade:.2f}")
        total_trades_placeholder.metric("Total Trades", f"{bot.performance_tracker.total_trades}")

    # Analyze your portfolio separately
    log_info_highlight("Analyzing portfolio tickers...")
    bot.run(portfolio_tickers, period='5y')
    bot.run(portfolio_tickers, period='1y')

    # Analyze other tickers separately
    log_info_highlight("Analyzing additional tickers...")
    bot.run(other_tickers, period='5y')
    bot.run(other_tickers, period='1y')

    # Update the dashboard with initial analysis results
    update_dashboard()

    # Real-time tracking loop
    log_info_highlight("Starting real-time analysis...")
    try:
        while True:
            bot.run_real_time(portfolio_tickers, other_tickers=other_tickers, period='1y', interval=30)
            update_dashboard()  # Update the dashboard in real-time
    except KeyboardInterrupt:
        logger.info("Real-time trading loop terminated by user.")

if __name__ == "__main__":
    main()

