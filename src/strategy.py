import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from indicators import momentum_signal, mean_reversion_signal, dual_moving_average_signal, macd_signal, combine_signals

class Strategy(ABC):
    """Base abstract class for all trading strategies"""
    
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.positions = None
        self.history = None
    
    @abstractmethod
    def generate_signals(self, data):
        """Generate trading signals based on data"""
        pass
    
    def apply_risk_management(self, signals, data, risk_params=None):
        """Apply risk management rules to trading signals"""
        risk_params = risk_params or {}
        
        # Default risk parameters
        stop_loss = risk_params.get('stop_loss', 0.05)
        take_profit = risk_params.get('take_profit', 0.1)
        max_position_size = risk_params.get('max_position_size', 1.0)
        
        # Initialize position sizes based on signals
        position_sizes = signals.copy()
        position_sizes = position_sizes * max_position_size
        
        # Apply stop-loss and take-profit
        if self.positions is not None and len(self.positions) > 0:
            for i in range(1, len(position_sizes)):
                if self.positions.iloc[i-1] != 0:
                    # Calculate return since position entry
                    entry_price = data.loc[self.positions.iloc[i-1] != 0].iloc[0]['Price']
                    current_price = data.iloc[i]['Price']
                    position_return = (current_price / entry_price - 1) * np.sign(self.positions.iloc[i-1])
                    
                    # Apply stop-loss
                    if position_return < -stop_loss:
                        position_sizes.iloc[i] = 0
                    
                    # Apply take-profit
                    if position_return > take_profit:
                        position_sizes.iloc[i] = 0
        
        return position_sizes
    
    def backtest(self, data, risk_params=None):
        """Backtest the strategy on historical data"""
        # Generate trading signals
        signals = self.generate_signals(data)
        
        # Apply risk management
        self.positions = self.apply_risk_management(signals, data, risk_params)
        
        # Calculate returns
        data = data.copy()
        data['Signal'] = self.positions.shift(1).fillna(0)  # Apply signal from previous day
        data['Return'] = data['Price'].pct_change()
        data['Strategy'] = data['Signal'] * data['Return']
        
        # Calculate cumulative returns
        data['Cumulative_Market'] = (1 + data['Return']).cumprod()
        data['Cumulative_Strategy'] = (1 + data['Strategy']).cumprod()
        
        self.history = data
        return data


class MomentumStrategy(Strategy):
    """Strategy based on price momentum"""
    
    def __init__(self, config=None):
        super().__init__('Momentum', config)
        # Default parameters
        self.window = self.config.get('momentum_window', 12)
        self.threshold = self.config.get('momentum_threshold', 0.0)
    
    def generate_signals(self, data):
        """Generate momentum-based trading signals"""
        price = data['Price']
        return momentum_signal(price, window=self.window, threshold=self.threshold)


class MeanReversionStrategy(Strategy):
    """Strategy based on mean reversion"""
    
    def __init__(self, config=None):
        super().__init__('MeanReversion', config)
        # Default parameters
        self.window = self.config.get('mean_reversion_window', 20)
        self.threshold = self.config.get('mean_reversion_threshold', 1.5)
    
    def generate_signals(self, data):
        """Generate mean reversion trading signals"""
        price = data['Price']
        return mean_reversion_signal(price, window=self.window, threshold=self.threshold)


class MovingAverageCrossoverStrategy(Strategy):
    """Strategy based on moving average crossover"""
    
    def __init__(self, config=None):
        super().__init__('MovingAverageCrossover', config)
        # Default parameters
        self.fast_window = self.config.get('fast_ma_window', 50)
        self.slow_window = self.config.get('slow_ma_window', 200)
    
    def generate_signals(self, data):
        """Generate moving average crossover trading signals"""
        price = data['Price']
        return dual_moving_average_signal(price, fast_window=self.fast_window, slow_window=self.slow_window)


class MACDStrategy(Strategy):
    """Strategy based on MACD indicator"""
    
    def __init__(self, config=None):
        super().__init__('MACD', config)
        # Default parameters
        self.fast_period = self.config.get('macd_fast_period', 12)
        self.slow_period = self.config.get('macd_slow_period', 26)
        self.signal_period = self.config.get('macd_signal_period', 9)
    
    def generate_signals(self, data):
        """Generate MACD-based trading signals"""
        price = data['Price']
        return macd_signal(price, 
                          fast_period=self.fast_period, 
                          slow_period=self.slow_period, 
                          signal_period=self.signal_period)


class MLStrategy(Strategy):
    """Strategy based on machine learning model predictions"""
    
    def __init__(self, model, config=None):
        super().__init__('ML', config)
        self.model = model
        self.threshold = self.config.get('ml_threshold', 0.0)
        self.window_size = self.config.get('window_size', 60)
    
    def prepare_features(self, data):
        """Prepare features for model prediction"""
        # Simple implementation using price changes as features
        # In a real-world scenario, this would include more complex feature engineering
        prices = data['Price'].values
        X = []
        for i in range(self.window_size, len(prices)):
            X.append(prices[i-self.window_size:i])
        
        X = np.array(X)
        # Reshape X to be [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X
    
    def generate_signals(self, data):
        """Generate trading signals based on model predictions"""
        # Prepare features
        X = self.prepare_features(data)
        
        # Skip initial window where we don't have enough data
        signals = pd.Series(0, index=data.index)
        
        if len(X) == 0:
            return signals
        
        # Generate predictions using the PyTorch model wrapper
        if hasattr(self.model, 'predict'):
            # Use the wrapper's predict method if available
            predictions = self.model.predict(X).flatten()
        else:
            # Use the PyTorch model directly with proper conversion
            import torch
            import numpy as np
            
            # Convert to tensor for prediction
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                predictions = self.model(X_tensor).numpy().flatten()
        
        # Convert predictions to signals
        signal_indices = data.index[self.window_size:]
        signals_array = np.zeros(len(predictions))
        
        for i in range(1, len(predictions)):
            if predictions[i] - predictions[i-1] > self.threshold:
                signals_array[i] = 1
            elif predictions[i] - predictions[i-1] < -self.threshold:
                signals_array[i] = -1
            else:
                signals_array[i] = 0
        
        # Update the signals Series
        signals.loc[signal_indices] = signals_array
        
        return signals


class EnsembleStrategy(Strategy):
    """Combines multiple strategies with optional weighting"""
    
    def __init__(self, strategies, weights=None, config=None):
        super().__init__('Ensemble', config)
        self.strategies = strategies
        
        if weights is None:
            # Equal weighting if not specified
            self.weights = {strategy.name: 1/len(strategies) for strategy in strategies}
        else:
            self.weights = weights
    
    def generate_signals(self, data):
        """Generate combined signals from multiple strategies"""
        signals_dict = {}
        
        # Generate signals for each strategy
        for strategy in self.strategies:
            signals_dict[strategy.name] = strategy.generate_signals(data)
        
        # Combine signals
        return combine_signals(signals_dict, self.weights)


def generate_signals(predictions, threshold=0.0):
    """
    Generate trading signals based on the predicted prices (legacy function).
    
    Parameters:
        predictions (np.array): Array of predicted prices.
        threshold (float): Minimum difference to trigger a signal.
        
    Returns:
        signals (np.array): Trading signals (1 for buy, -1 for sell, 0 for hold).
    """
    signals = []
    for i in range(1, len(predictions)):
        if predictions[i] - predictions[i-1] > threshold:
            signals.append(1)
        elif predictions[i] - predictions[i-1] < -threshold:
            signals.append(-1)
        else:
            signals.append(0)
    signals.insert(0, 0)  # No signal for the first prediction
    return np.array(signals)


def backtest_strategy(data, signals):
    """
    Backtest the trading strategy using generated signals (legacy function).
    
    Parameters:
        data (DataFrame): DataFrame with 'Price' and 'Predicted' columns.
        signals (np.array): Trading signals.
    
    Returns:
        data (DataFrame): DataFrame with added columns for returns and cumulative returns.
    """
    data = data.copy()
    data['Signal'] = signals
    data['Return'] = data['Price'].pct_change()
    # Compute strategy returns: using the signal from the previous time step
    data['Strategy'] = data['Signal'].shift(1) * data['Return']
    data['Cumulative_Market'] = (1 + data['Return']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy']).cumprod()
    return data