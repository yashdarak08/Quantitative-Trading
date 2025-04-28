import numpy as np
import pandas as pd

def sma(data, window=20):
    """
    Calculate Simple Moving Average (SMA)
    
    Parameters:
        data (pd.Series): Price series
        window (int): Window size for moving average
        
    Returns:
        pd.Series: Simple Moving Average
    """
    return data.rolling(window=window).mean()

def ema(data, window=20):
    """
    Calculate Exponential Moving Average (EMA)
    
    Parameters:
        data (pd.Series): Price series
        window (int): Window size for moving average
        
    Returns:
        pd.Series: Exponential Moving Average
    """
    return data.ewm(span=window, min_periods=window, adjust=False).mean()

def rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Parameters:
        data (pd.Series): Price series
        window (int): Window size for RSI calculation
        
    Returns:
        pd.Series: RSI values
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Parameters:
        data (pd.Series): Price series
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        pd.DataFrame: DataFrame with 'MACD', 'Signal', and 'Histogram' columns
    """
    # Calculate MACD line
    fast_ema = ema(data, window=fast_period)
    slow_ema = ema(data, window=slow_period)
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = ema(macd_line, window=signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Create DataFrame with results
    result = pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    }, index=data.index)
    
    return result

def bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Parameters:
        data (pd.Series): Price series
        window (int): Window size for moving average
        num_std (float): Number of standard deviations for bands
        
    Returns:
        pd.DataFrame: DataFrame with 'Middle', 'Upper', and 'Lower' columns
    """
    middle_band = sma(data, window=window)
    std = data.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    result = pd.DataFrame({
        'Middle': middle_band,
        'Upper': upper_band,
        'Lower': lower_band
    }, index=data.index)
    
    return result

def atr(data, window=14):
    """
    Calculate Average True Range (ATR)
    
    Parameters:
        data (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns
        window (int): Window size for ATR calculation
        
    Returns:
        pd.Series: ATR values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.DataFrame({
        'TR1': tr1,
        'TR2': tr2,
        'TR3': tr3
    }).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=window).mean()
    
    return atr

def momentum(data, window=10):
    """
    Calculate Momentum
    
    Parameters:
        data (pd.Series): Price series
        window (int): Window size for momentum calculation
        
    Returns:
        pd.Series: Momentum values
    """
    return data / data.shift(window) - 1

def mean_reversion_signal(data, window=20, threshold=1.5):
    """
    Generate mean reversion signals based on z-score
    
    Parameters:
        data (pd.Series): Price series
        window (int): Lookback period for mean and standard deviation
        threshold (float): Z-score threshold for generating signals
        
    Returns:
        pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    # Calculate z-score
    z_score = (data - rolling_mean) / rolling_std
    
    # Generate signals based on z-score
    signals = pd.Series(0, index=data.index)
    signals[z_score < -threshold] = 1  # Buy when price is below mean (negative z-score)
    signals[z_score > threshold] = -1  # Sell when price is above mean (positive z-score)
    
    return signals

def momentum_signal(data, window=12, threshold=0.0):
    """
    Generate momentum signals based on rate of change
    
    Parameters:
        data (pd.Series): Price series
        window (int): Lookback period for momentum calculation
        threshold (float): Threshold for generating signals
        
    Returns:
        pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
    """
    # Calculate momentum (rate of change)
    mom = momentum(data, window)
    
    # Generate signals based on momentum
    signals = pd.Series(0, index=data.index)
    signals[mom > threshold] = 1     # Buy when momentum is positive
    signals[mom < -threshold] = -1   # Sell when momentum is negative
    
    return signals

def dual_moving_average_signal(data, fast_window=50, slow_window=200):
    """
    Generate trading signals based on dual moving average crossover
    
    Parameters:
        data (pd.Series): Price series
        fast_window (int): Fast moving average window
        slow_window (int): Slow moving average window
        
    Returns:
        pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
    """
    # Calculate fast and slow moving averages
    fast_ma = sma(data, window=fast_window)
    slow_ma = sma(data, window=slow_window)
    
    # Initialize signals
    signals = pd.Series(0, index=data.index)
    
    # Golden Cross (fast MA crosses above slow MA)
    golden_cross = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
    
    # Death Cross (fast MA crosses below slow MA)
    death_cross = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
    
    # Generate signals
    signals[golden_cross] = 1
    signals[death_cross] = -1
    
    return signals

def macd_signal(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Generate trading signals based on MACD crossover
    
    Parameters:
        data (pd.Series): Price series
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
    """
    # Calculate MACD
    macd_data = macd(data, fast_period, slow_period, signal_period)
    
    # Initialize signals
    signals = pd.Series(0, index=data.index)
    
    # Buy signal: MACD crosses above signal line
    buy_signal = (macd_data['MACD'] > macd_data['Signal']) & (macd_data['MACD'].shift() <= macd_data['Signal'].shift())
    
    # Sell signal: MACD crosses below signal line
    sell_signal = (macd_data['MACD'] < macd_data['Signal']) & (macd_data['MACD'].shift() >= macd_data['Signal'].shift())
    
    # Generate signals
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    
    return signals

def combine_signals(signals_dict, weights=None):
    """
    Combine multiple trading signals with optional weighting
    
    Parameters:
        signals_dict (dict): Dictionary with strategy names as keys and signal Series as values
        weights (dict, optional): Dictionary with strategy names as keys and weights as values
        
    Returns:
        pd.Series: Combined trading signals
    """
    if weights is None:
        # Equal weighting if not specified
        weights = {strategy: 1/len(signals_dict) for strategy in signals_dict.keys()}
    
    # Combine signals
    combined = pd.Series(0, index=list(signals_dict.values())[0].index)
    
    for strategy, signal in signals_dict.items():
        combined += signal * weights[strategy]
    
    # Discretize the combined signal
    combined[combined > 0.2] = 1
    combined[combined < -0.2] = -1
    combined[(combined >= -0.2) & (combined <= 0.2)] = 0
    
    return combined