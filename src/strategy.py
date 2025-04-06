import numpy as np
import pandas as pd

def generate_signals(predictions, threshold=0.0):
    """
    Generate trading signals based on the predicted prices.
    
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
    Backtest the trading strategy using generated signals.
    
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
