import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical data for the given ticker from Yahoo Finance.
    Returns a DataFrame with the adjusted close prices.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']].rename(columns={'Adj Close': 'Price'})
    data.dropna(inplace=True)
    return data

def prepare_dataset(data, window_size=60):
    """
    Prepare the dataset for training.
    Creates sequences of length `window_size` to be used for time series forecasting.
    
    Returns:
        X: Array of input sequences (samples, window_size, 1)
        y: Array of target prices
    """
    prices = data['Price'].values
    X, y = [], []
    for i in range(window_size, len(prices)):
        X.append(prices[i-window_size:i])
        y.append(prices[i])
    X, y = np.array(X), np.array(y)
    # Reshape X to be [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y
