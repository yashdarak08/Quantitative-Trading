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
    import numpy as np
    
    prices = data['Price'].values
    X, y = [], []
    for i in range(window_size, len(prices)):
        X.append(prices[i-window_size:i])
        y.append(prices[i])
    X, y = np.array(X), np.array(y)
    # Reshape X to be [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def create_pytorch_datasets(X_train, y_train, X_test, y_test):
    """
    Create PyTorch datasets from numpy arrays.
    
    Parameters:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        train_dataset, test_dataset: PyTorch datasets
    """
    from torch.utils.data import TensorDataset
    import torch
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset

def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader from a dataset.
    
    Parameters:
        dataset: PyTorch Dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Add this function to src/data_loader.py:

def load_data(ticker_or_path, start_date=None, end_date=None, source='yahoo'):
    """
    Load financial data from various sources.
    
    Parameters:
        ticker_or_path (str): Ticker symbol or file path
        start_date (str, optional): Start date (format: 'YYYY-MM-DD')
        end_date (str, optional): End date (format: 'YYYY-MM-DD')
        source (str, optional): Data source ('yahoo', 'csv', 'excel')
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    import os
    
    # Check if it's a file path
    if os.path.exists(ticker_or_path):
        if ticker_or_path.endswith('.csv'):
            data = pd.read_csv(ticker_or_path, index_col=0, parse_dates=True)
        elif ticker_or_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(ticker_or_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {ticker_or_path}")
    # Fetch from online source
    elif source.lower() == 'yahoo':
        import yfinance as yf
        data = yf.download(ticker_or_path, start=start_date, end=end_date)
        # Rename columns for consistency
        if 'Adj Close' in data.columns:
            data = data[['Adj Close']].rename(columns={'Adj Close': 'Price'})
        else:
            # If Adj Close is not available, use Close
            data = data[['Close']].rename(columns={'Close': 'Price'})
    else:
        raise ValueError(f"Unsupported data source: {source}")
    
    # Ensure the DataFrame has the expected format
    if 'Price' not in data.columns:
        if 'Close' in data.columns:
            data['Price'] = data['Close']
        else:
            raise ValueError("No Price or Close column found in the data")
    
    # Make sure the index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Add required columns if missing
    if 'High' not in data.columns:
        data['High'] = data['Price']
    if 'Low' not in data.columns:
        data['Low'] = data['Price']
    if 'Close' not in data.columns:
        data['Close'] = data['Price']
    
    # Filter by date if provided
    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index <= pd.to_datetime(end_date)]
    
    data.dropna(inplace=True)
    return data