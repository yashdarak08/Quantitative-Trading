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