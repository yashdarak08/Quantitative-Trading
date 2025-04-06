import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import fetch_data, prepare_dataset
from model import build_model
from strategy import generate_signals, backtest_strategy

def load_config(path='config/config.yaml'):
    """
    Load configuration parameters from a YAML file.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration parameters
    config = load_config()
    
    # Choose ticker for training (using S&P500 index here)
    ticker = config['data']['tickers'][1]  # "^GSPC"
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    print("Fetching data for:", ticker)
    data = fetch_data(ticker, start_date, end_date)
    print("Data shape:", data.shape)
    
    # Prepare dataset with a sliding window (e.g., 60 days)
    window_size = 60
    X, y = prepare_dataset(data, window_size)
    
    # Split dataset into training and testing sets (80/20 split)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Build the forecasting model (choose between 'LSTM' and 'RNN')
    model_type = 'LSTM'
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=config['model']['lstm_units'],
        dropout_rate=config['model']['dropout'],
        model_type=model_type
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=config['model']['epochs'],
        batch_size=config['model']['batch_size'],
        validation_data=(X_test, y_test)
    )
    
    # Generate predictions on the test set
    predictions = model.predict(X_test).flatten()
    
    # Align predictions with the original data (account for window offset)
    test_data = data.iloc[split_index + window_size:].copy()
    test_data = test_data.iloc[:len(predictions)]
    test_data['Predicted'] = predictions
    
    # Generate trading signals based on predictions
    signals = generate_signals(predictions, threshold=config['strategy']['threshold'])
    
    # Backtest the trading strategy using generated signals
    result = backtest_strategy(test_data, signals)
    
    # Plot cumulative returns for the market vs. strategy
    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result['Cumulative_Market'], label='Market Return')
    plt.plot(result.index, result['Cumulative_Strategy'], label='Strategy Return')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Backtest: Strategy vs. Market Returns')
    plt.legend()
    plt.show()
    
    # Print final cumulative returns
    final_market = result['Cumulative_Market'].iloc[-1]
    final_strategy = result['Cumulative_Strategy'].iloc[-1]
    print("Final Market Return:", final_market)
    print("Final Strategy Return:", final_strategy)

if __name__ == '__main__':
    main()
