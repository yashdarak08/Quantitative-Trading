import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# Import modules
from data_loader import fetch_data, prepare_dataset
from model import build_model, ModelTrainer, ModelWrapper, TimeSeriesDataset
from strategy import (MomentumStrategy, MeanReversionStrategy, MovingAverageCrossoverStrategy, 
                     MACDStrategy, EnsembleStrategy, MLStrategy)
from backtest import BacktestEngine
from risk_management import RiskManager
from optimization import (optimize_momentum_strategy, optimize_mean_reversion_strategy, 
                          optimize_moving_average_strategy, optimize_macd_strategy)
from visualization import create_performance_report

# In src/main.py, replace the load_config function:

def load_config(path='config/config.yaml'):
    """
    Load configuration parameters from a YAML file.
    """
    import os
    
    # Check if the file exists
    if not os.path.exists(path):
        print(f"Warning: Config file {path} not found. Using default configuration.")
        return {
            'data': {
                'tickers': ['^GSPC'],  # S&P 500 by default
                'start_date': '2020-01-01',
                'end_date': '2023-01-01'
            },
            'model': {
                'epochs': 50,
                'batch_size': 32,
                'lstm_units': 50,
                'dropout': 0.2
            },
            'strategy': {
                'threshold': 0.0,
                'stop_loss': 0.05,
                'take_profit': 0.1,
                'transaction_cost': 0.001
            }
        }
    
    # Load configuration from file
    with open(path, 'r') as f:
        try:
            import yaml
            config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}. Using default configuration.")
            return {}
    
    return config

def save_model(model, path='models'):
    """
    Save the trained model.
    """
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(path, f'model_{timestamp}.pt')
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
        'timestamp': timestamp
    }, model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    """
    Load a saved model.
    """
    checkpoint = torch.load(model_path)
    
    # Determine input/output dimensions and model type
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    else:
        # Try to infer model type from state dict
        state_dict = checkpoint['model_state_dict']
        if any('lstm' in key for key in state_dict.keys()):
            model_type = 'LSTMModel'
        elif any('gru' in key for key in state_dict.keys()):
            model_type = 'GRUModel'
        else:
            model_type = 'RNNModel'
    
    # Get input/output dimensions from first and last layer
    state_dict = checkpoint['model_state_dict']
    for key, value in state_dict.items():
        if 'weight_ih_l0' in key:  # First layer weights
            input_dim = value.shape[1] // 4 if model_type == 'LSTMModel' else value.shape[1]
        if 'fc.weight' in key:  # Last layer weights
            output_dim = value.shape[0]
            hidden_dim = value.shape[1]
    
    # Create model of the right type
    if model_type == 'LSTMModel':
        from model import LSTMModel
        model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif model_type == 'GRUModel':
        from model import GRUModel
        model = GRUModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    else:
        from model import RNNModel
        model = RNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer and wrapper for consistent interface
    trainer = ModelTrainer(model)
    wrapper = ModelWrapper(model, trainer)
    
    return wrapper

def train_forecasting_model(X_train, y_train, X_test, y_test, config):
    """
    Train a forecasting model.
    """
    # Get model parameters from config
    lstm_units = config['model']['lstm_units']
    dropout = config['model']['dropout']
    epochs = config['model']['epochs']
    batch_size = config['model']['batch_size']
    
    # Determine input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build the model
    model = build_model(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout,
        model_type='LSTM'
    )
    
    # Create dataset and data loader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TimeSeriesDataset(X_test, y_test)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create trainer
    trainer = ModelTrainer(model, learning_rate=0.001)
    
    # Train the model
    print("Training model...")
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/training_history.png')
    
    # Create wrapper for consistent interface
    wrapper = ModelWrapper(model, trainer)
    wrapper.history = history
    
    return wrapper

def create_strategy(strategy_name, config, model=None):
    """
    Create a strategy instance based on name.
    """
    if strategy_name == 'momentum':
        return MomentumStrategy(config)
    elif strategy_name == 'mean_reversion':
        return MeanReversionStrategy(config)
    elif strategy_name == 'moving_average':
        return MovingAverageCrossoverStrategy(config)
    elif strategy_name == 'macd':
        return MACDStrategy(config)
    elif strategy_name == 'ml':
        if model is None:
            raise ValueError("Model is required for ML strategy")
        return MLStrategy(model, config)
    elif strategy_name == 'ensemble':
        # Create an ensemble of all strategies
        strategies = [
            MomentumStrategy(config),
            MeanReversionStrategy(config),
            MovingAverageCrossoverStrategy(config),
            MACDStrategy(config)
        ]
        # Add ML strategy if model is provided
        if model is not None:
            strategies.append(MLStrategy(model, config))
        
        # Equal weighting for all strategies
        weights = {strategy.name: 1/len(strategies) for strategy in strategies}
        
        return EnsembleStrategy(strategies, weights, config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def run_backtest(data, strategy, config):
    """
    Run backtest for a strategy.
    """
    print(f"Running backtest for strategy: {strategy.name}")
    backtest = BacktestEngine(data, strategy, config)
    results = backtest.run()
    
    # Print performance summary
    print(backtest.get_performance_summary())
    
    # Create and save performance report
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    figures = create_performance_report(
        results, 
        backtest.performance_metrics, 
        backtest.trades, 
        strategy.name
    )
    
    # Save figures
    for i, fig in enumerate(figures):
        filename = os.path.join(reports_dir, f"{strategy.name}_{timestamp}_{i}.png")
        fig.savefig(filename)
        plt.close(fig)
    
    return results, backtest.performance_metrics

def optimize_strategy(data, strategy_name, config):
    """
    Optimize strategy parameters.
    """
    print(f"Optimizing {strategy_name} strategy...")
    
    if strategy_name == 'momentum':
        result = optimize_momentum_strategy(data, config)
    elif strategy_name == 'mean_reversion':
        result = optimize_mean_reversion_strategy(data, config)
    elif strategy_name == 'moving_average':
        result = optimize_moving_average_strategy(data, config)
    elif strategy_name == 'macd':
        result = optimize_macd_strategy(data, config)
    else:
        raise ValueError(f"Optimization not implemented for strategy: {strategy_name}")
    
    # Print best parameters
    print("Best parameters found:")
    for param, value in result['params'].items():
        print(f"  {param}: {value}")
    
    # Print best metrics
    print("\nPerformance with best parameters:")
    metrics = result['metrics']
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio_strategy']:.2f}")
    print(f"  Annual Return: {metrics['annual_return_strategy']:.2%}")
    print(f"  Max Drawdown: {metrics['max_drawdown_strategy']:.2%}")
    
    return result['params']

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantitative Trading System')
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--strategy', default='ensemble', choices=['momentum', 'mean_reversion', 'moving_average', 'macd', 'ml', 'ensemble'], help='Trading strategy to use')
    parser.add_argument('--train', action='store_true', help='Train forecasting model')
    parser.add_argument('--optimize', action='store_true', help='Optimize strategy parameters')
    parser.add_argument('--model_path', help='Path to saved model')
    parser.add_argument('--device', default='', help='Device to use (cpu, cuda)')
    args = parser.parse_args()
    
    # Set device for PyTorch
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Data parameters
    tickers = config['data']['tickers']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    # Process each ticker
    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        
        # Fetch data
        print("Fetching data...")
        data = fetch_data(ticker, start_date, end_date)
        
        # Get price data with additional columns for indicators
        if 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
            # If only 'Price' column is available, add dummy columns for High/Low/Close
            # This handles the case from the original data_loader.py which only returns 'Price'
            data['High'] = data['Price']
            data['Low'] = data['Price']
            data['Close'] = data['Price']
        
        # Split data into training and testing sets (80/20)
        split_index = int(0.8 * len(data))
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        
        # Train or load forecasting model if needed
        model = None
        if args.strategy == 'ml' or args.strategy == 'ensemble':
            if args.train:
                # Prepare dataset with a sliding window (e.g., 60 days)
                window_size = 60
                X_train, y_train = prepare_dataset(train_data, window_size)
                X_test, y_test = prepare_dataset(test_data, window_size)
                
                # Train model
                model = train_forecasting_model(X_train, y_train, X_test, y_test, config)
                
                # Save model
                model_path = save_model(model.model)
                model.model_path = model_path
            elif args.model_path:
                # Load saved model
                model = load_model(args.model_path)
                print(f"Loaded model from {args.model_path}")
            else:
                print("Warning: ML strategy selected but no model provided. Will use other strategies for ensemble.")
        
        # Optimize strategy parameters if requested
        if args.optimize:
            strategy_to_optimize = args.strategy
            if strategy_to_optimize == 'ensemble':
                # For ensemble, optimize each component strategy
                for strategy_name in ['momentum', 'mean_reversion', 'moving_average', 'macd']:
                    params = optimize_strategy(test_data, strategy_name, config)
                    # Update config with optimized parameters
                    config.update(params)
            else:
                params = optimize_strategy(test_data, strategy_to_optimize, config)
                # Update config with optimized parameters
                config.update(params)
        
        # Create strategy
        strategy = create_strategy(args.strategy, config, model)
        
        # Run backtest on test data
        results, metrics = run_backtest(test_data, strategy, config)
        
        # Print final results
        print("\nFinal Results:")
        print(f"Strategy: {strategy.name}")
        print(f"Total Return: {metrics['total_return_strategy']:.2%}")
        print(f"Annual Return: {metrics['annual_return_strategy']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio_strategy']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown_strategy']:.2%}")
        
        # Compare to market
        print(f"\nMarket Return: {metrics['total_return_market']:.2%}")
        print(f"Market Annual Return: {metrics['annual_return_market']:.2%}")
        print(f"Market Sharpe Ratio: {metrics['sharpe_ratio_market']:.2f}")
        print(f"Market Maximum Drawdown: {metrics['max_drawdown_market']:.2%}")
        
        # Save results to CSV
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        results.to_csv(os.path.join(results_dir, f"{ticker}_{strategy.name}_{timestamp}.csv"))

if __name__ == '__main__':
    main()