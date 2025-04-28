import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project path for imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules from the project
from src.data_loader import load_data
from src.strategy import (
    MomentumStrategy, MeanReversionStrategy, MovingAverageCrossoverStrategy, 
    MACDStrategy, EnsembleStrategy
)
from src.backtest import BacktestEngine

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantitative Trading System')
    
    parser.add_argument('--config', default='config/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--ticker', default=None,
                        help='Ticker symbol to use (overrides config)')
    parser.add_argument('--strategy', default='ensemble', 
                        choices=['momentum', 'meanreversion', 'movingaverage', 'macd', 'ensemble'],
                        help='Trading strategy to use')
    parser.add_argument('--start', default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.ticker:
        config['data']['tickers'] = [args.ticker]
    if args.start:
        config['data']['start_date'] = args.start
    if args.end:
        config['data']['end_date'] = args.end
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each ticker
    for ticker in config['data']['tickers']:
        print(f"\nProcessing ticker: {ticker}")
        
        # Load data
        print(f"Loading data for {ticker} from {config['data']['start_date']} to {config['data']['end_date']}")
        try:
            data = load_data(
                ticker, 
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            print(f"Loaded {len(data)} rows of data")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            continue
        
        # Split data for training/testing (80/20)
        split_index = int(0.8 * len(data))
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        
        print(f"Training data: {train_data.shape[0]} rows ({train_data.index[0]} to {train_data.index[-1]})")
        print(f"Testing data: {test_data.shape[0]} rows ({test_data.index[0]} to {test_data.index[-1]})")
        
        # Create strategy
        print(f"Creating {args.strategy} strategy")
        if args.strategy == 'momentum':
            strategy = MomentumStrategy(config)
        elif args.strategy == 'meanreversion':
            strategy = MeanReversionStrategy(config)
        elif args.strategy == 'movingaverage':
            strategy = MovingAverageCrossoverStrategy(config)
        elif args.strategy == 'macd':
            strategy = MACDStrategy(config)
        elif args.strategy == 'ensemble':
            # Create an ensemble of all strategies
            strategies = [
                MomentumStrategy(config),
                MeanReversionStrategy(config),
                MovingAverageCrossoverStrategy(config),
                MACDStrategy(config)
            ]
            # Equal weighting for all strategies
            weights = {strategy.name: 1/len(strategies) for strategy in strategies}
            strategy = EnsembleStrategy(strategies, weights, config)
        
        # Run backtest
        print("Running backtest...")
        backtest = BacktestEngine(test_data, strategy, config)
        backtest.run()
        
        # Print performance summary
        print(backtest.get_performance_summary())
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output, f"{ticker}_{strategy.name}_{timestamp}.csv")
        backtest.portfolio.to_csv(results_path)
        print(f"Results saved to {results_path}")
        
        # Generate and save performance plot
        plt.figure(figsize=(12, 6))
        plt.plot(backtest.portfolio.index, backtest.portfolio['Cumulative_Market'], 'b-', label='Market')
        plt.plot(backtest.portfolio.index, backtest.portfolio['Cumulative_Strategy'], 'g-', label='Strategy')
        plt.title(f'{ticker}: {strategy.name} Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(args.output, f"{ticker}_{strategy.name}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Performance plot saved to {plot_path}")

if __name__ == "__main__":
    main()