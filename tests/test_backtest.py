import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.backtest import BacktestEngine
from src.strategy import MomentumStrategy, Strategy

class MockStrategy(Strategy):
    """Mock strategy for testing"""
    def __init__(self, config=None):
        super().__init__('MockStrategy', config)
    
    def generate_signals(self, data):
        """Generate simple alternating signals for testing"""
        signals = pd.Series(0, index=data.index)
        for i in range(len(signals)):
            if i % 4 == 0:
                signals.iloc[i] = 1
            elif i % 4 == 2:
                signals.iloc[i] = -1
        return signals

class TestBacktestEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and objects"""
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        prices = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)  # Trending with noise
        self.data = pd.DataFrame({
            'Price': prices,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices
        }, index=dates)
        
        # Create strategy and config
        self.config = {
            'transaction_cost': 0.001,
            'slippage': 0.001,
            'risk_free_rate': 0.02,
            'stop_loss': 0.05,
            'take_profit': 0.1
        }
        self.strategy = MockStrategy(self.config)
        
        # Create backtest engine
        self.backtest = BacktestEngine(self.data, self.strategy, self.config)
    
    def test_initialization(self):
        """Test if the backtest engine initializes correctly"""
        self.assertEqual(self.backtest.transaction_cost, 0.001)
        self.assertEqual(self.backtest.slippage, 0.001)
        self.assertEqual(self.backtest.risk_free_rate, 0.02)
        self.assertEqual(self.backtest.strategy.name, 'MockStrategy')
    
    def test_run(self):
        """Test if backtest runs and returns expected dataframe"""
        result = self.backtest.run()
        
        # Check if all expected columns are created
        expected_columns = [
            'Price', 'High', 'Low', 'Close', 'Signal', 'Position', 
            'Return', 'Transaction', 'Slippage', 'Cost', 
            'Strategy_Gross', 'Strategy_Net', 
            'Cumulative_Market', 'Cumulative_Strategy'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check if there are positions in the result
        self.assertTrue((result['Position'] != 0).any())
        
        # Check if cumulative returns exist and start at 1
        self.assertAlmostEqual(result['Cumulative_Market'].iloc[0], 1.0)
        self.assertAlmostEqual(result['Cumulative_Strategy'].iloc[0], 1.0)
    
    def test_record_trades(self):
        """Test if trades are recorded correctly"""
        self.backtest.run()
        
        # Check if trades were recorded
        self.assertTrue(len(self.backtest.trades) > 0)
        
        # Check trade record structure
        first_trade = self.backtest.trades[0]
        self.assertIn('entry_date', first_trade)
        self.assertIn('entry_price', first_trade)
        self.assertIn('position', first_trade)
        self.assertIn('exit_date', first_trade)
        self.assertIn('exit_price', first_trade)
        self.assertIn('profit_loss', first_trade)
        self.assertIn('duration', first_trade)
    
    def test_calculate_performance_metrics(self):
        """Test if performance metrics are calculated correctly"""
        self.backtest.run()
        metrics = self.backtest.performance_metrics
        
        # Check if key metrics exist
        expected_metrics = [
            'total_return_strategy', 'annual_return_strategy', 
            'volatility_strategy', 'sharpe_ratio_strategy',
            'max_drawdown_strategy', 'win_rate', 'total_trades'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Basic sanity checks on metrics
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        self.assertEqual(metrics['total_trades'], len(self.backtest.trades))
    
    def test_get_performance_summary(self):
        """Test if performance summary is generated correctly"""
        self.backtest.run()
        summary = self.backtest.get_performance_summary()
        
        # Check that summary contains key information
        self.assertIn('PERFORMANCE SUMMARY', summary)
        self.assertIn(self.strategy.name, summary)
        self.assertIn('RETURNS:', summary)
        self.assertIn('RISK METRICS:', summary)
        self.assertIn('RISK-ADJUSTED PERFORMANCE:', summary)
        self.assertIn('TRADE STATISTICS:', summary)

if __name__ == "__main__":
    unittest.main()