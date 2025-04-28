import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.strategy import (
    Strategy, MomentumStrategy, MeanReversionStrategy, 
    MovingAverageCrossoverStrategy, MACDStrategy, EnsembleStrategy, 
    generate_signals, backtest_strategy
)

class TestStrategy(unittest.TestCase):
    
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
        
        # Create config for strategies
        self.config = {
            'momentum_window': 12,
            'momentum_threshold': 0.01,
            'mean_reversion_window': 20,
            'mean_reversion_threshold': 1.5,
            'fast_ma_window': 10,
            'slow_ma_window': 50,
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9
        }
    
    def test_generate_signals(self):
        """Test the generate_signals utility function"""
        predictions = np.array([100, 101, 102, 101, 100])
        signals = generate_signals(predictions, threshold=0.5)
        
        # Expected: first value 0, then positive differences yield 1 and negative yield -1
        expected = np.array([0, 1, 1, -1, -1])
        np.testing.assert_array_equal(signals, expected)
    
    def test_backtest_strategy(self):
        """Test the backtest_strategy utility function"""
        data = pd.DataFrame({
            'Price': [100, 102, 101, 103, 105]
        }, index=pd.date_range(start='2020-01-01', periods=5, freq='D'))
        
        signals = np.array([0, 1, -1, 1, 0])
        result = backtest_strategy(data, signals)
        
        # Check that cumulative return columns are created
        self.assertIn('Cumulative_Market', result.columns)
        self.assertIn('Cumulative_Strategy', result.columns)
        
        # Check that signals were applied
        self.assertTrue((result['Signal'] == signals).all())
        
        # Check that strategy returns were calculated
        self.assertEqual(result['Strategy'].iloc[0], 0)  # First day has no return
        self.assertEqual(result['Strategy'].iloc[1], 0)  # Uses signal from previous day
        
        # Check second day with signal=1 from day 1
        expected_return = result['Return'].iloc[2] * 1
        self.assertEqual(result['Strategy'].iloc[2], expected_return)
    
    def test_momentum_strategy(self):
        """Test MomentumStrategy implementation"""
        strategy = MomentumStrategy(self.config)
        
        # Check that the strategy name is correct
        self.assertEqual(strategy.name, 'Momentum')
        
        # Generate signals and check output
        signals = strategy.generate_signals(self.data)
        
        # Check that signals are the right type and length
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.data))
        
        # Check that signals are valid values (-1, 0, 1)
        unique_signals = signals.unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
    
    def test_mean_reversion_strategy(self):
        """Test MeanReversionStrategy implementation"""
        strategy = MeanReversionStrategy(self.config)
        
        # Check that the strategy name is correct
        self.assertEqual(strategy.name, 'MeanReversion')
        
        # Generate signals and check output
        signals = strategy.generate_signals(self.data)
        
        # Check that signals are the right type and length
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.data))
        
        # Check that signals are valid values (-1, 0, 1)
        unique_signals = signals.unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
    
    def test_moving_average_crossover_strategy(self):
        """Test MovingAverageCrossoverStrategy implementation"""
        strategy = MovingAverageCrossoverStrategy(self.config)
        
        # Check that the strategy name is correct
        self.assertEqual(strategy.name, 'MovingAverageCrossover')
        
        # Generate signals and check output
        signals = strategy.generate_signals(self.data)
        
        # Check that signals are the right type and length
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.data))
        
        # Check that signals are valid values (-1, 0, 1)
        unique_signals = signals.unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
    
    def test_macd_strategy(self):
        """Test MACDStrategy implementation"""
        strategy = MACDStrategy(self.config)
        
        # Check that the strategy name is correct
        self.assertEqual(strategy.name, 'MACD')
        
        # Generate signals and check output
        signals = strategy.generate_signals(self.data)
        
        # Check that signals are the right type and length
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.data))
        
        # Check that signals are valid values (-1, 0, 1)
        unique_signals = signals.unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
    
    def test_ensemble_strategy(self):
        """Test EnsembleStrategy implementation"""
        # Create component strategies
        momentum = MomentumStrategy(self.config)
        mean_reversion = MeanReversionStrategy(self.config)
        
        # Create ensemble with equal weights
        strategies = [momentum, mean_reversion]
        ensemble = EnsembleStrategy(strategies, config=self.config)
        
        # Check that the strategy name is correct
        self.assertEqual(ensemble.name, 'Ensemble')
        
        # Generate signals and check output
        signals = ensemble.generate_signals(self.data)
        
        # Check that signals are the right type and length
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.data))
        
        # Check that signals are valid values (-1, 0, 1)
        unique_signals = signals.unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])
        
        # Create ensemble with custom weights
        weights = {
            'Momentum': 0.7,
            'MeanReversion': 0.3
        }
        weighted_ensemble = EnsembleStrategy(strategies, weights, self.config)
        
        # Generate signals with weighted ensemble
        weighted_signals = weighted_ensemble.generate_signals(self.data)
        
        # Check that signals are valid
        self.assertIsInstance(weighted_signals, pd.Series)
        self.assertEqual(len(weighted_signals), len(self.data))
    
    def test_strategy_backtest(self):
        """Test the backtest method of the Strategy base class"""
        # Create a simple strategy
        strategy = MomentumStrategy(self.config)
        
        # Run backtest
        result = strategy.backtest(self.data)
        
        # Check that result contains expected columns
        expected_columns = [
            'Price', 'High', 'Low', 'Close',
            'Signal', 'Return', 'Strategy',
            'Cumulative_Market', 'Cumulative_Strategy'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that cumulative returns start at 1
        self.assertAlmostEqual(result['Cumulative_Market'].iloc[0], 1.0)
        self.assertAlmostEqual(result['Cumulative_Strategy'].iloc[0], 1.0)
        
        # Check that positions were stored
        self.assertIsNotNone(strategy.positions)
        self.assertEqual(len(strategy.positions), len(self.data))
        
        # Check that history was stored
        self.assertIsNotNone(strategy.history)
        self.assertEqual(len(strategy.history), len(self.data))
    
    def test_apply_risk_management(self):
        """Test the apply_risk_management method of the Strategy base class"""
        # Create a strategy and generate signals
        strategy = MomentumStrategy(self.config)
        signals = strategy.generate_signals(self.data)
        
        # Set up risk parameters
        risk_params = {
            'stop_loss': 0.05,
            'take_profit': 0.1,
            'max_position_size': 0.2
        }
        
        # Apply risk management
        positions = strategy.apply_risk_management(signals, self.data, risk_params)
        
        # Check that positions have the right type and length
        self.assertIsInstance(positions, pd.Series)
        self.assertEqual(len(positions), len(self.data))
        
        # Check that maximum position size is respected
        self.assertLessEqual(positions.max(), risk_params['max_position_size'])
        self.assertGreaterEqual(positions.min(), -risk_params['max_position_size'])

if __name__ == "__main__":
    unittest.main()