import unittest
import pandas as pd
import numpy as np
from src.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, momentum,
    mean_reversion_signal, momentum_signal, dual_moving_average_signal,
    macd_signal, combine_signals
)

class TestIndicators(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create synthetic price data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create a price trend with a pattern: up-down-up-down
        base_price = 100
        trend = np.concatenate([
            np.linspace(0, 15, 25),   # Up trend
            np.linspace(15, 0, 25),   # Down trend
            np.linspace(0, 10, 25),   # Up trend
            np.linspace(10, 5, 25)    # Down trend
        ])
        
        # Add noise to the price
        noise = np.random.normal(0, 1, 100)
        prices = base_price + trend + noise
        
        # Create the DataFrame
        self.data = pd.DataFrame({
            'Price': prices,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices
        }, index=dates)
        
        # Create series for tests
        self.price_series = self.data['Price']
    
    def test_sma(self):
        """Test Simple Moving Average"""
        # Test with window of 10
        result = sma(self.price_series, window=10)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct (window-1 NaN values at the beginning)
        self.assertEqual(len(result), len(self.price_series))
        self.assertEqual(result.isna().sum(), 9)  # 9 NaN values for window=10
        
        # Verify SMA calculation for a few points
        for i in range(10, 15):
            expected = self.price_series.iloc[i-10:i].mean()
            self.assertAlmostEqual(result.iloc[i], expected)
    
    def test_ema(self):
        """Test Exponential Moving Average"""
        # Test with window of 10
        result = ema(self.price_series, window=10)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Check that the first window-1 values are NaN
        self.assertEqual(result.iloc[:9].isna().sum(), 9)
        
        # Verify that EMA exists for the rest of the array
        self.assertEqual(result.iloc[9:].isna().sum(), 0)
    
    def test_rsi(self):
        """Test Relative Strength Index"""
        # Test with window of 14
        result = rsi(self.price_series, window=14)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Check that values are in the correct range (0-100)
        valid_values = result.dropna()
        self.assertTrue(all(valid_values >= 0))
        self.assertTrue(all(valid_values <= 100))
    
    def test_macd(self):
        """Test Moving Average Convergence Divergence"""
        # Test with default parameters
        result = macd(self.price_series)
        
        # Check that result is a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that it has the expected columns
        self.assertIn('MACD', result.columns)
        self.assertIn('Signal', result.columns)
        self.assertIn('Histogram', result.columns)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Check that Histogram is correctly calculated as MACD - Signal
        np.testing.assert_almost_equal(
            result['Histogram'].dropna().values,
            (result['MACD'] - result['Signal']).dropna().values
        )
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands"""
        # Test with default parameters
        result = bollinger_bands(self.price_series)
        
        # Check that result is a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that it has the expected columns
        self.assertIn('Middle', result.columns)
        self.assertIn('Upper', result.columns)
        self.assertIn('Lower', result.columns)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Verify that Upper > Middle > Lower for non-NaN values
        valid_indices = result.dropna().index
        self.assertTrue(all(result.loc[valid_indices, 'Upper'] > result.loc[valid_indices, 'Middle']))
        self.assertTrue(all(result.loc[valid_indices, 'Middle'] > result.loc[valid_indices, 'Lower']))
    
    def test_atr(self):
        """Test Average True Range"""
        # Test with default parameters
        result = atr(self.data)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.data))
        
        # Check that values are non-negative (ATR is always positive)
        self.assertTrue(all(result.dropna() >= 0))
    
    def test_momentum(self):
        """Test Momentum indicator"""
        # Test with window of 10
        result = momentum(self.price_series, window=10)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Verify momentum calculation for a few points
        for i in range(10, 15):
            expected = self.price_series.iloc[i] / self.price_series.iloc[i-10] - 1
            self.assertAlmostEqual(result.iloc[i], expected)
    
    def test_mean_reversion_signal(self):
        """Test Mean Reversion Signal generator"""
        # Test with default parameters
        result = mean_reversion_signal(self.price_series)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Check that signals are in {-1, 0, 1}
        unique_values = result.dropna().unique()
        for value in unique_values:
            self.assertIn(value, [-1, 0, 1])
    
    def test_momentum_signal(self):
        """Test Momentum Signal generator"""
        # Test with default parameters
        result = momentum_signal(self.price_series)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Check that signals are in {-1, 0, 1}
        unique_values = result.dropna().unique()
        for value in unique_values:
            self.assertIn(value, [-1, 0, 1])
    
    def test_dual_moving_average_signal(self):
        """Test Dual Moving Average Signal generator"""
        # Test with smaller windows for testing
        result = dual_moving_average_signal(self.price_series, fast_window=5, slow_window=10)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Check that signals are in {-1, 0, 1}
        unique_values = result.dropna().unique()
        for value in unique_values:
            self.assertIn(value, [-1, 0, 1])
    
    def test_macd_signal(self):
        """Test MACD Signal generator"""
        # Test with default parameters
        result = macd_signal(self.price_series)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that the length is correct
        self.assertEqual(len(result), len(self.price_series))
        
        # Check that signals are in {-1, 0, 1}
        unique_values = result.dropna().unique()
        for value in unique_values:
            self.assertIn(value, [-1, 0, 1])
    
    def test_combine_signals(self):
        """Test signal combination function"""
        # Create test signals
        signal1 = pd.Series([1, 0, -1, 0, 1], index=range(5))
        signal2 = pd.Series([0, 1, 0, -1, -1], index=range(5))
        signals_dict = {
            'signal1': signal1,
            'signal2': signal2
        }
        
        # Test with equal weights
        equal_result = combine_signals(signals_dict)
        self.assertIsInstance(equal_result, pd.Series)
        self.assertEqual(len(equal_result), 5)
        
        # Test with custom weights
        weights = {
            'signal1': 0.7,
            'signal2': 0.3
        }
        weighted_result = combine_signals(signals_dict, weights)
        self.assertIsInstance(weighted_result, pd.Series)
        self.assertEqual(len(weighted_result), 5)
        
        # Expected results for weighted case
        # signal1: [1, 0, -1, 0, 1] * 0.7 = [0.7, 0, -0.7, 0, 0.7]
        # signal2: [0, 1, 0, -1, -1] * 0.3 = [0, 0.3, 0, -0.3, -0.3]
        # Combined: [0.7, 0.3, -0.7, -0.3, 0.4]
        # After discretization: [1, 1, -1, -1, 1]
        expected = pd.Series([1, 1, -1, -1, 1], index=range(5))
        pd.testing.assert_series_equal(weighted_result, expected)

if __name__ == "__main__":
    unittest.main()