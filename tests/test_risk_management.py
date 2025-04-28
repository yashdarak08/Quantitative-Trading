import unittest
import pandas as pd
import numpy as np
from src.risk_management import RiskManager

class TestRiskManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and objects"""
        # Create config with specific risk parameters
        self.config = {
            'max_position_size': 0.2,
            'min_position_size': 0.05,
            'max_drawdown': 0.2,
            'stop_loss': 0.05,
            'take_profit': 0.1,
            'risk_aversion': 0.5
        }
        
        # Create risk manager
        self.risk_manager = RiskManager(self.config)
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, 100))
        
        # Create signals
        signals = pd.Series(0, index=dates)
        for i in range(0, 100, 10):  # Every 10 days, alternate signals
            signals.iloc[i] = 1 if (i/10) % 2 == 0 else -1
        
        # Create volatility series (21-day rolling standard deviation of returns)
        returns = pd.Series(np.diff(np.log(prices), prepend=np.log(prices[0])), index=dates)
        volatility = returns.rolling(21).std()
        
        # Create portfolio values (starting at 10000 and applying returns)
        portfolio_values = 10000 * (1 + returns).cumprod()
        
        self.dates = dates
        self.prices = pd.Series(prices, index=dates)
        self.signals = signals
        self.volatility = volatility
        self.portfolio_values = portfolio_values
    
    def test_initialization(self):
        """Test if the RiskManager initializes correctly"""
        self.assertEqual(self.risk_manager.max_position_size, 0.2)
        self.assertEqual(self.risk_manager.min_position_size, 0.05)
        self.assertEqual(self.risk_manager.max_drawdown, 0.2)
        self.assertEqual(self.risk_manager.stop_loss, 0.05)
        self.assertEqual(self.risk_manager.take_profit, 0.1)
        self.assertEqual(self.risk_manager.risk_aversion, 0.5)
    
    def test_calculate_position_size(self):
        """Test position size calculation based on signal and volatility"""
        # Test with positive signal
        pos_size = self.risk_manager.calculate_position_size(
            signal=0.5, 
            volatility=0.05
        )
        
        # Check that position is positive and within bounds
        self.assertGreater(pos_size, 0)
        self.assertLessEqual(pos_size, self.risk_manager.max_position_size)
        self.assertGreaterEqual(pos_size, self.risk_manager.min_position_size)
        
        # Test with negative signal
        neg_size = self.risk_manager.calculate_position_size(
            signal=-0.5, 
            volatility=0.05
        )
        
        # Check that position is negative and within bounds
        self.assertLess(neg_size, 0)
        self.assertGreaterEqual(neg_size, -self.risk_manager.max_position_size)
        self.assertLessEqual(neg_size, -self.risk_manager.min_position_size)
        
        # Test with zero signal
        zero_size = self.risk_manager.calculate_position_size(
            signal=0, 
            volatility=0.05
        )
        
        # Check that position is zero
        self.assertEqual(zero_size, 0)
        
        # Test Kelly criterion calculation
        kelly_size = self.risk_manager.calculate_position_size(
            signal=0.5, 
            volatility=0.05,
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03
        )
        
        # Check that Kelly size is reasonable
        self.assertGreater(kelly_size, 0)
        self.assertLessEqual(kelly_size, self.risk_manager.max_position_size)
    
    def test_apply_stop_loss(self):
        """Test stop loss application logic"""
        # Test case 1: No position (should return 0)
        result = self.risk_manager.apply_stop_loss(0, 100, 105)
        self.assertEqual(result, 0)
        
        # Test case 2: Long position with price above entry (no stop loss)
        result = self.risk_manager.apply_stop_loss(0.1, 100, 105)
        self.assertEqual(result, 0.1)
        
        # Test case 3: Long position with small loss (no stop loss)
        result = self.risk_manager.apply_stop_loss(0.1, 100, 97)  # 3% loss
        self.assertEqual(result, 0.1)
        
        # Test case 4: Long position with loss exceeding stop loss
        result = self.risk_manager.apply_stop_loss(0.1, 100, 94)  # 6% loss
        self.assertEqual(result, 0)
        
        # Test case 5: Short position with price below entry (no stop loss)
        result = self.risk_manager.apply_stop_loss(-0.1, 100, 95)
        self.assertEqual(result, -0.1)
        
        # Test case 6: Short position with small loss (no stop loss)
        result = self.risk_manager.apply_stop_loss(-0.1, 100, 103)  # 3% loss for short
        self.assertEqual(result, -0.1)
        
        # Test case 7: Short position with loss exceeding stop loss
        result = self.risk_manager.apply_stop_loss(-0.1, 100, 106)  # 6% loss for short
        self.assertEqual(result, 0)
    
    def test_apply_take_profit(self):
        """Test take profit application logic"""
        # Test case 1: No position (should return 0)
        result = self.risk_manager.apply_take_profit(0, 100, 105)
        self.assertEqual(result, 0)
        
        # Test case 2: Long position with small gain (no take profit)
        result = self.risk_manager.apply_take_profit(0.1, 100, 105)  # 5% gain
        self.assertEqual(result, 0.1)
        
        # Test case 3: Long position with gain exceeding take profit
        result = self.risk_manager.apply_take_profit(0.1, 100, 112)  # 12% gain
        self.assertEqual(result, 0)
        
        # Test case 4: Short position with small gain (no take profit)
        result = self.risk_manager.apply_take_profit(-0.1, 100, 95)  # 5% gain for short
        self.assertEqual(result, -0.1)
        
        # Test case 5: Short position with gain exceeding take profit
        result = self.risk_manager.apply_take_profit(-0.1, 100, 89)  # 11% gain for short
        self.assertEqual(result, 0)
    
    def test_apply_drawdown_control(self):
        """Test drawdown control logic"""
        # Test case 1: No position (should return 0)
        result = self.risk_manager.apply_drawdown_control(0, 9000, 10000)
        self.assertEqual(result, 0)
        
        # Test case 2: Small drawdown (no reduction)
        result = self.risk_manager.apply_drawdown_control(0.1, 9000, 10000)  # 10% drawdown
        self.assertEqual(result, 0.1)
        
        # Test case 3: Drawdown near threshold
        result = self.risk_manager.apply_drawdown_control(0.1, 8100, 10000)  # 19% drawdown
        self.assertAlmostEqual(result, 0.1)
        
        # Test case 4: Drawdown exceeding threshold
        result = self.risk_manager.apply_drawdown_control(0.1, 7500, 10000)  # 25% drawdown
        self.assertLess(result, 0.1)  # Position should be reduced
        self.assertGreater(result, 0)  # But not to zero
        
        # Test case 5: Extreme drawdown
        result = self.risk_manager.apply_drawdown_control(0.1, 5000, 10000)  # 50% drawdown
        self.assertEqual(result, 0)  # Position should be reduced to zero
    
    def test_apply_risk_management(self):
        """Test the complete risk management pipeline"""
        # Apply risk management to our test data
        position_sizes = self.risk_manager.apply_risk_management(
            self.signals, 
            self.prices, 
            self.volatility,
            self.portfolio_values
        )
        
        # Check that we have position sizes for all dates
        self.assertEqual(len(position_sizes), len(self.dates))
        
        # Check that positions are within bounds
        max_abs_position = max(abs(position_sizes))
        self.assertLessEqual(max_abs_position, self.risk_manager.max_position_size)
        
        # Check that there are some non-zero positions
        self.assertGreater((position_sizes != 0).sum(), 0)
        
        # Check that positions align with signals direction
        for i in range(len(self.signals)):
            if self.signals.iloc[i] > 0:
                self.assertGreaterEqual(position_sizes.iloc[i], 0)
            elif self.signals.iloc[i] < 0:
                self.assertLessEqual(position_sizes.iloc[i], 0)

if __name__ == "__main__":
    unittest.main()