import numpy as np
import pandas as pd

class RiskManager:
    """
    Risk management class for controlling position sizes and risk exposure.
    """
    
    def __init__(self, config=None):
        """
        Initialize the risk manager.
        
        Parameters:
            config (dict, optional): Configuration parameters for risk management
        """
        self.config = config or {}
        
        # Default risk parameters
        self.max_position_size = self.config.get('max_position_size', 0.2)  # Maximum 20% of portfolio in a single position
        self.min_position_size = self.config.get('min_position_size', 0.05)  # Minimum 5% of portfolio in a position
        self.max_drawdown = self.config.get('max_drawdown', 0.2)  # Maximum drawdown threshold
        self.stop_loss = self.config.get('stop_loss', 0.05)  # 5% stop loss
        self.take_profit = self.config.get('take_profit', 0.1)  # 10% take profit
        self.risk_aversion = self.config.get('risk_aversion', 0.5)  # Risk aversion coefficient for Kelly criterion
    
    def calculate_position_size(self, signal, volatility, win_rate=None, avg_win=None, avg_loss=None):
        """
        Calculate the optimal position size based on volatility and signal strength.
        
        Parameters:
            signal (float): Trading signal strength (-1 to 1)
            volatility (float): Asset volatility (standard deviation of returns)
            win_rate (float, optional): Historical win rate
            avg_win (float, optional): Average win percentage
            avg_loss (float, optional): Average loss percentage
            
        Returns:
            float: Position size as a percentage of portfolio
        """
        # Base position size on signal strength
        base_size = abs(signal) * self.max_position_size
        
        # Adjust for volatility (reduce position size for higher volatility)
        vol_factor = 0.2 / max(volatility, 0.001)  # Normalize around 20% volatility
        vol_adjusted_size = base_size * min(1.0, vol_factor)
        
        # If win rate and average win/loss are provided, apply Kelly criterion
        if win_rate is not None and avg_win is not None and avg_loss is not None and avg_loss != 0:
            # Kelly fraction = win_rate - (1 - win_rate) / (avg_win / abs(avg_loss))
            kelly_fraction = win_rate - (1 - win_rate) / (avg_win / abs(avg_loss))
            
            # Apply risk aversion factor to Kelly criterion (fractional Kelly)
            kelly_size = max(0, kelly_fraction * self.risk_aversion)
            
            # Final position size as weighted average of volatility-adjusted and Kelly
            position_size = 0.5 * vol_adjusted_size + 0.5 * kelly_size * self.max_position_size
        else:
            position_size = vol_adjusted_size
        
        # Ensure position size is within bounds
        position_size = max(min(position_size, self.max_position_size), self.min_position_size if signal != 0 else 0)
        
        # Adjust the direction based on the signal
        if signal < 0:
            position_size = -position_size
        
        return position_size
    
    def apply_stop_loss(self, position, entry_price, current_price):
        """
        Apply stop loss rule to an existing position.
        
        Parameters:
            position (float): Current position size (positive for long, negative for short)
            entry_price (float): Price at position entry
            current_price (float): Current price
            
        Returns:
            float: Updated position size after applying stop loss
        """
        if position == 0:
            return 0
        
        # Calculate return since entry
        position_return = (current_price / entry_price - 1) * np.sign(position)
        
        # Apply stop loss
        if position_return < -self.stop_loss:
            return 0
        else:
            return position
    
    def apply_take_profit(self, position, entry_price, current_price):
        """
        Apply take profit rule to an existing position.
        
        Parameters:
            position (float): Current position size (positive for long, negative for short)
            entry_price (float): Price at position entry
            current_price (float): Current price
            
        Returns:
            float: Updated position size after applying take profit
        """
        if position == 0:
            return 0
        
        # Calculate return since entry
        position_return = (current_price / entry_price - 1) * np.sign(position)
        
        # Apply take profit
        if position_return > self.take_profit:
            return 0
        else:
            return position
    
    def apply_drawdown_control(self, position, portfolio_value, peak_value):
        """
        Reduce position size if portfolio drawdown exceeds threshold.
        
        Parameters:
            position (float): Current position size
            portfolio_value (float): Current portfolio value
            peak_value (float): Peak portfolio value
            
        Returns:
            float: Updated position size after applying drawdown control
        """
        # Calculate current drawdown
        drawdown = 1 - portfolio_value / peak_value
        
        # If drawdown exceeds threshold, reduce position size
        if drawdown > self.max_drawdown:
            reduction_factor = 1 - (drawdown - self.max_drawdown) / 0.1  # Linear reduction over next 10%
            reduction_factor = max(0, min(1, reduction_factor))  # Ensure between 0 and 1
            return position * reduction_factor
        else:
            return position
    
    def apply_risk_management(self, signals, prices, volatility=None, portfolio_values=None):
        """
        Apply all risk management rules to a series of signals.
        
        Parameters:
            signals (pd.Series): Raw trading signals
            prices (pd.Series): Asset prices
            volatility (pd.Series, optional): Asset volatility
            portfolio_values (pd.Series, optional): Portfolio values
            
        Returns:
            pd.Series: Risk-adjusted position sizes
        """
        # Initialize position sizes
        position_sizes = pd.Series(0.0, index=signals.index)
        
        # If volatility not provided, calculate it
        if volatility is None:
            returns = prices.pct_change()
            volatility = returns.rolling(window=21).std()
        
        # Apply position sizing
        for i in range(len(signals)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Calculate position size
            position_sizes.iloc[i] = self.calculate_position_size(
                signals.iloc[i],
                volatility.iloc[i] if i < len(volatility) else 0.1  # Use default if not available
            )
        
        # Apply stop loss and take profit
        if len(position_sizes) > 1:
            for i in range(1, len(position_sizes)):
                if position_sizes.iloc[i-1] != 0:
                    # Find entry price (first non-zero position)
                    entry_idx = i-1
                    while entry_idx > 0 and position_sizes.iloc[entry_idx-1] == position_sizes.iloc[entry_idx]:
                        entry_idx -= 1
                    
                    entry_price = prices.iloc[entry_idx]
                    current_price = prices.iloc[i]
                    
                    # Apply stop loss
                    position_sizes.iloc[i] = self.apply_stop_loss(
                        position_sizes.iloc[i-1],
                        entry_price,
                        current_price
                    )
                    
                    # Apply take profit
                    position_sizes.iloc[i] = self.apply_take_profit(
                        position_sizes.iloc[i],
                        entry_price,
                        current_price
                    )
        
        # Apply drawdown control if portfolio values provided
        if portfolio_values is not None:
            peak_values = portfolio_values.cummax()
            
            for i in range(1, len(position_sizes)):
                position_sizes.iloc[i] = self.apply_drawdown_control(
                    position_sizes.iloc[i],
                    portfolio_values.iloc[i],
                    peak_values.iloc[i]
                )
        
        return position_sizes